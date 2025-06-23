"""
Time Series Forecasting com TinyTimeMixer (TTM) - IBM Granite

Este módulo implementa um modelo de previsão de séries temporais utilizando o TinyTimeMixer
da IBM Granite, aplicado para previsão de vendas e faturamento por produto e UF.

Instalação das dependências:
pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.22"
pip install transformers torch accelerate bitsandbytes pandas scikit-learn gdown peft

Autor: [Seu Nome]
Data: [Data Atual]
Versão: 1.0
"""

import pandas as pd
import torch
import numpy as np
import math
import os

# Importações específicas da biblioteca Hugging Face e TSFM para treinamento
from transformers import (
    BitsAndBytesConfig,      # Configuração para quantização de bits
    TrainingArguments,       # Argumentos de treinamento do modelo
    Trainer,                 # Classe principal para treinamento
    EarlyStoppingCallback,   # Callback para parada antecipada
)
from torch.optim import AdamW                    # Otimizador AdamW
from torch.optim.lr_scheduler import OneCycleLR  # Scheduler de learning rate
from peft import LoraConfig, get_peft_model      # Para fine-tuning eficiente

# Componentes da biblioteca IBM Granite TSFM para processamento de séries temporais
from tsfm_public import (
    TimeSeriesPreprocessor,      # Preprocessador de séries temporais
    TinyTimeMixerForPrediction,  # Modelo TTM para previsão
    ForecastDFDataset,           # Dataset personalizado para previsão
    TrackingCallback,            # Callback para tracking de métricas
)
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig

# =============================================================================
# SEÇÃO 1: CARREGAMENTO E CONFIGURAÇÃO INICIAL DOS DADOS
# =============================================================================

def load_data():
    """
    Carrega os dados de vendas e faturamento do arquivo CSV.
    
    Se o arquivo não for encontrado, cria um DataFrame de exemplo para demonstração
    com dados sintéticos de vendas e faturamento por produto e UF.
    
    Returns:
        pd.DataFrame: DataFrame com colunas date, produto_cat, uf_cat, vendas, faturamento
    """
    try:
        # Tenta carregar o arquivo de dados real
        df = pd.read_csv("./dados/db_tratado-w.csv", parse_dates=["date"])
        print("Dados carregados do arquivo db_tratado-w.csv")
        return df
    except FileNotFoundError:
        print("Erro: Arquivo db_tratado-w.csv não encontrado. Criando DataFrame de exemplo...")
        
        # Criar dados de exemplo para demonstração
        data_example = {
            'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29', '2023-02-05',
                                    '2023-02-12', '2023-02-19', '2023-02-26', '2023-03-05', '2023-03-12', '2023-03-19']),
            'produto_cat': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Categoria do produto (codificada)
            'uf_cat': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],        # UF categorizada (codificada)
            'vendas': [300, 310, 305, 320, 315, 330, 325, 340, 335, 350, 345, 360],      # Volume de vendas
            'faturamento': [1000.0, 1100.0, 1050.0, 1200.0, 1150.0, 1300.0, 1250.0,     # Valor do faturamento
                           1400.0, 1350.0, 1500.0, 1450.0, 1600.0]
        }
        df_single_product = pd.DataFrame(data_example)
        
        # Simular múltiplos produtos (1-19) e UFs (1-4) para ter um dataset mais robusto
        df = pd.concat([df_single_product.copy().assign(produto_cat=i, uf_cat=j)
                        for i in range(1, 20) for j in range(1, 5)], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        print("DataFrame de exemplo criado com 19 produtos x 4 UFs = 76 combinações")
        return df

# Carregar dados
df = load_data()

print("\nInformações do DataFrame carregado:")
print(df.info())
print("\nPrimeiras linhas do DataFrame:")
print(df.head())

# =============================================================================
# CONFIGURAÇÕES PRINCIPAIS DO MODELO E DATASET
# =============================================================================

# Configurações das colunas do dataset
TIMESTAMP_COLUMN = 'date'                          # Coluna de timestamp (data)
ID_COLUMNS = ['produto_cat', 'uf_cat']             # Colunas identificadoras das séries
TARGET_COLUMNS = ['vendas', 'faturamento']         # Variáveis alvo para previsão
STATIC_CATEGORICAL_COLUMNS = ['uf_cat']            # Variáveis categóricas estáticas
CONTROL_COLUMNS = []                               # Variáveis de controle (vazio neste caso)

# Configurações temporais do modelo
CONTEXT_LENGTH = 104    # Janela de contexto: 2 anos de histórico semanal (52*2=104)
PREDICTION_LENGTH = 26  # Horizonte de previsão: 6 meses semanais (~26 semanas)

# Configuração automática do dispositivo de processamento
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDispositivo de processamento: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
    print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# SEÇÃO 2: PRÉ-PROCESSAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

def resample_data_to_weekly(df, timestamp_col, id_cols):
    """
    Resampling dos dados para frequência semanal consistente.
    
    Agrupa por combinações produto-UF e aplica resampling semanal,
    preenchendo valores faltantes com forward fill.
    
    Args:
        df (pd.DataFrame): DataFrame original
        timestamp_col (str): Nome da coluna de timestamp
        id_cols (list): Lista das colunas identificadoras
    
    Returns:
        pd.DataFrame: DataFrame com frequência semanal consistente
    """
    print("Aplicando resampling para frequência semanal...")
    
    # Ordenar dados por timestamp e IDs para consistência
    df = df.sort_values([timestamp_col] + id_cols).reset_index(drop=True)
    
    df_resampled = []
    total_groups = df.groupby(id_cols).ngroups
    processed_groups = 0
    
    # Processar cada combinação produto-UF separadamente
    for group_keys, group_df in df.groupby(id_cols):
        # Definir timestamp como índice para resampling
        group_df = group_df.set_index(timestamp_col)
        
        # Aplicar resampling semanal (W) pegando último valor de cada semana
        group_df = group_df.resample('W').last().ffill()
        
        # Resetar índice para voltar timestamp como coluna
        group_df = group_df.reset_index()
        
        # Restaurar colunas identificadoras (produto_cat, uf_cat)
        for i, col in enumerate(id_cols):
            group_df[col] = group_keys[i] if isinstance(group_keys, tuple) else group_keys
        
        df_resampled.append(group_df)
        processed_groups += 1
        
        # Log de progresso a cada 10 grupos processados
        if processed_groups % 10 == 0:
            print(f"Processados {processed_groups}/{total_groups} grupos produto-UF")
    
    return pd.concat(df_resampled, ignore_index=True)

# Aplicar resampling nos dados
df = resample_data_to_weekly(df, TIMESTAMP_COLUMN, ID_COLUMNS)
print(f"Resampling concluído. Dataset final: {len(df)} registros")

def create_time_series_preprocessor():
    """
    Cria e configura o preprocessador de séries temporais TTM.
    
    Returns:
        TimeSeriesPreprocessor: Preprocessador configurado para o dataset
    """
    print("Configurando preprocessador de séries temporais...")
    
    return TimeSeriesPreprocessor(
        id_columns=ID_COLUMNS,                      # Colunas identificadoras das séries
        timestamp_column=TIMESTAMP_COLUMN,          # Coluna de timestamp
        target_columns=TARGET_COLUMNS,              # Variáveis alvo para previsão
        static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,  # Variáveis categóricas
        scaling_id_columns=ID_COLUMNS,              # Colunas para escalonamento por grupo
        context_length=CONTEXT_LENGTH,              # Tamanho da janela de contexto
        prediction_length=PREDICTION_LENGTH,        # Horizonte de previsão
        scaling=True,                              # Aplicar normalização/escalonamento
        scaler_type="standard",                    # Tipo de escalonamento (StandardScaler)
        encode_categorical=True,                   # Codificar variáveis categóricas
        control_columns=CONTROL_COLUMNS,           # Variáveis de controle externas
        observable_columns=[],                     # Variáveis observáveis durante previsão
        freq='W'                                   # Frequência dos dados (semanal)
    )

# Criar e treinar preprocessador
tsp = create_time_series_preprocessor()
print("Treinando preprocessador com todos os dados...")
trained_tsp = tsp.train(df)

# Aplicar preprocessamento aos dados
print("Aplicando transformações de preprocessamento...")
df_processed = trained_tsp.preprocess(df)
print(f"Preprocessamento concluído. Shape dos dados processados: {df_processed.shape}")

# =============================================================================
# SEÇÃO 3: DIVISÃO DOS DADOS E CRIAÇÃO DOS DATASETS
# =============================================================================

def split_data_by_combinations(df_processed, id_columns, train_frac=0.7, val_frac=0.15, random_state=42):
    """
    Divide os dados preservando combinações produto-UF inteiras.
    
    Evita vazamento de dados garantindo que uma combinação produto-UF
    não apareça em múltiplos conjuntos (treino/validação/teste).
    
    Args:
        df_processed (pd.DataFrame): Dados preprocessados
        id_columns (list): Colunas identificadoras para preservar
        train_frac (float): Fração para treinamento (padrão: 0.7)
        val_frac (float): Fração para validação (padrão: 0.15)
        random_state (int): Seed para reprodutibilidade
    
    Returns:
        tuple: (train_data, valid_data, test_data)
    """
    print("Dividindo dados preservando combinações produto-UF...")
    
    # Obter combinações únicas de produto-UF
    unique_combinations = df_processed[id_columns].drop_duplicates()
    total_combinations = len(unique_combinations)
    
    # Divisão estratificada das combinações
    train_combinations = unique_combinations.sample(frac=train_frac, random_state=random_state)
    remaining = unique_combinations.drop(train_combinations.index)
    
    # Da parte restante, dividir entre validação e teste
    val_frac_remaining = val_frac / (1 - train_frac)  # Ajustar fração para o restante
    valid_combinations = remaining.sample(frac=val_frac_remaining, random_state=random_state)
    test_combinations = remaining.drop(valid_combinations.index)
    
    # Filtrar dados processados baseado nas combinações
    train_data = df_processed.merge(train_combinations, on=id_columns)
    valid_data = df_processed.merge(valid_combinations, on=id_columns)
    test_data = df_processed.merge(test_combinations, on=id_columns)
    
    print(f"Divisão concluída:")
    print(f"  Treino: {len(train_combinations):>3} combinações ({len(train_data):>4} registros)")
    print(f"  Validação: {len(valid_combinations):>3} combinações ({len(valid_data):>4} registros)")
    print(f"  Teste: {len(test_combinations):>3} combinações ({len(test_data):>4} registros)")
    print(f"  Total: {total_combinations} combinações únicas")
    
    return train_data, valid_data, test_data

# Aplicar divisão dos dados
train_data, valid_data, test_data = split_data_by_combinations(df_processed, ID_COLUMNS)

def create_forecast_datasets(train_data, valid_data, test_data, trained_tsp):
    """
    Cria os datasets específicos para o modelo TTM de previsão.
    
    Args:
        train_data, valid_data, test_data (pd.DataFrame): Dados divididos
        trained_tsp (TimeSeriesPreprocessor): Preprocessador treinado
    
    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    print("Criando datasets TTM para treinamento...")
    
    # Obter frequency token do preprocessador
    freq_token = trained_tsp.get_frequency_token(trained_tsp.freq)
    print(f"Frequency token: {freq_token}")
    
    # Configuração comum para todos os datasets
    dataset_config = {
        'id_columns': ID_COLUMNS,
        'timestamp_column': TIMESTAMP_COLUMN,
        'target_columns': TARGET_COLUMNS,
        'control_columns': CONTROL_COLUMNS,
        'static_categorical_columns': STATIC_CATEGORICAL_COLUMNS,
        'context_length': CONTEXT_LENGTH,
        'prediction_length': PREDICTION_LENGTH,
        'frequency_token': freq_token
    }
    
    # Criar datasets para cada divisão
    train_dataset = ForecastDFDataset(train_data, **dataset_config)
    valid_dataset = ForecastDFDataset(valid_data, **dataset_config)
    test_dataset = ForecastDFDataset(test_data, **dataset_config)
    
    print(f"Datasets TTM criados:")
    print(f"  Treino: {len(train_dataset):>4} amostras")
    print(f"  Validação: {len(valid_dataset):>4} amostras")
    print(f"  Teste: {len(test_dataset):>4} amostras")
    
    return train_dataset, valid_dataset, test_dataset

# Criar datasets TTM
train_dataset, valid_dataset, test_dataset = create_forecast_datasets(
    train_data, valid_data, test_data, trained_tsp
)

# =============================================================================
# SEÇÃO 4: CONFIGURAÇÃO E INICIALIZAÇÃO DO MODELO TTM
# =============================================================================

def create_ttm_config(trained_tsp):
    """
    Cria configuração personalizada para o modelo TinyTimeMixer.
    
    Adapta o modelo pré-treinado para o dataset específico,
    configurando canais de entrada, saída e variáveis categóricas.
    
    Args:
        trained_tsp (TimeSeriesPreprocessor): Preprocessador treinado
    
    Returns:
        TinyTimeMixerConfig: Configuração do modelo TTM
    """
    print("Configurando modelo TinyTimeMixer...")
    
    config = TinyTimeMixerConfig(
        context_length=CONTEXT_LENGTH,                      # Janela de contexto histórico
        prediction_length=PREDICTION_LENGTH,                # Horizonte de previsão
        num_input_channels=trained_tsp.num_input_channels,  # Número de canais de entrada
        prediction_channel_indices=trained_tsp.prediction_channel_indices,  # Índices dos canais alvo
        exogenous_channel_indices=trained_tsp.exogenous_channel_indices,    # Índices de variáveis exógenas
        decoder_mode="mix_channel",                         # Modo de decodificação (mix de canais)
        categorical_vocab_size_list=trained_tsp.categorical_vocab_size_list,  # Tamanhos dos vocabulários categóricos
    )
    
    print(f"Configuração TTM:")
    print(f"  Context Length: {config.context_length}")
    print(f"  Prediction Length: {config.prediction_length}")
    print(f"  Input Channels: {config.num_input_channels}")
    print(f"  Prediction Channels: {len(config.prediction_channel_indices)}")
    print(f"  Categorical Vocabularies: {config.categorical_vocab_size_list}")
    
    return config

def load_pretrained_ttm_model(config):
    """
    Carrega modelo TTM pré-treinado da IBM Granite.
    
    Utiliza o modelo granite-timeseries-ttm-r2 como base e adapta
    para a configuração específica do dataset.
    
    Args:
        config (TinyTimeMixerConfig): Configuração do modelo
    
    Returns:
        TinyTimeMixerForPrediction: Modelo TTM carregado
    """
    print("Carregando modelo TTM pré-treinado...")
    
    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",    # Modelo base pré-treinado
        config=config,                              # Configuração personalizada
        device_map="auto",                          # Mapeamento automático de dispositivos
        ignore_mismatched_sizes=True,              # Ignorar incompatibilidades de tamanho
    )
    
    print(f"Modelo carregado: {model.__class__.__name__}")
    return model

def setup_selective_fine_tuning(model):
    """
    Configura fine-tuning seletivo congelando a maioria das camadas.
    
    Mantém apenas as camadas fully-connected (fc1/fc2) treináveis,
    reduzindo significativamente o número de parâmetros a treinar.
    
    Args:
        model: Modelo TTM carregado
    
    Returns:
        None (modifica modelo in-place)
    """
    print("Configurando fine-tuning seletivo...")
    
    # Congelar todas as camadas inicialmente
    for param in model.parameters():
        param.requires_grad = False
    
    # Descongelar apenas camadas fc1 e fc2 (fully-connected)
    trainable_layers = []
    for name, module in model.named_modules():
        if 'fc1' in name or 'fc2' in name:
            if isinstance(module, torch.nn.Linear):
                module.requires_grad_(True)
                trainable_layers.append(name)
    
    # Calcular estatísticas de parâmetros
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (trainable_params / total_params) * 100
    
    print(f"Fine-tuning seletivo configurado:")
    print(f"  Camadas treináveis: {trainable_layers}")
    print(f"  Parâmetros treináveis: {trainable_params:,} / {total_params:,} ({trainable_percentage:.1f}%)")
    print(f"  Redução de parâmetros: {100-trainable_percentage:.1f}%")

# Executar configuração do modelo
model_config = create_ttm_config(trained_tsp)
model = load_pretrained_ttm_model(model_config)
setup_selective_fine_tuning(model)

# =============================================================================
# SEÇÃO 5: CONFIGURAÇÃO E EXECUÇÃO DO TREINAMENTO
# =============================================================================

# Hiperparâmetros de treinamento
LEARNING_RATE = 5e-4                # Taxa de aprendizado otimizada para fine-tuning
NUM_TRAIN_EPOCHS = 100              # Número máximo de épocas (com early stopping)
PER_DEVICE_TRAIN_BATCH_SIZE = 2     # Batch size para treinamento (limitado por memória)
PER_DEVICE_EVAL_BATCH_SIZE = 4      # Batch size para validação (pode ser maior)

def create_training_arguments():
    """
    Cria argumentos de treinamento otimizados para o modelo TTM.
    
    Configura estratégias de salvamento, avaliação e logging
    para garantir o melhor modelo seja preservado.
    
    Returns:
        TrainingArguments: Configuração de treinamento
    """
    return TrainingArguments(
        output_dir="./results_ttm_model",              # Diretório de saída
        overwrite_output_dir=True,                     # Sobrescrever resultados anteriores
        learning_rate=LEARNING_RATE,                   # Taxa de aprendizado
        num_train_epochs=NUM_TRAIN_EPOCHS,             # Número de épocas
        do_eval=True,                                  # Executar avaliação
        eval_strategy="epoch",                         # Avaliar a cada época
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,  # Batch size treino
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,    # Batch size validação
        dataloader_num_workers=0,                      # Workers para carregamento (evita problemas)
        report_to="none",                              # Não reportar para ferramentas externas
        save_strategy="epoch",                         # Salvar modelo a cada época
        logging_strategy="epoch",                      # Log de métricas a cada época
        save_total_limit=2,                           # Manter apenas 2 checkpoints
        logging_dir="./results_ttm_model/logs",       # Diretório de logs
        load_best_model_at_end=True,                  # Carregar melhor modelo ao final
        metric_for_best_model="eval_loss",            # Métrica para seleção do melhor modelo
        greater_is_better=False,                      # Menor loss é melhor
        use_cpu=DEVICE == "cpu",                      # Forçar CPU se necessário
    )

def create_training_callbacks():
    """
    Cria callbacks para controle avançado do treinamento.
    
    Inclui early stopping para evitar overfitting e tracking
    personalizado de métricas durante o treinamento.
    
    Returns:
        list: Lista de callbacks configurados
    """
    # Early stopping: para quando não há melhoria por 15 épocas
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=15,     # Paciência: 15 épocas sem melhoria
        early_stopping_threshold=0.0,   # Threshold mínimo para considerar melhoria
    )
    
    # Callback para tracking personalizado de métricas
    tracking_callback = TrackingCallback()
    
    return [early_stopping_callback, tracking_callback]

def create_optimizer_and_scheduler(model, train_dataset_size):
    """
    Cria otimizador e scheduler de learning rate otimizados.
    
    Utiliza AdamW com OneCycleLR para convergência rápida e estável.
    
    Args:
        model: Modelo para otimização
        train_dataset_size (int): Tamanho do dataset de treino
    
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Otimizador AdamW (versão melhorada do Adam)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler OneCycleLR para variação cíclica do learning rate
    steps_per_epoch = math.ceil(train_dataset_size / PER_DEVICE_TRAIN_BATCH_SIZE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,                # Learning rate máximo
        epochs=NUM_TRAIN_EPOCHS,             # Total de épocas
        steps_per_epoch=steps_per_epoch,     # Steps por época
    )
    
    print(f"Otimização configurada:")
    print(f"  Otimizador: AdamW (lr={LEARNING_RATE})")
    print(f"  Scheduler: OneCycleLR")
    print(f"  Steps por época: {steps_per_epoch}")
    
    return optimizer, scheduler

class TTMTrainer(Trainer):
    """
    Trainer customizado para o modelo TinyTimeMixer.
    
    Sobrescreve o método compute_loss para filtrar adequadamente
    as entradas compatíveis com o modelo TTM.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Calcula a função de perda personalizada para TTM.
        
        Filtra apenas as chaves de entrada válidas para o modelo TTM,
        evitando erros de entrada incompatível.
        
        Args:
            model: Modelo TTM
            inputs (dict): Inputs do batch
            return_outputs (bool): Se deve retornar outputs além da loss
            num_items_in_batch: Número de itens no batch
        
        Returns:
            torch.Tensor ou tuple: Loss (e outputs se solicitado)
        """
        # Chaves válidas aceitas pelo modelo TTM
        valid_keys = [
            'past_values',              # Valores históricos
            'future_values',            # Valores futuros (targets)
            'past_observed_mask',       # Máscara de valores observados no passado
            'future_observed_mask',     # Máscara de valores observados no futuro
            'freq_token',               # Token de frequência temporal
            'static_categorical_values' # Valores categóricos estáticos
        ]
        
        # Filtrar apenas entradas válidas
        filtered_inputs = {k: v for k, v in inputs.items() if k in valid_keys}
        
        # Forward pass no modelo
        outputs = model(**filtered_inputs)
        
        # Extrair loss dos outputs
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return (loss, outputs) if return_outputs else loss

def train_ttm_model():
    """
    Executa o treinamento completo do modelo TTM.
    
    Configura todos os componentes necessários e executa o treinamento
    com monitoramento e salvamento automático do melhor modelo.
    
    Returns:
        Trainer: Trainer após conclusão do treinamento
    """
    print("\n" + "="*80)
    print("INICIANDO TREINAMENTO DO MODELO TTM")
    print("="*80)
    
    # Criar componentes de treinamento
    training_args = create_training_arguments()
    callbacks = create_training_callbacks()
    optimizer, scheduler = create_optimizer_and_scheduler(model, len(train_dataset))
    
    # Inicializar trainer customizado
    trainer = TTMTrainer(
        model=model,                        # Modelo TTM configurado
        args=training_args,                 # Argumentos de treinamento
        train_dataset=train_dataset,        # Dataset de treino
        eval_dataset=valid_dataset,         # Dataset de validação
        callbacks=callbacks,                # Callbacks (early stopping, tracking)
        optimizers=(optimizer, scheduler),  # Otimizador e scheduler
    )
    
    print(f"Trainer configurado:")
    print(f"  Epochs máximas: {NUM_TRAIN_EPOCHS}")
    print(f"  Batch size treino: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Batch size validação: {PER_DEVICE_EVAL_BATCH_SIZE}")
    print(f"  Early stopping: 15 épocas de paciência")
    print(f"  Dispositivo: {DEVICE}")
    
    print("\nIniciando treinamento...")
    
    # Executar treinamento
    trainer.train()
    
    print("\n" + "="*80)
    print("TREINAMENTO CONCLUÍDO")
    print("="*80)
    
    return trainer

def save_final_model(trainer, save_path="./final_ttm_model"):
    """
    Salva o modelo final treinado.
    
    Args:
        trainer: Trainer após treinamento
        save_path (str): Caminho para salvar o modelo
    """
    print(f"\nSalvando modelo final em: {save_path}")
    trainer.save_model(save_path)
    
    # Salvar também informações do preprocessador
    import pickle
    preprocessor_path = f"{save_path}/preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(trained_tsp, f)
    
    print(f"Modelo salvo com sucesso!")
    print(f"  Modelo: {save_path}")
    print(f"  Preprocessador: {preprocessor_path}")

# =============================================================================
# EXECUÇÃO DO TREINAMENTO
# =============================================================================

if __name__ == "__main__":
    # Executar treinamento
    trainer = train_ttm_model()
    
    # Salvar modelo final
    save_final_model(trainer)
    
    print("\n🎉 Pipeline de treinamento TTM finalizado com sucesso!")
