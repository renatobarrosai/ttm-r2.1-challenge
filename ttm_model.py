#- pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.22"
#

import pandas as pd
import torch
import numpy as np
import math # Para math.ceil no scheduler
import os # Para os.cpu_count

# Importações específicas da biblioteca Hugging Face e TSFM
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from peft import LoraConfig, get_peft_model # Importante para aplicar LoRA ao modelo

# Componentes da biblioteca IBM Granite TSFM
from tsfm_public import (
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    ForecastDFDataset,
    TrackingCallback,
)
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.models.tinytimemixer.configuration_tinytimemixer import TinyTimeMixerConfig

# --- 1. Dados e Configurações Iniciais ---
# Carregamento do seu arquivo de dados. O dtype da coluna 'date' já é datetime64[ns], o que é ideal.
try:
    df = pd.read_csv("./dados/db_tratado-w.csv", parse_dates=["date"])
except FileNotFoundError:
    print("Erro: Arquivo db_tratado-w.csv não encontrado. Por favor, verifique o caminho.")
    # Exemplo de DataFrame para demonstração se o arquivo não estiver presente:
    print("Criando um DataFrame de exemplo para prosseguir com a demonstração...")
    data_example = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29', '2023-02-05',
                                '2023-02-12', '2023-02-19', '2023-02-26', '2023-03-05', '2023-03-12', '2023-03-19']),
        'produto_cat': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'uf_cat': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'vendas': [3-14],
        'faturamento': [1000.0, 1100.0, 1050.0, 1200.0, 1150.0, 1300.0, 1250.0, 1400.0, 1350.0, 1500.0, 1450.0, 1600.0]
    }
    df_single_product = pd.DataFrame(data_example)
    # Simular múltiplos produtos e UFs para atingir ~6440 entradas
    df = pd.concat([df_single_product.copy().assign(produto_cat=i, uf_cat=j)
                    for i in range(1, 20) for j in range(1, 5)], ignore_index=True)
    df['date'] = pd.to_datetime(df['date']) # Ensure datetime type after concat
    print("DataFrame de exemplo criado.")


print("\nDataFrame carregado/criado com sucesso:")
print(df.info())
print(df.head())

# Definindo as configurações do seu dataset para o pré-processador
TIMESTAMP_COLUMN = 'date'
ID_COLUMNS = ['produto_cat'] # Coluna que identifica unicamente cada série temporal de produto
TARGET_COLUMNS = ['vendas', 'faturamento'] # Colunas que serão previstas
STATIC_CATEGORICAL_COLUMNS = ['uf_cat'] # Coluna categórica que não varia no tempo para uma série
CONTROL_COLUMNS = [] # Não há colunas exógenas 'controláveis' ou 'observáveis' além das estáticas

# Granularidade temporal semanal.
# Vamos definir o context_length para 2 anos de dados semanais (aproximadamente 104 semanas)
# e prediction_length para 12 semanas (aproximadamente 3 meses).
CONTEXT_LENGTH = 104 # 2 anos de histórico semanal
PREDICTION_LENGTH = 12 # 3 meses de previsão semanal

# Configuração do dispositivo de hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# --- 2. Pré-processamento e Divisão dos Dados com TimeSeriesPreprocessor ---

# Instanciando o TimeSeriesPreprocessor
# Conforme a Seção 2.3 da refatoração, o escalonamento deve ser independente por canal.
# 'scaling_id_columns=ID_COLUMNS + STATIC_CATEGORICAL_COLUMNS' garante que o escalonamento
# (normalização) seja feito para cada combinação única de produto_cat e uf_cat, que define
# o que entendemos por "canal" para previsão.
tsp = TimeSeriesPreprocessor(
    id_columns=ID_COLUMNS,
    timestamp_column=TIMESTAMP_COLUMN,
    target_columns=TARGET_COLUMNS,
    static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,
    control_columns=CONTROL_COLUMNS, # Lista vazia, conforme definido acima
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    scaling=True, # Habilita a normalização dos dados
    scaler_type="standard", # Usa StandardScaler (média 0, desvio padrão 1)
    scaling_id_columns=ID_COLUMNS + STATIC_CATEGORICAL_COLUMNS, # Escalonamento por canal produto-UF [15]
    encode_categorical=True, # Codifica 'uf_cat' para que o modelo possa processá-la [16, 17]
)

# Divisão estratégica dos dados (70% treino, 15% validação, 15% teste) em ordem cronológica
# A função 'prepare_data_splits' da TSFM facilita essa divisão.
train_data_raw, valid_data_raw, test_data_raw = prepare_data_splits(
    df,
    id_columns=ID_COLUMNS + STATIC_CATEGORICAL_COLUMNS, # IDs para garantir a divisão por canal
    context_length=CONTEXT_LENGTH, # Necessário para que a divisão considere janelas de contexto
    split_config={"train": 0.7, "test": 0.15} # 'valid' = 0.15 é calculado implicitamente
)

# Treinar o preprocessor APENAS com os dados de treino para evitar data leakage (vazamento de dados) [10]
trained_tsp = tsp.train(train_data_raw)

# Preparar os datasets para o PyTorch/Hugging Face Trainer
# 'ForecastDFDataset' é um wrapper que formata os dados pré-processados para o modelo.
# O 'frequency_token' é adicionado para informar o modelo sobre a granularidade dos dados.
train_dataset = ForecastDFDataset(trained_tsp.preprocess(train_data_raw),
                                  id_columns=ID_COLUMNS,
                                  timestamp_column=TIMESTAMP_COLUMN,
                                  target_columns=TARGET_COLUMNS,
                                  control_columns=CONTROL_COLUMNS,
                                  static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,
                                  context_length=CONTEXT_LENGTH,
                                  prediction_length=PREDICTION_LENGTH,
                                  frequency_token=trained_tsp.get_frequency_token(trained_tsp.freq)
                                  )

valid_dataset = ForecastDFDataset(trained_tsp.preprocess(valid_data_raw),
                                  id_columns=ID_COLUMNS,
                                  timestamp_column=TIMESTAMP_COLUMN,
                                  target_columns=TARGET_COLUMNS,
                                  control_columns=CONTROL_COLUMNS,
                                  static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,
                                  context_length=CONTEXT_LENGTH,
                                  prediction_length=PREDICTION_LENGTH,
                                  frequency_token=trained_tsp.get_frequency_token(trained_tsp.freq)
                                  )

test_dataset = ForecastDFDataset(trained_tsp.preprocess(test_data_raw),
                                 id_columns=ID_COLUMNS,
                                 timestamp_column=TIMESTAMP_COLUMN,
                                 target_columns=TARGET_COLUMNS,
                                 control_columns=CONTROL_COLUMNS,
                                 static_categorical_columns=STATIC_CATEGORICAL_COLUMNS,
                                 context_length=CONTEXT_LENGTH,
                                 prediction_length=PREDICTION_LENGTH,
                                 frequency_token=trained_tsp.get_frequency_token(trained_tsp.freq)
                                 )

print(f"\nTamanho do dataset de treino: {len(train_dataset)}")
print(f"Tamanho do dataset de validação: {len(valid_dataset)}")
print(f"Tamanho do dataset de teste: {len(test_dataset)}")

# --- 3. Instanciação e Customização da Arquitetura TTM com QLoRA ---

# Definir a configuração customizada do modelo TinyTimeMixer
# Embora a Seção 1.3/3.1 do plano estratégico mencione 'from_config' para iniciar do zero [9, 18],
# a aplicação de QLoRA (Seção 3.2) é tipicamente feita via 'from_pretrained' que carrega a arquitetura
# e aplica a quantização aos pesos (mesmo que sejam aleatórios ou descartados para fine-tuning completo).
# A forma mais prática de usar QLoRA com a arquitetura Granite TTM é via 'from_pretrained'
# com uma 'config' customizada.

model_config = TinyTimeMixerConfig(
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    num_input_channels=trained_tsp.num_input_channels, # Total de canais (vendas, faturamento, uf_cat codificada)
    prediction_channel_indices=trained_tsp.prediction_channel_indices, # Índices dos alvos ('vendas', 'faturamento')
    exogenous_channel_indices=trained_tsp.exogenous_channel_indices, # Índices dos canais exógenos (se houver, vazio aqui)
    decoder_mode="mix_channel", # Habilita o modelo a capturar correlações entre diferentes canais (produtos) [19, 20]
    categorical_vocab_size_list=trained_tsp.categorical_vocab_size_list, # Tamanho do vocabulário para 'uf_cat' [19, 21]
    # Outros parâmetros da arquitetura TTM podem ser ajustados aqui se necessário
)

# Configuração da quantização (QLoRA) para otimizar o uso de memória [13]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Carrega o modelo com seus pesos quantizados em 4 bits [22, 23]
    bnb_4bit_quant_type="nf4", # Tipo de quantização "NormalFloat 4" [22]
    bnb_4bit_compute_dtype=torch.bfloat16, # Tipo de dado para cálculos internos, para manter estabilidade [22]
)

# Carregamento da arquitetura do modelo com a configuração customizada e quantização
# 'ignore_mismatched_sizes=True' é útil se a configuração customizada (e.g., num_input_channels, prediction_length)
# difere da pre-treinada, forçando a re-inicialização das camadas relevantes.
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2", # Caminho da arquitetura base TTM-R2
    config=model_config, # Passa a configuração customizada
    quantization_config=bnb_config, # Aplica a configuração de quantização
    device_map="auto", # Mapeia automaticamente para GPU ou CPU
    ignore_mismatched_sizes=True, # Permite carregar a arquitetura mesmo com diferenças de tamanho de camada
)

# Configuração do LoRA para PEFT (Parameter-Efficient Fine-Tuning)
# O TTM é baseado em MLP-Mixer, que não usa "q_proj", "v_proj" como Transformers tradicionais.
# É crucial identificar as camadas lineares corretas dentro da arquitetura TTM para aplicar o LoRA.
# Nomes comuns para camadas lineares em MLPs podem ser 'mlp.fc1', 'mlp.fc2', 'linear', 'c_fc1', etc.
# Para este exemplo, usaremos 'c_fc1' e 'c_fc2' como um palpite educado de nomes de camadas lineares internas.
# **Importante**: Estes 'target_modules' podem precisar ser verificados na implementação exata do
# 'TinyTimeMixerForPrediction' para garantir que o LoRA seja aplicado às camadas corretas.
lora_config = LoraConfig(
    r=16, # O "rank" da decomposição, um valor baixo para eficiência [24]
    lora_alpha=32, # Fator de escala para os pesos adaptados [24]
    target_modules=["c_fc1", "c_fc2"], # **Pode precisar de ajuste**: nomes de camadas lineares no TTM [12]
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION", # Tipo de tarefa para PEFT. 'FEATURE_EXTRACTION' é um palpite genérico para TS.
)

# Aplica a configuração LoRA ao modelo
model = get_peft_model(model, lora_config)
print("\nModelo configurado para PEFT (LoRA):")
model.print_trainable_parameters() # Mostra quantos parâmetros serão treinados (apenas os do LoRA)

# --- 4. Configuração e Execução do Treinamento ---

# Configuração dos argumentos de treinamento usando 'TrainingArguments'
# Ajustar 'per_device_train_batch_size' e 'learning_rate' é crucial para o desempenho e para evitar OOM.
LEARNING_RATE = 5e-4 # Taxa de aprendizado inicial, um bom ponto de partida [25]
NUM_TRAIN_EPOCHS = 100 # Número máximo de épocas. Early stopping pode parar antes [26]
PER_DEVICE_TRAIN_BATCH_SIZE = 8 # Ajustar de acordo com a RAM/VRAM disponível. Comece pequeno e aumente.
PER_DEVICE_EVAL_BATCH_SIZE = PER_DEVICE_TRAIN_BATCH_SIZE * 2 # Lotes de avaliação podem ser maiores

training_args = TrainingArguments(
    output_dir="./results_ttm_finetuned", # Diretório para salvar checkpoints e logs
    overwrite_output_dir=True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    do_eval=True, # Realizar avaliação durante o treinamento
    evaluation_strategy="epoch", # Avaliar no dataset de validação a cada época [25]
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 0, # Otimiza o carregamento de dados
    report_to="none", # Desabilita o reporte para plataformas como Weights & Biases
    save_strategy="epoch", # Salva um checkpoint do modelo a cada época [25]
    logging_strategy="epoch", # Registra logs a cada época
    save_total_limit=1, # Mantém apenas o melhor checkpoint (com menor 'eval_loss')
    logging_dir="./results_ttm_finetuned/logs",
    load_best_model_at_end=True, # Recarrega o melhor modelo (baseado em eval_loss) ao final [27]
    metric_for_best_model="eval_loss", # Métrica para determinar o "melhor" modelo [27]
    greater_is_better=False, # Para 'loss', um valor menor é melhor
    use_cpu=DEVICE == "cpu", # Força o uso da CPU se não houver GPU disponível
)

# Callbacks para o treinamento eficiente
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10, # Parar o treinamento se 'eval_loss' não melhorar por 10 épocas [28, 29]
    early_stopping_threshold=0.0, # Um limiar mínimo para considerar uma melhoria [28, 29]
)
tracking_callback = TrackingCallback() # Um callback da TSFM para monitoramento de tempo/estatísticas [30, 31]

# Otimizador e scheduler de learning rate
# AdamW é um otimizador popular para deep learning. OneCycleLR ajusta a taxa de aprendizado dinamicamente.
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = OneCycleLR(
    optimizer,
    LEARNING_RATE,
    epochs=NUM_TRAIN_EPOCHS,
    steps_per_epoch=math.ceil(len(train_dataset) / PER_DEVICE_TRAIN_BATCH_SIZE),
)

# Instanciando o Trainer da Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback, tracking_callback],
    optimizers=(optimizer, scheduler), # Passa o otimizador e o scheduler customizados
)

print("\nIniciando o treinamento do modelo...")
trainer.train()
print("Treinamento concluído.")

# Salvar o modelo fine-tuned
# Com PEFT, o 'trainer.save_model' salva os adaptadores LoRA. Para inferência, você pode carregar
# o modelo base e então os adaptadores, ou mesclá-los em um único modelo (se a biblioteca PEFT suportar para TTM).
trainer.save_model("./final_ttm_model_with_lora")

# Exemplo de como mesclar os adaptadores com o modelo base e salvar (se aplicável e suportado para TTM)
# Note: Esta parte pode variar e exigir testes ou consulta à documentação específica da PEFT/TSFM
# para TinyTimeMixer, pois 'merge_and_unload' é mais comum em modelos de linguagem.
# try:
#     # Carrega o modelo base original (não quantizado ou LoRA)
#     base_model = TinyTimeMixerForPrediction.from_pretrained(
#         "ibm-granite/granite-timeseries-ttm-r2",
#         config=model_config,
#         device_map="auto"
#     )
#     # Carrega os adaptadores LoRA
#     model_with_lora = PeftModel.from_pretrained(base_model, "./final_ttm_model_with_lora")
#     # Mescla os adaptadores
#     merged_model = model_with_lora.merge_and_unload()
#     merged_model.save_pretrained("./final_ttm_model_merged")
#     print(f"Modelo mesclado salvo em: ./final_ttm_model_merged")
# except Exception as e:
#     print(f"Não foi possível mesclar o modelo LoRA: {e}. O modelo LoRA foi salvo separadamente.")

print(f"Modelo (com adaptadores LoRA) salvo em: {training_args.output_dir}")
