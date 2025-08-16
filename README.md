# 🚀 Time Series Forecasting com TinyTimeMixer (TTM) - IBM Granite

## 📋 Visão Geral

Este projeto implementa um modelo de previsão de séries temporais utilizando o **TinyTimeMixer (TTM)** da IBM Granite para prever vendas e faturamento por produto e UF. O modelo utiliza técnicas avançadas de deep learning para séries temporais, incluindo fine-tuning seletivo e preprocessamento automatizado.

### 🎯 Objetivo

Desenvolver um sistema de previsão capaz de:
- Prever vendas e faturamento para diferentes produtos e UFs
- Utilizar janelas de contexto de 2 anos (104 semanas) para previsões de 6 meses (26 semanas)
- Aplicar fine-tuning eficiente em modelo pré-treinado
- Garantir reprodutibilidade e robustez nas previsões

## 🏗️ Arquitetura do Projeto

### Componentes Principais

1. **Carregamento e Preparação dos Dados**
   - Carregamento do dataset de vendas/faturamento
   - Geração de dados sintéticos para demonstração
   - Resampling para frequência semanal consistente

2. **Pré-processamento**
   - Normalização/escalonamento por grupo produto-UF
   - Codificação de variáveis categóricas
   - Divisão estratificada preservando combinações

3. **Configuração do Modelo TTM**
   - Carregamento do modelo pré-treinado IBM Granite
   - Configuração personalizada para o dataset
   - Fine-tuning seletivo (apenas camadas fc1/fc2)

4. **Treinamento**
   - Otimização AdamW com OneCycleLR scheduler
   - Early stopping com paciência de 15 épocas
   - Monitoramento automático de métricas

5. **Salvamento**
   - Modelo final otimizado
   - Preprocessador para inferência
   - Logs de treinamento e métricas

## 🛠️ Configuração do Ambiente

### Pré-requisitos

- Python 3.8+
- CUDA 11.8+ (opcional, para GPU)
- 16GB+ RAM recomendado
- 10GB+ espaço em disco

### Instalação

1. **Clone o repositório:**
```bash
git clone url-do-repositorio>](https://github.com/renatobarrosai/ttm-r2.1-challenge.git
cd treinamento_semana
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate  # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Instale o IBM Granite TSFM:**
```bash
pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.22"
```

## 📊 Estrutura dos Dados

### Formato Esperado

O arquivo `./dados/db_tratado-w.csv` deve conter:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `date` | datetime | Data no formato YYYY-MM-DD |
| `produto_cat` | int | Categoria do produto (codificada) |
| `uf_cat` | int | UF categorizada (codificada) |
| `vendas` | float | Volume de vendas |
| `faturamento` | float | Valor do faturamento |

### Exemplo de Dados

```csv
date,produto_cat,uf_cat,vendas,faturamento
2023-01-01,1,2,300,1000.0
2023-01-08,1,2,310,1100.0
2023-01-15,1,2,305,1050.0
```

## 🚀 Execução

### Treinamento do Modelo

```bash
python ttm_model.py
```

### Principais Parâmetros Configuráveis

```python
# Configurações temporais
CONTEXT_LENGTH = 104        # 2 anos de histórico semanal
PREDICTION_LENGTH = 26      # 6 meses de previsão

# Hiperparâmetros de treinamento
LEARNING_RATE = 5e-4        # Taxa de aprendizado
NUM_TRAIN_EPOCHS = 100      # Máximo de épocas
BATCH_SIZE = 2              # Batch size para treinamento
```

## 📈 Monitoramento

### Métricas Acompanhadas

- **Loss de Treinamento**: Função de perda durante treinamento
- **Loss de Validação**: Função de perda no conjunto de validação
- **Early Stopping**: Controle automático de overfitting
- **Parâmetros Treináveis**: Monitoramento de eficiência

### Logs e Outputs

```
./results_ttm_model/
├── logs/                   # TensorBoard logs
├── checkpoint-*/           # Checkpoints intermediários
└── pytorch_model.bin       # Modelo final

./final_ttm_model/
├── config.json            # Configuração do modelo
├── pytorch_model.bin      # Pesos do modelo
└── preprocessor.pkl       # Preprocessador serializado
```

## 🔧 Funcionalidades Principais

### 1. Carregamento Inteligente de Dados
- Detecção automática de arquivo de dados
- Geração de dados sintéticos para demonstração
- Validação de formato e consistência

### 2. Preprocessamento Robusto
- Resampling automático para frequência semanal
- Normalização por grupo produto-UF
- Divisão estratificada sem vazamento de dados

### 3. Fine-tuning Eficiente
- Congelamento seletivo de camadas
- Redução de ~95% dos parâmetros treináveis
- Otimização focada em camadas específicas

### 4. Treinamento Otimizado
- Early stopping inteligente
- Scheduler de learning rate adaptativo
- Salvamento automático do melhor modelo

## 📋 Estrutura do Código

```
ttm_model.py
├── SEÇÃO 1: Carregamento e Configuração Inicial
├── SEÇÃO 2: Pré-processamento e Preparação
├── SEÇÃO 3: Divisão dos Dados e Criação dos Datasets
├── SEÇÃO 4: Configuração e Inicialização do Modelo
└── SEÇÃO 5: Configuração e Execução do Treinamento
```

### Principais Funções

| Função | Descrição |
|--------|-----------|
| `load_data()` | Carrega dados ou gera exemplos sintéticos |
| `resample_data_to_weekly()` | Aplica resampling semanal consistente |
| `split_data_by_combinations()` | Divide dados preservando combinações |
| `create_ttm_config()` | Configura modelo TTM personalizado |
| `setup_selective_fine_tuning()` | Configura fine-tuning seletivo |
| `train_ttm_model()` | Executa treinamento completo |

## 🎯 Resultados Esperados

### Performance
- **Redução de Parâmetros**: ~95% menos parâmetros treináveis
- **Tempo de Treinamento**: Significativamente reduzido vs. full fine-tuning
- **Generalização**: Melhor performance em dados não vistos

### Outputs do Modelo
- Previsões de vendas para próximas 26 semanas
- Previsões de faturamento para próximas 26 semanas
- Intervalos de confiança (quando disponíveis)
- Métricas de performance por produto-UF

## 🔍 Troubleshooting

### Problemas Comuns

1. **Erro de Memória GPU**
   ```python
   # Reduzir batch size
   PER_DEVICE_TRAIN_BATCH_SIZE = 1
   ```

2. **Arquivo de Dados Não Encontrado**
   - O código automaticamente gera dados sintéticos
   - Verifique o caminho: `./dados/db_tratado-w.csv`

3. **Dependências Não Instaladas**
   ```bash
   pip install -r requirements.txt
   pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.22"
   ```

## 📚 Referências

- [IBM Granite Time Series Foundation Models](https://github.com/ibm-granite/granite-tsfm)
- [TinyTimeMixer Paper](https://arxiv.org/abs/2401.03955)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Autores

- **Renato Barros** - *Desenvolvimento inicial* - [renatobarrosai](https://github.com/seuusername](https://github.com/renatobarrosai/)

## 🙏 Agradecimentos

- IBM Research pela disponibilização dos modelos Granite
- Hugging Face pela infraestrutura de modelos
- Comunidade open-source de Time Series Forecasting

---

**⚡ Para mais informações ou suporte, abra uma issue no repositório!**
