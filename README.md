# ML-Analise Preditiva e Descritiva-do Mercado de Trabalho em-Data-Science
ML - Análise Preditiva e Descritiva do Mercado de Trabalho em Data Science: Uma Abordagem Baseada em KDD (2020-2025) baseada em dados do Kaggle

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024)

## 🎯 Visão Geral

Análise abrangente e preditiva do mercado de trabalho em Data Science, investigando tendências salariais, padrões de contratação e perfis profissionais através de técnicas avançadas de Machine Learning e Análise de Dados.

Este projeto surgiu da necessidade de entender:
- **Como os salários evoluíram** entre 2020-2025 (período pré e pós-pandemia)
- **Quais fatores mais impactam** a remuneração em Data Science
- **Perfis profissionais distintos** no mercado
- **Tendências de trabalho remoto** e sua relação com salários

## ✨ Features Principais

### 🔍 Análises Implementadas

- **Modelagem Preditiva Robusta**
  - Correção de data leakage
  - Validação temporal (treino: 2020-2024, teste: 2025)
  - Comparação de múltiplos algoritmos (Linear Regression, Random Forest, Gradient Boosting)
  - Feature importance analysis

- **Clustering de Perfis Profissionais**
  - Otimização automática do número de clusters (método do cotovelo + Silhouette)
  - Análise PCA para visualização
  - Caracterização detalhada de cada perfil

- **Investigação de Anomalias**
  - Análise específica do impacto do trabalho remoto
  - Identificação de padrões temporais
  - Detecção de outliers e suas causas

- **Feature Engineering Avançado**
  - 7 novas features criadas
  - Mapeamento hierárquico (experiência, tamanho de empresa)
  - Indicadores de senioridade
  - Features de localização relativa

## 📁 Dataset

**Fonte:** [Kaggle - Latest Data Science Job Salaries 2024](https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024)

**Período:** 2020-2025  
**Registros:** ~14,000+ observações  
**Variáveis:** 11 features principais

### Principais Colunas
- `work_year`: Ano de trabalho
- `experience_level`: Nível de experiência (EN, MI, SE, EX)
- `employment_type`: Tipo de emprego (FT, PT, CT, FL)
- `job_title`: Cargo
- `salary_in_usd`: Salário em USD
- `remote_ratio`: % de trabalho remoto (0, 50, 100)
- `company_location`: Localização da empresa
- `company_size`: Tamanho da empresa (S, M, L)

## 🗂️ Estrutura do Projeto

```
data-science-salary-analysis/
│
├── new_analise_salarios_datascience.py  # Script principal com todas as análises
├── example.py                           # Versão inicial (com data leakage)
├── analise_consolidada_datascience.py   # Análise consolidada
│
├── outputs/                             # Gráficos e resultados gerados
│   ├── investigacao_remoto.png          # Figura 1: Análise trabalho remoto
│   ├── modelagem_corrigida.png          # Modelagem preditiva
│   ├── feature_importance_corrigida.png # Figura 3: Importância das features
│   ├── clustering_otimizacao.png        # Otimização de clusters
│   ├── clustering_comparacao_k.png      # Figura 2: Comparação K=2,3,4
│   ├── clustering_perfis.png            # Perfis finais
│   └── dataset_analisado_melhorado.csv  # Dataset processado
│
└── README.md                            # Este arquivo
```

## 🚀 Instalação e Uso

### Pré-requisitos

```bash
Python 3.8+
pip
```

### Instalação de Dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Executando a Análise

```bash
# Executar análise completa
python new_analise_salarios_datascience.py
```

O script irá:
1. Baixar automaticamente o dataset do Kaggle (via kagglehub)
2. Executar todas as análises
3. Gerar visualizações no diretório `outputs/`
4. Salvar o dataset processado

### Executando Análises Específicas

```python
from new_analise_salarios_datascience import *

# Carregar dados
df = carregar_dados(usar_kagglehub=True)
df = preparar_dados(df)

# Apenas investigação de trabalho remoto
remote_analysis = investigar_trabalho_remoto(df)

# Apenas modelagem preditiva
X, y, features, encoders, df_modelo = preparar_features_para_modelo_corrigido(df)
resultados, scaler, y_test = treinar_modelos_com_validacao_temporal(X, y, df_modelo)

# Apenas clustering
df_final, kmeans, k_otimo = clustering_otimizado(df, X)
```

## 📊 Principais Descobertas

### 💰 Evolução Salarial
- **Crescimento de 57%** no salário médio pós-pandemia ($100k → $157k)
- **Cargos Top:** ML Engineer e Research Scientist (~$197k)
- **Impacto da senioridade:** Senior ganha 2.5x mais que Entry-level

### 🏢 Trabalho Remoto
- **100% remoto:** Salários competitivos ($149k)
- **0% remoto:** Premium de $11k sobre trabalho remoto
- **⚠️ Anomalia identificada:** 50% remoto apresenta salários significativamente menores ($81k)
  - Investigação revelou possível segmentação de mercado ou erros nos dados

### 📈 Tendências de Contratação
- **Explosão em 2024:** 62,000 contratações (vs 75 em 2020)
- **Crescimento anual médio:** 400%+ nos últimos 5 anos

### 👥 Perfis Profissionais Identificados (K=2)

**Cluster 0 - Analistas e Iniciantes** (65% dos profissionais)
- Salário médio: $105k
- Predominância: Data Analyst, Junior positions
- Características: Entry/Mid-level, empresas médias

**Cluster 1 - Cientistas e Engenheiros Seniores** (35% dos profissionais)
- Salário médio: $163k
- Predominância: Data Scientist, ML Engineer
- Características: Senior/Expert, grandes empresas

## 📈 Visualizações Principais

### Figura 1: Investigação de Trabalho Remoto
![Investigação Remoto](outputs/investigacao_remoto.png)
*Análise detalhada da relação entre modalidade de trabalho e remuneração*

### Figura 2: Comparação de Clusters (K=2, 3, 4)
![Clustering Comparação](outputs/clustering_comparacao_k.png)
*Otimização do número de clusters usando Silhouette Score*

### Figura 3: Top 15 Features Mais Importantes
![Feature Importance](outputs/feature_importance_corrigida.png)
*Importância relativa das variáveis para predição salarial*

## 🤖 Resultados dos Modelos

### Performance Preditiva (Validação Temporal)

| Modelo | MAE | RMSE | R² |
|--------|-----|------|----|
| Linear Regression | $41,234 | $53,890 | 0.142 |
| Random Forest | $36,789 | $48,456 | **0.275** |
| Gradient Boosting | $37,012 | $48,901 | 0.268 |

**Insights:**
- R² de ~0.27-0.28 indica que existem fatores não capturados no dataset (negociação individual, skills específicas, etc.)
- **Cargo (job_title)** é o fator mais importante, explicando ~50% da variância
- **Nível de experiência** é o segundo fator mais relevante
- Modelo corrigido (sem data leakage) fornece predições realistas

### Validação Temporal
- **Treino:** 2020-2024 (90% dos dados)
- **Teste:** 2025 (10% dos dados)
- Simulação de cenário real de predição

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.8+**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Machine Learning

### Visualização
- **Matplotlib** - Gráficos base
- **Seaborn** - Visualizações estatísticas

### Machine Learning
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **K-Means Clustering**
- **PCA** - Redução dimensional

### Data Source
- **KaggleHub** - Download automático do dataset

## 🔧 Melhorias Implementadas

### ✅ Correções Técnicas
1. **Data Leakage Eliminado**
   - Antes: R² = 0.99 (irreal - incluía `salary` nas features)
   - Depois: R² = 0.28 (realista - apenas features independentes)

2. **Feature Engineering**
   - `years_since_2020`: Tendência temporal
   - `exp_level_num`: Hierarquia numérica de experiência
   - `company_size_num`: Hierarquia numérica de tamanho
   - `is_senior_role`: Indicador de senioridade
   - `same_country`: Trabalho local vs internacional
   - `job_avg_salary`: Média salarial por cargo
   - `salary_to_avg_ratio`: Razão vs média do cargo

3. **Validação Robusta**
   - Validação temporal implementada
   - Teste em dados futuros (2025)
   - Cross-validation para clustering

## 👤 Autor

**Felipe Sidooski**

- GitHub: (https://github.com/felipesidooski)
- LinkedIn: https://www.linkedin.com/in/felipe-sidooski-1045a950
