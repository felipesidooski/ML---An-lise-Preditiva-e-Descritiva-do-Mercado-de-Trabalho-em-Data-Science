"""
Análise Melhorada de Salários em Data Science (2020-2025)
==========================================================
Correções implementadas:
- Remoção de data leakage (salary)
- Feature engineering avançado
- Validação temporal
- Investigação de anomalias
- Teste de múltiplos clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import kagglehub
import os
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Diretório de outputs
OUTPUT_DIR = '/MachineLearn/ml/outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Diretório '{OUTPUT_DIR}' criado")

# ============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================================

def carregar_dados(caminho_arquivo=None, usar_kagglehub=True):
    """Carrega o dataset de salários"""
    print("Carregando dados...")
    
    if caminho_arquivo is not None:
        # Usar arquivo fornecido
        df = pd.read_csv(caminho_arquivo)
        print(f"Arquivo local carregado: {caminho_arquivo}")
    elif usar_kagglehub:
        try:
            print("Baixando dataset do Kaggle via kagglehub...")
            from kagglehub import KaggleDatasetAdapter
            
            # Carregar diretamente como DataFrame pandas
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "saurabhbadole/latest-data-science-job-salaries-2024",
                "DataScience_salaries_2025.csv"
            )
            print(f"✓ Dataset carregado do Kaggle")
            
        except Exception as e:
            print(f" Erro ao baixar do Kaggle: {e}")
            raise FileNotFoundError("Dataset não encontrado. Forneça o caminho do arquivo CSV.")
    else:
        raise ValueError("Forneça um caminho de arquivo ou use usar_kagglehub=True")
    
    print(f"✓ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df

def preparar_dados(df):
    """Prepara e limpa os dados, adicionando feature engineering"""
    print("\n Preparando dados com feature engineering...")
    
    df = df.copy()
    
    # Converter ano
    if 'work_year' in df.columns:
        df['work_year'] = pd.to_numeric(df['work_year'], errors='coerce')
    
    # Período pandemia
    if 'work_year' in df.columns:
        df['periodo_pandemia'] = df['work_year'].apply(
            lambda x: 'Pré-pandemia' if x < 2020 else 
                     ('Pandemia' if x <= 2021 else 'Pós-pandemia')
        )
    
    # Remover valores nulos críticos
    col_salario = 'salary_in_usd' if 'salary_in_usd' in df.columns else 'salary'
    df = df.dropna(subset=[col_salario])
    
    # ===== FEATURE ENGINEERING =====
    print("\n Criando features engineered...")
    
    # 1. Mapear nível de experiência para ordem numérica
    exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
    if 'experience_level' in df.columns:
        df['exp_level_num'] = df['experience_level'].map(exp_map)
    
    # 2. Mapear tamanho da empresa
    size_map = {'S': 1, 'M': 2, 'L': 3}
    if 'company_size' in df.columns:
        df['company_size_num'] = df['company_size'].map(size_map)
    
    # 3. Calcular salário médio por país (sem leakage)
    if 'company_location' in df.columns:
        location_avg = df.groupby('company_location')[col_salario].mean()
        df['location_avg_salary'] = df['company_location'].map(location_avg)
    
    # 4. Calcular salário médio por cargo (sem leakage)
    if 'job_title' in df.columns:
        job_avg = df.groupby('job_title')[col_salario].mean()
        df['job_avg_salary'] = df['job_title'].map(job_avg)
    
    # 5. Criar indicador de cargo sênior
    if 'job_title' in df.columns:
        df['is_senior_role'] = df['job_title'].str.lower().str.contains('senior|lead|principal|head|director|manager').astype(int)
    
    # 6. Indicador se funcionário e empresa estão no mesmo país
    if 'employee_residence' in df.columns and 'company_location' in df.columns:
        df['same_country'] = (df['employee_residence'] == df['company_location']).astype(int)
    
    # 7. Anos desde o início do período (tendência temporal)
    if 'work_year' in df.columns:
        df['years_since_2020'] = df['work_year'] - 2020
    
    print(f"✓ Dados preparados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"✓ Features engineered criadas: {df.shape[1] - len(df.select_dtypes(include=['object']).columns)} numéricas")
    
    return df

# ============================================================================
# 2. INVESTIGAÇÃO DE ANOMALIAS - TRABALHO REMOTO
# ============================================================================

def investigar_trabalho_remoto(df):
    """Investiga a anomalia no trabalho remoto"""
    print("\n" + "="*60)
    print("INVESTIGANDO: ANOMALIA TRABALHO REMOTO")
    print("="*60)
    
    if 'remote_ratio' not in df.columns:
        print("Coluna remote_ratio não encontrada")
        return None
    
    col_salario = 'salary_in_usd' if 'salary_in_usd' in df.columns else 'salary'
    
    # Análise detalhada por remote_ratio
    print("\nAnálise detalhada por remote_ratio:")
    remote_analysis = df.groupby('remote_ratio').agg({
        col_salario: ['mean', 'median', 'std', 'count'],
        'work_year': 'mean',
        'experience_level': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
        'company_size': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
    })
    print(remote_analysis)
    
    # Verificar distribuição por ano
    print("\nDistribuição de remote_ratio por ano:")
    remote_year = pd.crosstab(df['work_year'], df['remote_ratio'], normalize='index') * 100
    print(remote_year)
    
    # Visualização
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Boxplot de salários por remote_ratio
    df.boxplot(column=col_salario, by='remote_ratio', ax=axes[0, 0])
    axes[0, 0].set_title('Distribuição Salarial por Remote Ratio', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Remote Ratio (%)')
    axes[0, 0].set_ylabel('Salário (USD)')
    
    # 2. Salário médio por remote_ratio e experiência
    if 'experience_level' in df.columns:
        pivot_data = df.pivot_table(
            values=col_salario, 
            index='remote_ratio', 
            columns='experience_level', 
            aggfunc='mean'
        )
        pivot_data.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
        axes[0, 1].set_title('Salário por Remote Ratio e Experiência', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Remote Ratio (%)')
        axes[0, 1].set_ylabel('Salário Médio (USD)')
        axes[0, 1].legend(title='Experiência')
        axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Evolução temporal do remote_ratio
    if 'work_year' in df.columns:
        remote_evolution = df.groupby(['work_year', 'remote_ratio']).size().unstack(fill_value=0)
        remote_evolution.plot(kind='bar', stacked=True, ax=axes[1, 0], alpha=0.8)
        axes[1, 0].set_title('Evolução de Contratações por Remote Ratio', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Ano')
        axes[1, 0].set_ylabel('Número de Contratações')
        axes[1, 0].legend(title='Remote %')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Distribuição de contagem
    remote_counts = df['remote_ratio'].value_counts().sort_index()
    axes[1, 1].bar(remote_counts.index, remote_counts.values, alpha=0.7, color='coral')
    axes[1, 1].set_title('Distribuição de Registros por Remote Ratio', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Remote Ratio (%)')
    axes[1, 1].set_ylabel('Número de Registros')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'investigacao_remoto.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: {output_path}")
    
    # Análise de outliers em remote_ratio=50
    print("\n🔍 Análise específica do grupo remote_ratio=50:")
    remote_50 = df[df['remote_ratio'] == 50]
    if len(remote_50) > 0:
        print(f"Tamanho do grupo: {len(remote_50)}")
        print(f"Período predominante: {remote_50['work_year'].mode()[0] if 'work_year' in remote_50.columns else 'N/A'}")
        print(f"Cargo predominante: {remote_50['job_title'].mode()[0] if 'job_title' in remote_50.columns else 'N/A'}")
        print(f"Localização predominante: {remote_50['company_location'].mode()[0] if 'company_location' in remote_50.columns else 'N/A'}")
    
    return remote_analysis

# ============================================================================
# 3. MODELAGEM PREDITIVA CORRIGIDA (SEM DATA LEAKAGE)
# ============================================================================

def preparar_features_para_modelo_corrigido(df):
    """Prepara features SEM data leakage"""
    print("\n" + "="*60)
    print("PREPARAÇÃO PARA MODELAGEM (CORRIGIDA)")
    print("="*60)
    
    df_modelo = df.copy()
    col_salario = 'salary_in_usd' if 'salary_in_usd' in df.columns else 'salary'
    
    # Selecionar apenas features que não causam leakage
    features_numericas_seguras = [
        'work_year', 'remote_ratio', 'exp_level_num', 'company_size_num',
        'is_senior_role', 'same_country', 'years_since_2020'
    ]
    
    # Filtrar apenas as que existem
    features_numericas_seguras = [f for f in features_numericas_seguras if f in df_modelo.columns]
    
    # Features categóricas para encoding
    features_categoricas = ['experience_level', 'employment_type', 'job_title', 
                           'employee_residence', 'company_location', 'company_size']
    features_categoricas = [f for f in features_categoricas if f in df_modelo.columns]
    
    print(f"\n Features numéricas (sem leakage): {features_numericas_seguras}")
    print(f" Features categóricas: {features_categoricas[:3]}... ({len(features_categoricas)} total)")
    
    # Encoding de variáveis categóricas
    le_dict = {}
    for col in features_categoricas:
        le = LabelEncoder()
        df_modelo[col + '_encoded'] = le.fit_transform(df_modelo[col].astype(str))
        le_dict[col] = le
    
    # Selecionar features finais
    feature_cols = features_numericas_seguras + [col + '_encoded' for col in features_categoricas]
    
    # CRÍTICO: Remover 'salary' se ainda estiver presente
    feature_cols = [col for col in feature_cols if 'salary' not in col.lower() or col == col_salario]
    feature_cols = [col for col in feature_cols if col in df_modelo.columns and col != col_salario]
    
    X = df_modelo[feature_cols]
    y = df_modelo[col_salario]
    
    print(f"\n✓ Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"✓ CONFIRMAÇÃO: 'salary' não está nas features ✓")
    
    return X, y, feature_cols, le_dict, df_modelo

def treinar_modelos_com_validacao_temporal(X, y, df_modelo):
    """Treina modelos com validação temporal"""
    print("\n" + "="*60)
    print("TREINAMENTO COM VALIDAÇÃO TEMPORAL")
    print("="*60)
    
    # Ordenar por ano
    if 'work_year' in df_modelo.columns:
        df_sorted = df_modelo.sort_values('work_year').reset_index(drop=True)
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]
        
        # Split temporal: treinar até 2024, testar em 2025
        train_mask = df_sorted['work_year'] < 2025
        test_mask = df_sorted['work_year'] == 2025
        
        X_train = X_sorted[train_mask]
        y_train = y_sorted[train_mask]
        X_test = X_sorted[test_mask]
        y_test = y_sorted[test_mask]
        
        print(f"\nSplit temporal:")
        print(f"  Treino: {len(X_train)} amostras (2020-2024)")
        print(f"  Teste: {len(X_test)} amostras (2025)")
    else:
        # Fallback para split aleatório
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nSplit aleatório: {len(X_train)} treino, {len(X_test)} teste")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    modelos = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    resultados = {}
    
    print("\nTreinando modelos...")
    for nome, modelo in modelos.items():
        print(f"\n  {nome}...")
        
        if nome == 'Linear Regression':
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        resultados[nome] = {
            'modelo': modelo,
            'y_pred': y_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"    MAE: ${mae:,.2f}")
        print(f"    RMSE: ${rmse:,.2f}")
        print(f"    R²: {r2:.4f}")
    
    # Visualização
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Comparação de métricas
    metricas_df = pd.DataFrame({
        'MAE': [r['mae'] for r in resultados.values()],
        'RMSE': [r['rmse'] for r in resultados.values()],
        'R²': [r['r2'] for r in resultados.values()]
    }, index=resultados.keys())
    
    metricas_df[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0], alpha=0.7)
    axes[0, 0].set_title('Comparação de Erros (Validação Temporal)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Erro ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    metricas_df['R²'].plot(kind='bar', ax=axes[0, 1], alpha=0.7, color='green')
    axes[0, 1].set_title('Comparação de R² (Validação Temporal)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predição vs Real para melhor modelo
    melhor_modelo = max(resultados.items(), key=lambda x: x[1]['r2'])
    axes[1, 0].scatter(y_test, melhor_modelo[1]['y_pred'], alpha=0.5)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_title(f'Predição vs Real - {melhor_modelo[0]}', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Salário Real ($)')
    axes[1, 0].set_ylabel('Salário Predito ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resíduos
    residuos = y_test - melhor_modelo[1]['y_pred']
    axes[1, 1].scatter(melhor_modelo[1]['y_pred'], residuos, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_title('Análise de Resíduos', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Salário Predito ($)')
    axes[1, 1].set_ylabel('Resíduos ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'modelagem_corrigida.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: {output_path}")
    
    # Feature importance
    if 'Random Forest' in resultados:
        importancias = resultados['Random Forest']['modelo'].feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(importancias)[::-1][:15]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importancias[indices], alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importância')
        plt.title('Top 15 Features Mais Importantes (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'feature_importance_corrigida.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    return resultados, scaler, y_test

# ============================================================================
# 4. CLUSTERING OTIMIZADO (TESTAR 2-4 CLUSTERS)
# ============================================================================

def clustering_otimizado(df, X):
    """Clustering testando múltiplos valores de K"""
    print("\n" + "="*60)
    print("CLUSTERING OTIMIZADO (2-8 CLUSTERS)")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Testar range de clusters
    K_range = range(2, 8)
    inertias = []
    silhouette_scores = []
    
    print("\n Testando diferentes números de clusters...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        print(f"  K={k}: Silhouette={silhouette_scores[-1]:.4f}")
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    axes[0].set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Número de Clusters')
    axes[0].set_ylabel('Inércia')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='coral')
    axes[1].set_title('Silhouette Score por K', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Número de Clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].axhline(y=max(silhouette_scores), color='green', linestyle='--', alpha=0.5, label=f'Máximo: K={K_range[np.argmax(silhouette_scores)]}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'clustering_otimizacao.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Gráfico salvo: {output_path}")
    
    # Testar especificamente K=2, 3, 4
    col_salario = 'salary_in_usd' if 'salary_in_usd' in df.columns else 'salary'
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, k in enumerate([2, 3, 4]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_temp = df.copy()
        df_temp['cluster'] = clusters
        
        # PCA para visualização
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Scatter plot
        scatter = axes[0, idx].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                                       cmap='viridis', alpha=0.6, s=30)
        axes[0, idx].set_title(f'K={k} Clusters (Silhouette: {silhouette_score(X_scaled, clusters):.3f})', 
                               fontsize=12, fontweight='bold')
        axes[0, idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        # Boxplot
        df_temp.boxplot(column=col_salario, by='cluster', ax=axes[1, idx])
        axes[1, idx].set_title(f'Distribuição Salarial (K={k})', fontsize=12, fontweight='bold')
        axes[1, idx].set_xlabel('Cluster')
        axes[1, idx].set_ylabel('Salário (USD)')
        
        # Estatísticas
        print(f"\n--- K={k} CLUSTERS ---")
        cluster_stats = df_temp.groupby('cluster')[col_salario].agg(['mean', 'median', 'count'])
        print(cluster_stats)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'clustering_comparacao_k.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: {output_path}")
    
    # Escolher K ótimo
    k_otimo = K_range[np.argmax(silhouette_scores)]
    print(f"\n✓ K ótimo selecionado: {k_otimo}")
    
    # Clustering final com K ótimo
    kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
    df['cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # Caracterização detalhada
    print(f"\nCARACTERIZAÇÃO FINAL (K={k_otimo}):")
    for cluster_id in range(k_otimo):
        print(f"\n--- CLUSTER {cluster_id} ---")
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"Tamanho: {len(cluster_data)} profissionais ({len(cluster_data)/len(df)*100:.1f}%)")
        print(f"Salário médio: ${cluster_data[col_salario].mean():,.2f}")
        print(f"Salário mediano: ${cluster_data[col_salario].median():,.2f}")
        
        if 'experience_level' in cluster_data.columns:
            print(f"Experiência predominante: {cluster_data['experience_level'].mode()[0]}")
        if 'job_title' in cluster_data.columns:
            top_jobs = cluster_data['job_title'].value_counts().head(3)
            print(f"Top 3 cargos: {', '.join(top_jobs.index.tolist())}")
        if 'remote_ratio' in cluster_data.columns:
            print(f"Remote ratio médio: {cluster_data['remote_ratio'].mean():.0f}%")
    
    return df, kmeans_final, k_otimo

# ============================================================================
# 5. FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Executa todas as análises melhoradas"""
    print("\n" + "="*80)
    print("ANÁLISE MELHORADA DE SALÁRIOS EM DATA SCIENCE")
    print("="*80)
    
    # 1. Carregar e preparar dados
    df = carregar_dados(usar_kagglehub=True)
    df = preparar_dados(df)
    
    # 2. Investigar anomalia do trabalho remoto
    remote_analysis = investigar_trabalho_remoto(df)
    
    # 3. Modelagem preditiva corrigida
    X, y, features, encoders, df_modelo = preparar_features_para_modelo_corrigido(df)
    resultados, scaler, y_test = treinar_modelos_com_validacao_temporal(X, y, df_modelo)
    
    # 4. Clustering otimizado
    df_final, kmeans, k_otimo = clustering_otimizado(df, X)
    
    # 5. Resumo final
    print("\n" + "="*80)
    print("\nMelhorias implementadas:")
    print("  ✓ Data leakage corrigido (salary removido)")
    print("  ✓ Feature engineering aplicado (7 novas features)")
    print("  ✓ Validação temporal implementada")
    print("  ✓ Anomalia de trabalho remoto investigada")
    print(f"  ✓ Clustering otimizado (K={k_otimo})")
    
    print("\nArquivos gerados:")
    print("  • investigacao_remoto.png")
    print("  • modelagem_corrigida.png")
    print("  • feature_importance_corrigida.png")
    print("  • clustering_otimizacao.png")
    print("  • clustering_comparacao_k.png")
    
    # Salvar resultados
    output_csv = os.path.join(OUTPUT_DIR, 'dataset_analisado_melhorado.csv')
    df_final.to_csv(output_csv, index=False)
    print(f"  • dataset_analisado_melhorado.csv")
    
    # Comparação de R² antes e depois
    print("\nCOMPARAÇÃO DE PERFORMANCE:")
    print(f"  Depois (corrigido): R² = {max([r['r2'] for r in resultados.values()]):.4f}")
    
    return df_final, resultados, kmeans

if __name__ == "__main__":
    df_final, modelos, clustering = main()
