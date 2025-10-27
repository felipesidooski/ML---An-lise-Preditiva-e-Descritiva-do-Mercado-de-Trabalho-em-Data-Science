import os
from analise_salarios_datascience import *

# ============================================
# OPÇÃO 1: Executar análise completa automática
# ============================================
print("OPÇÃO 1: Análise Completa Automática (com kagglehub)")
print("="*50)

# Basta executar a função main() - baixa automaticamente do Kaggle
# df_final, modelos, clustering = main()


# ============================================
# OPÇÃO 2: Executar análises individuais
# ============================================
print("\n\nOPÇÃO 2: Análises Individuais")
print("="*50)

# 1. Carregar dados (baixa automaticamente do Kaggle)
print("\n🔄 Baixando dataset do Kaggle...")
df = carregar_dados(usar_kagglehub=True)
df = preparar_dados(df)

# 2. Análises Temporais
print("\n--- Análises Temporais ---")
evolucao = analise_evolucao_salarial(df)
evolucao_cargo = analise_por_cargo(df)
impacto = analise_impacto_pandemia(df)
sazonalidade = analise_sazonalidade(df)

# 3. Modelagem Preditiva
print("\n--- Modelagem Preditiva ---")
X, y, features, encoders = preparar_features_para_modelo(df)
modelos, scaler = treinar_modelos_preditivos(X, y)

# 4. Clustering
print("\n--- Clustering de Perfis ---")
df_clusters, clusters, kmeans = clustering_perfis(df, X)

# 5. Salvar resultados em CSV
print("\n--- Salvando Resultados ---")
if not os.path.exists('outputs'):
    os.makedirs('outputs')
df_clusters.to_csv('outputs/dataset_com_clusters.csv', index=False)
print("✓ Dataset com clusters salvo!")


# ============================================
# OPÇÃO 3: Análises customizadas
# ============================================
print("\n\nOPÇÃO 3: Análises Customizadas")
print("="*50)

# Exemplo: Análise específica por nível de experiência
if 'experience_level' in df.columns:
    print("\n📊 Análise por Nível de Experiência:")
    exp_analysis = df.groupby('experience_level')['salary_in_usd'].agg([
        'mean', 'median', 'min', 'max', 'count'
    ])
    print(exp_analysis)

# Exemplo: Top 5 países com maiores salários
if 'company_location' in df.columns:
    print("\n🌍 Top 5 Países com Maiores Salários:")
    top_paises = df.groupby('company_location')['salary_in_usd'].mean().sort_values(ascending=False).head(5)
    print(top_paises)

# Exemplo: Análise de trabalho remoto
if 'remote_ratio' in df.columns:
    print("\n💻 Impacto do Trabalho Remoto:")
    remote_analysis = df.groupby('remote_ratio')['salary_in_usd'].agg(['mean', 'count'])
    print(remote_analysis)


print("\n\n" + "="*50)
print("✅ Análises concluídas!")
print("="*50)
print("\n📁 Arquivos gerados na pasta '/mnt/user-data/outputs/':")
print("  • evolucao_salarial.png")
print("  • evolucao_por_cargo.png")
print("  • impacto_pandemia.png")
print("  • sazonalidade_contratacoes.png")
print("  • modelagem_preditiva.png")
print("  • feature_importance.png")
print("  • elbow_method.png")
print("  • clustering_perfis.png")
print("  • dataset_com_clusters.csv")