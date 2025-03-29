# Resumo do Projeto: Modelo de Machine Learning para Classificação de Crédito

## Visão Geral
Este projeto implementa um modelo de Machine Learning para classificação de crédito usando o algoritmo Naive Bayes Gaussiano. O objetivo é prever se um cliente é um bom ou mau pagador ("good" ou "bad") com base em diversas características financeiras e pessoais.

## Bibliotecas Utilizadas
- **MLflow**: Para rastreamento de experimentos e métricas.
- **Pandas**: Para manipulação e análise de dados.
- **NumPy**: Para operações numéricas.
- **Scikit-learn**: Para:
  - Divisão de dados (`train_test_split`)
  - Métricas de avaliação (`accuracy_score`, `recall_score`, `precision_score`, `f1_score`, `roc_auc_score`, `log_loss`, `confusion_matrix`)
  - Implementação do modelo (`GaussianNB`)

## Pré-processamento de Dados
1. **Carregamento dos Dados**:  
   - Os dados foram carregados a partir de um arquivo CSV chamado `credit.csv`, contendo 1000 entradas e 21 colunas.

2. **Limpeza dos Dados**:  
   - Remoção de valores nulos (`dropna`).
   - Eliminação de duplicatas (`drop_duplicates`).
   - Normalização dos nomes das colunas (conversão para minúsculas e substituição de espaços por underscores).

3. **Codificação de Variáveis Categóricas**:  
   - Todas as colunas do tipo `object` foram convertidas para códigos numéricos usando `astype('category').cat.codes`.

## Divisão dos Dados
- Os dados foram divididos em conjuntos de treinamento (70%) e teste (30%) usando `train_test_split`, com `random_state=123` para reprodutibilidade.

## Treinamento do Modelo
- **Algoritmo**: Naive Bayes Gaussiano (`GaussianNB`).
- **Experimento MLflow**:  
  - Foi criado um experimento chamado `NB_modulo4_MLOPS` para rastrear as métricas do modelo.
  - As métricas registradas incluem:
    - Acurácia
    - Recall
    - Precisão
    - F1-score
    - AUC (Área sob a curva ROC)
    - Log Loss

## Métricas de Desempenho
O modelo alcançou os seguintes resultados no conjunto de teste:
- **Acurácia**: 71.67%
- **Recall**: 77.00%
- **Precisão**: 79.79%
- **F1-score**: 78.37%
- **AUC**: 69.00%
- **Log Loss**: 10.21

## Arquivos do Projeto
- **`contrucaoML.ipynb`**: Notebook contendo todo o código, desde a importação dos dados até a avaliação do modelo.
- **`credit.csv`**: Conjunto de dados utilizado para treinamento e teste (não incluído no resumo, mas essencial para reprodução).

## Instruções para Reprodução
1. Certifique-se de ter as bibliotecas instaladas (`mlflow`, `pandas`, `numpy`, `scikit-learn`).
2. Execute o notebook `contrucaoML.ipynb` em um ambiente Python compatível.
3. Verifique se o arquivo `credit.csv` está no mesmo diretório que o notebook.

## Observações
- O projeto demonstra um fluxo completo de Machine Learning, desde o pré-processamento até a avaliação do modelo, com o uso do MLflow para rastreamento de experimentos.
- As métricas sugerem que o modelo tem um desempenho razoável, mas há espaço para melhorias, como a exploração de outros algoritmos ou ajustes de hiperparâmetros.
