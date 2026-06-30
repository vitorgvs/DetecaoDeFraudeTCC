# Estrutura do Repositório Git - Projeto Data Science

Este documento apresenta a estrutura de diretórios e arquivos obrigatória para o repositório Git do projeto, conforme os requisitos estabelecidos. Cada integrante do grupo deve manter seu próprio repositório individual com esta organização.

---

## 📂 Árvore de Diretórios

```text
├── Dados/
│   ├── raw_data.csv
│   ├── clean_data.csv
│   └── abt.csv
├── DataPipeline/
│   ├── data_sanitization.py
│   ├── abt_transform.py
│   ├── exp_analysis.ipynb
│   └── pipeline_config.json       # Arquivo de configuração (variáveis, parâmetros e metadados)
├── Model/
│   ├── train.py
│   ├── model_config.json          # Arquivo de configuração (hiperparâmetros e metadados)
│   └── evaluation.ipynb
├── requirements.txt
└── Readme.md                      # Documentação principal do projeto
```

---

## 📝 Descrição Detalhada dos Componentes

### 1. `/Dados`
Diretório destinado ao armazenamento das bases de dados utilizadas nas diferentes etapas do ciclo de vida do modelo.
* **`raw_data.csv`**: Dado bruto original, sem nenhuma alteração ou tratamento prévio.
* **`clean_data.csv`**: Dado resultante do processo de sanitização e limpeza.
* **`abt.csv`**: *Analytical Base Table* (Tabela de Base Analítica). É a matriz final pronta para o modelo, contendo as variáveis explicativas (features) e a variável resposta (target) por linha de observação.

### 2. `/DataPipeline`
Contém os códigos responsáveis pela engenharia, tratamento e exploração dos dados.
* **`data_sanitization.py`**: Script em Python focado na limpeza, tratamento de valores nulos, duplicados, remoção de outliers e padronização dos dados brutos (`raw_data.csv` $\rightarrow$ `clean_data.csv`).
* **`abt_transform.py`**: Script que realiza as transformações matemáticas, encodings, criações de features (*feature engineering*) e consolida os dados limpos no formato final da ABT (`clean_data.csv` $\rightarrow$ `abt.csv`).
* **`exp_analysis.ipynb`**: Notebook Jupyter contendo a Análise Exploratória de Dados (EDA) realizada sobre os dados limpos, incluindo visualizações gráficas, correlações e estatísticas descritivas.
* **Arquivo de Configuração**: Centraliza variáveis globais do pipeline (ex: caminhos de arquivos, tipos de dados, colunas obrigatórias).

### 3. `/Model`
Concentra os artefatos relacionados à modelagem preditiva, treinamento e métricas de performance.
* **`train.py`**: Script estruturado para leitura da ABT, separação de bases (treino/teste/validação), treinamento do algoritmo escolhido e salvamento do modelo treinado.
* **`evaluation.ipynb`**: Notebook voltado para a avaliação detalhada do modelo utilizando métricas adequadas (Matriz de Confusão, Curva ROC-AUC, Precision-Recall, F1-Score) e análise de interpretabilidade (ex: SHAP, Feature Importance).
* **Arquivo de Configuração**: Parâmetros do modelo, semente aleatória (random state), hiperparâmetros de tunagem e metadados de versionamento.

### 4. Raiz do Repositório
* **`requirements.txt`**: Listagem com todas as dependências do projeto e suas respectivas versões (ex: `pandas`, `scikit-learn`, `numpy`).
* **`Readme.md`**: Guia principal de leitura e apresentação do projeto (veja o template abaixo).

---

## 📄 Template Sugerido para o `Readme.md` Principal

Abaixo está o esqueleto em Markdown pronto para ser copiado e preenchido no arquivo `Readme.md` da raiz do seu repositório:

```markdown
# Título do Projeto (Ex: Detecção de Fraudes Financeiras)

## 📝 Descrição do Projeto
[Insira aqui uma descrição contextualizada do projeto, a problemática abordada e o escopo da solução proposta.]

## 🎯 Objetivo de Negócio
* **Problema:** [O que a organização está perdendo ou deixando de otimizar?]
* **Meta de Negócio:** [Ex: Reduzir em X% o volume de fraudes financeiras sem impactar negativamente a experiência do cliente legítimo.]
* **Mapeamento Técnico:** [Como o modelo de Machine Learning resolve este problema.]

## 🧪 Resumo da Metodologia Utilizada
1.  **Sanitização:** [Breve explicação de como os dados brutos foram limpos].
2.  **Engenharia de Variáveis:** [Quais principais transformações ou features foram criadas para compor a ABT].
3.  **Modelagem:** [Quais algoritmos foram testados e qual foi o escolhido, justificando brevemente].
4.  **Avaliação:** [Resumo das principais métricas de validação atingidas e conclusões da interpretabilidade do modelo].

## ⚙️ Instruções de Como Treinar o Modelo

### Pré-requisitos
Certifique-se de ter o Python instalado (versão recomendada: 3.10+) e um ambiente virtual ativo.

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone <url-do-seu-repositorio>
cd <nome-do-repositorio>
pip install -r requirements.txt
```

### 2. Executar o Pipeline de Dados
Os dados brutos devem estar localizados em `/Dados/raw_data.csv`. Execute os scripts na ordem abaixo para gerar a ABT:
```bash
python DataPipeline/data_sanitization.py
python DataPipeline/abt_transform.py
```

### 3. Treinar o Modelo
Com a `abt.csv` gerada no diretório `/Dados`, execute o script de treinamento:
```bash
python Model/train.py
```
O modelo treinado e seus respectivos metadados serão salvos conforme configurado em `Model/model_config.json`.
```
