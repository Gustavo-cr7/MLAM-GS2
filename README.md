# Global Solution 2026.2 — Machine Learning & Análise Multivariada (MLAM)
## Desafio CitizenTech: Acessibilidade Habitacional Urbana e Planejamento de Moradia Cidadã
 
> **Curso:** Bacharelado em Ciência da Computação — 2º Semestre (Turma `1CCPJ`)  
> **Disciplina:** Machine Learning & Análise Multivariada (MLAM)

---

## 👥 1. Identificação dos Integrantes

| Nome Completo | RM |
| :--- | :--- |
| **Gustavo de Souza Correa** | **RM 570746** |
| **Arthur Tae Gun Yong** | **RM 570647** |

---

## 🏙️ 2. Apresentação do Problema (Contexto CitizenTech) e Objetivo das Análises

No contexto do desafio **CitizenTech**, um dos problemas mais críticos enfrentados pelas cidades e comunidades urbanas contemporâneas é a **crise de acessibilidade habitacional e a pressão especulativa imobiliária**. O encarecimento acelerado da moradia afasta famílias trabalhadoras dos centros urbanos dotados de infraestrutura, transporte público, saúde e educação, agravando a segregação socioespacial e o déficit habitacional.

### Objetivo Geral
Investigar estatisticamente e modelar preditivamente os determinantes socioeconômicos e estruturais do **valor mediano dos imóveis (`Valor_Imovel`)** em 20.640 setores censitários urbanos, fornecendo subsídios quantitativos rigorosos para que **secretarias municipais de habitação, planejadores urbanos e plataformas cívicas (GovTechs/CitizenTechs)** possam:
1. Compreender a distribuição de renda familiar dos bairros e a probabilidade de vulnerabilidade socioeconômica (**Distribuição Normal — Item 02**);
2. Auditar variações amostrais de renda em pesquisas de campo por meio de inferência estatística formal (**Testes de Hipóteses Z — Item 03**);
3. Quantificar o impacto direto da renda familiar sobre o preço das moradias (**Análise Bivariada e Regressão Linear Simples — Item 04**);
4. Estimar o valor imobiliário combinando múltiplos fatores socioeconômicos e físicos do bairro — renda mediana, idade das edificações e porte total de cômodos (**Análise Multivariada e Regressão Linear Múltipla — Item 05**);
5. Classificar automaticamente distritos urbanos em zonas de **Valor Alto (acima da mediana)** versus **Valor Acessível (abaixo ou igual à mediana)** para priorização de zonas especiais de interesse social e subsídios habitacionais (**Regressão Logística Binária — Item 06**).

---

## 📊 3. Descrição da Base de Dados e de sua Fonte Oficial (Item 01 e Item 08.a)

- **Nome da Base Original:** *California Housing Prices* (Censo Demográfico e Habitacional dos EUA — *1990 US Census*, compilado por Pace & Barry, 1997)
- **Fonte Pública Oficial (Kaggle):** `camnugent/california-housing-prices`
- **Endereço de Acesso Direto (URL):** <https://www.kaggle.com/datasets/camnugent/california-housing-prices>
- **Volume de Dados:** `20.640` registros brutos e `20.640` registros válidos após seleção e limpeza das variáveis de estudo (0% de perda nos atributos selecionados). Cada registro representa um setor censitário (*block group*).

### Procedimentos de Limpeza, Seleção e Transformação Documentados (`analise_mlam.py`)
1. **Padronização de Nomenclatura:** As colunas originais do censo foram renomeadas para o português a fim de facilitar a leitura técnica nos modelos e gráficos:
   - `median_income` $\rightarrow$ **`Renda_Mediana`**: Renda anual mediana das famílias residentes no bairro (expressa em dezenas de milhares de dólares: $1{,}0 = \text{US\$ } 10.000{,}00$).
   - `housing_median_age` $\rightarrow$ **`Idade_Imovel`**: Idade mediana das residências do setor censitário (em anos).
   - `total_rooms` $\rightarrow$ **`Total_Comodos`**: Número total acumulado de cômodos em todas as moradias do bairro (indicador de porte físico residencial).
   - `population` $\rightarrow$ **`Populacao_Bairro`**: População total residente no setor censitário.
   - `median_house_value` $\rightarrow$ **`Valor_Imovel`**: Valor de mercado mediano das residências do bairro (em dólares americanos — US\$).
2. **Remoção de Ausentes (`dropna`):** Aplicado filtro `.dropna()` sobre o subconjunto `colunas_analise`, mantendo todos os `20.640` setores válidos.
3. **Engenharia da Variável-Alvo Binária (`Valor_Alto`):** Calculou-se a mediana populacional de `Valor_Imovel` ($\text{Mediana} = \text{US\$ } 179.700{,}00$) e criou-se a coluna binária `Valor_Alto`:
   - `Valor_Alto = 1` se `Valor_Imovel > 179700.0` (`10.317` bairros — 49,99%);
   - `Valor_Alto = 0` se `Valor_Imovel <= 179700.0` (`10.323` bairros — 50,01%).

---

## 📁 4. Estrutura e Localização dos Arquivos no Repositório

| Arquivo | Descrição |
| :--- | :--- |
| [`housing_kaggle.csv`](./housing_kaggle.csv) | Base de dados processada e limpa (`.csv`) pronta para modelagem (`20.640` registros e 6 colunas). |
| [`housing_original.csv`](./housing_original.csv) | Cópia integral da base bruta original baixada do Kaggle (`housing.csv`). |
| [`analise_mlam.ipynb`](./analise_mlam.ipynb) | Jupyter Notebook interativo com todos os códigos, cálculos, testes, regressões e **saídas numéricas e gráficos preservados**. |
| [`Relatorio_Estatistico_MLAM.pdf`](./Relatorio_Estatistico_MLAM.pdf) | **Relatório Estatístico Executivo em PDF (8 páginas A4)** com tabelas, gráficos, interpretações e limitações. |
| [`README.md`](./README.md) | Documentação completa do repositório, instruções de execução e síntese executiva dos resultados. |
| [`requirements.txt`](./requirements.txt) | Lista de bibliotecas Python e dependências necessárias para reprodução do projeto. |
| `02_distribuicao_normal.png` a `06_matriz_confusao.png` | Figuras estatísticas em alta resolução (`300 DPI`) geradas pelo código Python. |

---

## ⚙️ 5. Bibliotecas, Dependências e Orientações de Execução

### Pré-requisitos e Bibliotecas Necessárias
O projeto utiliza **Python 3.10+** e as seguintes bibliotecas científicas (listadas em [`requirements.txt`](./requirements.txt)):
- `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `kagglehub`, `pillow`.

### Passo a Passo para Executar os Códigos
1. **Clonar o repositório e acessar a pasta:**
   ```bash
   git clone <LINK_DO_REPOSITORIO_GITHUB>
   cd mlam
   ```
2. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Executar o script principal de análise estatística e geração de gráficos:**
   ```bash
   python analise_mlam.py
   ```
4. **Abrir o Notebook com as saídas preservadas (opcional):**
   ```bash
   jupyter notebook analise_mlam.ipynb
   ```

---

## 📈 6. Síntese dos Principais Resultados Obtidos (Itens 02 a 06)

### Item 02 — Distribuição Normal de Probabilidade (`Renda_Mediana`)
- **Parâmetros Populacionais ($N = 20.640$):** Média $\mu = 3{,}8707$ ($\text{US\$ } 38.706{,}71\text{/ano}$), Mediana $Md = 3{,}5348$ ($\text{US\$ } 35.348{,}00\text{/ano}$) e Desvio Padrão Populacional $\sigma = 1{,}8998$ ($\text{US\$ } 18.997{,}76\text{/ano}$).
- **a) $P(X < \text{Média})$:** $0{,}5000$ (**50,00%**) $\rightarrow$ **Evento Equiprovável (50%)**.
- **b) $P(X > \text{Mediana})$:** $0{,}5702$ (**57,02%**) $\rightarrow$ **Evento Provável (Moderadamente Provável)**.

### Item 03 — Testes de Hipóteses para a Média (Teste Z em `Renda_Mediana`)
- **a) Teste Unilateral à Esquerda ($n = 20$, $\alpha = 6\%$):** $Z_{\text{calc}} = \mathbf{+0{,}1328} \ge Z_{\text{crit}} = \mathbf{-1{,}5548}$ | $p\text{-valor} = \mathbf{0{,}5528}$ $\rightarrow$ **Não se rejeita $H_0$**.
- **b) Teste Unilateral à Direita ($n = 25$, $\alpha = 9\%$):** $Z_{\text{calc}} = \mathbf{-0{,}2893} \le Z_{\text{crit}} = \mathbf{+1{,}3408}$ | $p\text{-valor} = \mathbf{0{,}6138}$ $\rightarrow$ **Não se rejeita $H_0$**.

### Item 04 — Análise Bivariada e Regressão Linear Simples (`Renda_Mediana` $\rightarrow$ `Valor_Imovel`)
- **Covariância e Correlação de Pearson:** $\text{Cov}(X,Y) = \mathbf{150.847{,}4828}$ | $r = \mathbf{0{,}6881}$ ($p\text{-valor} < 10^{-300}$).
- **Equação:** $\widehat{\text{Valor\_Imovel}} = 45.085{,}5767 + 41.793{,}8492 \cdot \text{Renda\_Mediana}$.
- **Adequação:** $R^2 = \mathbf{0{,}4734}$ (**47,34%**) | Teste $F = \mathbf{18.556{,}57}$ ($p = 0{,}0000$).
- **Previsão (`Renda_Mediana = 5.0`):** $\widehat{Y} = \mathbf{\text{US\$ } 254.054{,}82}$.

### Item 05 — Análise Multivariada e Regressão Linear Múltipla
- **Equação:** $\widehat{\text{Valor\_Imovel}} = -24.329{,}8794 + 42.467{,}0389 \cdot \text{Renda\_Mediana} + 1.974{,}9806 \cdot \text{Idade\_Imovel} + 3{,}8878 \cdot \text{Total\_Comodos}$.
- **Adequação:** $R^2 = \mathbf{0{,}5137}$ (**51,37%**) | $R^2_{\text{ajustado}} = \mathbf{0{,}5136}$ | Teste $F = \mathbf{7.265{,}81}$ ($p = 0{,}0000$).
- **Previsão (`Renda=4.5`, `Idade=25`, `Comodos=2500`):** $\widehat{Y} = \mathbf{\text{US\$ } 225.865{,}82}$.

### Item 06 — Regressão Logística Binária (Alvo: `Valor_Alto`)
- **Equação Logit:** $z = -5{,}5847 + 1{,}1095 \cdot \text{Renda\_Mediana} + 0{,}0441 \cdot \text{Idade\_Imovel} + 0{,}000095 \cdot \text{Total\_Comodos}$ | $P(\text{Valor\_Alto} = 1) = \frac{1}{1 + e^{-z}}$.
- **Desempenho (Teste = 6.192 bairros):** **Acurácia = 76,42%** | $VN = 2.443$, $FP = 654$, $FN = 806$, $VP = 2.289$.
- **Classificação (`Renda=4.5`, `Idade=25`, `Comodos=2500`):** $P(\text{Valor\_Alto} = 1) = \mathbf{67{,}87\%}$ $\rightarrow$ **Classe Prevista = `1` (Valor Alto)**.

---

## ⚠️ 7. Limitações Identificadas nas Análises

1. **Censura Superior (*Top-Coding*) no Censo Original:** `Valor_Imovel` truncado em `US$ 500.001,00`, `Idade_Imovel` em `52 anos` e `Renda_Mediana` em `15,0001`.
2. **Assimetria Positiva vs. Normalidade:** A renda empírica possui cauda longa à direita ($\mu > Md$).
3. **Variáveis Geoespaciais Omitidas:** $R^2 = 51{,}37\%$ indica que 48,63% da variação decorre de fatores externos (latitude, longitude, proximidade ao litoral).
4. **Agregação Ecológica por Setor Censitário:** Os dados representam medianas de bairros, não imóveis individuais.
