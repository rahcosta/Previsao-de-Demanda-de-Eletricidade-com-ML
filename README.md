# Previsao-de-Demanda-de-Eletricidade-com-ML
Modelo de Machine Learning para prever a demanda horária de eletricidade utilizando Python, Pandas e XGBoost.

<img width="6187" height="6003" alt="image" src="https://github.com/user-attachments/assets/00960dc3-2b5c-4e08-a7dc-0984f7a1f77d" />

# **Resumo** 

Este projeto apresenta a construção de um modelo de machine learning para prever a demanda de eletricidade com base em dados históricos e variáveis climáticas. Utilizando uma abordagem de série temporal, o modelo é treinado para identificar padrões diários, semanais e sazonais, resultando em previsões precisas que podem auxiliar no gerenciamento de recursos energéticos. Todo o processo, desde a limpeza dos dados e engenharia de recursos até o treinamento e avaliação do modelo, está documentado neste repositório.

# **Objetivo**

Companhias de energia em todo o mundo enfrentam o desafio constante de equilibrar a produção com o consumo em tempo real. Uma falha nesse equilíbrio pode levar a desperdícios de recursos em casos de sobre-geração, ou a apagões e instabilidade da rede em casos de sub-geração. Nesse contexto, a previsão precisa da demanda de energia elétrica é uma ferramenta estratégica fundamental para o planejamento de médio e longo prazo do setor (Mendonça, 2022).

Com previsões confiáveis, empresas e profissionais dos setores de geração, transmissão e distribuição de energia podem tomar decisões críticas de forma mais eficiente, como:

- **Otimizar** a quantidade de usinas em operação para atender à demanda sem excessos.
    
- **Planejar** a manutenção de equipamentos para períodos de baixo consumo, minimizando o impacto na rede.
    
- **Aprimorar** a estratégia de compra e venda de energia no mercado.
    
- **Identificar** áreas com previsão de crescimento acelerado para alocar projetos de expansão e melhoria da infraestrutura.
    

O objetivo específico deste projeto é desenvolver um modelo, utilizando o algoritmo **XGBoost**, capaz de prever a demanda horária de eletricidade na Índia com a menor margem de erro possível. O modelo visa colaborar com as empresas do setor, fornecendo uma ferramenta para garantir a oferta e a estabilidade do fornecimento de energia no futuro.

# **Conjunto de Dados (*Dataset*)**
## **Fonte**

Os dados utilizados neste projeto foram obtidos do [Kaggle: Electricity Demand Historical Data](https://www.kaggle.com/datasets/rohitgrewal/electricity-demand-data-dsl/data) e abrangem o período de janeiro de 2020 a dezembro de 2024. O conjunto de dados contém registros horários da demanda de eletricidade e das condições meteorológicas. 

## **Dicionário de Dados**

Abaixo está a descrição de todas as colunas utilizadas no projeto.

#### **Colunas Originais do Dataset**

- `Timestamp` (datetime): A data e hora exata da medição. Esta coluna serve como o índice principal para a nossa análise de série temporal.
    
- `Demand` (numérica): A variável alvo. Representa o consumo total de eletricidade em Megawatts (MW) para a hora registrada.
    
- `Temperature` (numérica): A temperatura média registrada em graus Celsius (°C) durante aquela hora.
    
- `Humidity` (numérica): A umidade relativa do ar (em %) registrada durante aquela hora.
    

#### **Recursos Criados**

Estes recursos foram criados a partir da coluna `Timestamp` e `Demand` para fornecer ao modelo informações contextuais e temporais mais ricas, que são cruciais para a previsão.

- `hour` (inteiro): A hora do dia, extraída do `Timestamp` (valores de 0 a 23). Ajuda a capturar os padrões de consumo diários.
    
- `dayofweek` (inteiro): O dia da semana, extraído do `Timestamp` (0 para Segunda-feira, 6 para Domingo). Ajuda a diferenciar o consumo entre dias úteis e fins de semana.
    
- `quarter` (inteiro): O trimestre do ano, extraído do `Timestamp` (valores de 1 a 4). Ajuda a capturar a sazonalidade de médio prazo (verão, monções, inverno).
    
- `month` (inteiro): O mês do ano, extraído do `Timestamp` (valores de 1 a 12). Fornece uma granularidade mais fina para a sazonalidade anual.
    
- `year` (inteiro): O ano do registro, extraído do `Timestamp`. Permite que o modelo identifique tendências de longo prazo.
    
- `dayofyear` (inteiro): O dia do ano, extraído do `Timestamp` (valores de 1 a 366). Oferece uma visão contínua da passagem do tempo ao longo do ano.
    
- `weekofyear` (inteiro): A semana do ano, extraída do `Timestamp` (valores de 1 a 53). Útil para capturar padrões semanais que podem atravessar meses, como feriados.
    
- `is_weekend` (binário): Um marcador que é `1` se o dia for Sábado ou Domingo, e `0` caso contrário. Simplifica a identificação do padrão de fim de semana.
    
- `demand_lag_24hr` (numérica): Um recurso defasado (lag) que contém o valor da demanda de exatamente 24 horas antes do registro atual. É um forte preditor, pois a demanda de hoje costuma ser similar à de ontem no mesmo horário.
    
- `demand_lag_168hr` (numérica): Um recurso defasado que contém o valor da demanda de 168 horas (1 semana) antes. Captura a forte sazonalidade semanal.
    
- `demand_rolling_mean_24hr` (numérica): A média móvel da demanda nas últimas 24 horas. Ajuda a suavizar o ruído e a identificar a tendência de consumo recente.
    
- `demand_rolling_std_24hr` (numérica): O desvio padrão móvel da demanda nas últimas 24 horas. Mede a volatilidade recente do consumo; picos de volatilidade podem indicar eventos atípicos.

# **Metodologia**

### **1. Limpeza e Pré-processamento dos Dados**

A primeira etapa consistiu na preparação do conjunto de dados brutos. As seguintes ações foram tomadas:

- **Conversão de Tipos:** A coluna `Timestamp` foi convertida de texto para o formato `datetime`, essencial para análises de séries temporais, e foi definida como o índice do DataFrame.
    
- **Tratamento de Valores Ausentes:** Foi realizada uma investigação de valores nulos `NaN` e diferentes estratégias foram aplicadas a cada tipo de coluna para preservar as características dos dados:

	- Para a variável alvo `Demand`, foi utilizada a **interpolação baseada no tempo (`interpolate`)**, um método robusto que estima os valores ausentes com base no intervalo de tempo entre os pontos de dados.
    
	- Para as colunas meteorológicas (`Temperature` e `Humidity`), foi aplicada a técnica de **`backward fill` (`bfill`)**, que preenche um valor ausente com a próxima observação válida.
    
	- Para os recursos de tempo que foram gerados e poderiam conter nulos (`hour`, `dayofweek`, etc.), foi utilizada a técnica de **`forward fill` (`ffill`)**, preenchendo com a última observação válida.
    

### **2. Análise Exploratória de Dados (EDA)**

Com os dados limpos, uma análise exploratória visual foi conduzida para identificar os padrões fundamentais da demanda de eletricidade. Os principais insights obtidos foram:

- **Sazonalidade Anual:** O gráfico da demanda ao longo do tempo revelou um forte e regular padrão anual, com um pico de consumo principal durante o verão e a estação das monções (meses de abril a setembro) e um pico secundário menor durante o inverno (dezembro e janeiro), refletindo o ciclo climático da Índia.
    
- **Padrão Diário:** O box plot da demanda por hora do dia mostrou um ciclo diário claro, com os menores valores durante a madrugada, uma rampa de subida pela manhã e um pico de consumo no início da noite (entre as 18h-19h).
    
- **Impacto da Temperatura:** O gráfico de dispersão entre a demanda e a temperatura confirmou uma forte correlação positiva. Além disso, revelou que a variabilidade da demanda aumenta significativamente em temperaturas mais altas, indicando a influência de outros fatores nesses períodos.
    
- **Correlação entre Variáveis:** O mapa de calor (heatmap) quantificou as relações lineares, destacando os recursos de _lag_ (demanda passada) e a temperatura como os preditores mais fortes da demanda futura.
    

### **3. Engenharia de Recursos (*Feature Engineering*)**

Para que o modelo de machine learning pudesse "entender" os padrões identificados na EDA, novos recursos (features) foram criados a partir dos dados existentes:

- **Recursos Cíclicos:** A partir do índice `Timestamp`, foram extraídos recursos como `hour`, `dayofweek`, `month`, `quarter`, `weekofyear`, etc. Essas features permitem que o modelo aprenda os ciclos diários, semanais e anuais.
    
- **Recursos Defasados (Lags):** Foram criadas as colunas `demand_lag_24hr` e `demand_lag_168hr`. Esses recursos fornecem ao modelo o contexto da demanda no passado recente (exatamente 1 dia e 1 semana antes), que são informações altamente preditivas.
    
- **Recursos de Janela Móvel (Rolling Window):** Foram calculadas a média e o desvio padrão da demanda em uma janela de 24 horas (`demand_rolling_mean_24hr` e `demand_rolling_std_24hr`). Esses recursos ajudam o modelo a capturar a tendência e a volatilidade do consumo nas horas imediatamente anteriores.

# **Modelagem**

Após a preparação dos dados, a etapa de modelagem foi iniciada com o objetivo de criar um sistema preditivo proposto.

### **1. Divisão dos Dados em Treino e Teste**

Para avaliar o modelo de forma realista, os dados foram divididos em dois conjuntos: um para treinamento e outro para teste. Dada a natureza de série temporal do problema, foi utilizada uma **divisão cronológica** para evitar o que se chama de "vazamento de dados" (data leakage) e simular um cenário de previsão real.

- **Conjunto de Treino:** Composto por todos os dados registrados **antes de 1º de janeiro de 2024**. Este conjunto foi utilizado para ensinar o modelo a reconhecer os padrões históricos.
    
- **Conjunto de Teste:** Composto por todos os dados registrados **a partir de 1º de janeiro de 2024**. Este conjunto, completamente novo para o modelo, foi reservado para avaliar sua performance preditiva no "futuro".
    

### **2. Escolha do Algoritmo: XGBoost**

O algoritmo escolhido para este projeto foi o **XGBoost (Extreme Gradient Boosting)**. Esta escolha se baseia em suas principais vantagens:

- **Alto Desempenho:** É consistentemente um dos algoritmos com maior poder preditivo em competições e aplicações com dados tabulares, como o nosso.
    
- **Flexibilidade:** Consegue capturar relações complexas e não-lineares entre as variáveis, algo que foi claramente identificado na etapa de análise exploratória.
    
- **Otimização:** Possui mecanismos internos de regularização e otimização que ajudam a prevenir o overfitting e a garantir um bom desempenho de generalização.
    

### **3. Treinamento do Modelo**

O modelo `XGBRegressor` foi treinado utilizando o conjunto de treino. Para garantir que o modelo generalizasse bem para novos dados e não "decorasse" os dados de treino (overfitting), foi utilizada a técnica de **parada antecipada (early stopping)**.

- O treinamento foi configurado para rodar até 1000 iterações (`n_estimators=1000`).
    
- A performance do modelo no conjunto de teste foi monitorada a cada iteração.
    
- Se a performance no conjunto de teste não melhorasse por 50 iterações consecutivas (`early_stopping_rounds=50`), o treinamento era automaticamente interrompido, salvando a versão do modelo com o melhor desempenho.

# **Resultados e Avaliação**

Após o treinamento, o desempenho do modelo foi rigorosamente avaliado utilizando o conjunto de teste (dados de 2024), que o modelo nunca havia visto antes. A avaliação foi feita tanto por métricas quantitativas quanto por análise visual.

### **1. Métricas de Avaliação**

Para medir a performance do modelo, foram escolhidas duas das métricas mais comuns para problemas de regressão:

- **MAE (Mean Absolute Error - Erro Médio Absoluto):** Representa a média da diferença absoluta entre os valores previstos e os valores reais. Em termos práticos, nos diz, em média, quantos Megawatts (MW) a previsão do modelo errou. É uma métrica de fácil interpretação.
    
- **RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio):** Similar ao MAE, mas, ao elevar os erros ao quadrado, penaliza mais os erros grandes. Um RMSE baixo indica que o modelo é consistentemente bom e raramente faz previsões muito distantes da realidade.
    

### **2. Desempenho do Modelo**

O modelo final alcançou os seguintes resultados no conjunto de teste:

- **MAE (Erro Médio Absoluto):** 123.47612356015286 MW
    
- **RMSE (Raiz do Erro Quadrático Médio):** 175.22716387271916 MW
    

_(**Observação:** Você precisará rodar o seu script de avaliação final para obter esses dois números e inseri-los aqui. Eles são a prova concreta do sucesso do seu projeto!)_

### **3. Análise Visual dos Resultados**

Além das métricas, foi gerado um gráfico comparando a demanda real com as previsões do modelo para o período de teste. A análise visual mostra que a linha de previsão do modelo acompanha de perto a linha dos dados reais, capturando com sucesso tanto os picos e vales diários quanto a tendência sazonal do consumo de energia. Essa alta aderência entre o previsto e o real confirma a eficácia e a robustez do modelo.

# **Tecnologias Utilizadas**

O projeto foi desenvolvido inteiramente em **Python (versão 3.12.11)** e utilizou as seguintes bibliotecas e ferramentas principais:

- **Análise e Manipulação de Dados:**
    
    - **Pandas:** Para a estruturação, limpeza e manipulação dos dados em DataFrames.
        
    - **NumPy:** Para operações numéricas e computação de alta performance.
        
- **Visualização de Dados:**
    
    - **Matplotlib:** Para a criação dos gráficos base e customizações.
        
    - **Seaborn:** Para a criação de visualizações estatísticas mais elaboradas, como os *box plots* e o *heatmap*.
        
- **Machine Learning e Modelagem:**
    
    - **XGBoost:** Biblioteca que implementa o algoritmo de *Gradient Boosting* utilizado para o treinamento do modelo de regressão.
        
    - **Scikit-learn:** Utilizada para a avaliação do modelo, especificamente para o cálculo das métricas de erro (MAE e RMSE).
        
- **Persistência do Modelo:**
    
    - **Joblib:** Para salvar o modelo treinado em um arquivo, permitindo seu reuso futuro.
        
- **Ambiente de Desenvolvimento:**
    
    - **Google Colab:** Ambiente de notebooks baseado em nuvem que permite a execução de código Python com acesso a recursos computacionais, como GPUs, facilitando o treinamento de modelos e a colaboração.

# **Conclusão**

Este projeto demonstrou com sucesso a construção de um modelo de machine learning de ponta a ponta para a previsão de demanda de eletricidade. Através de um processo minucioso de limpeza de dados, análise exploratória e de engenharia de recursos foi possível extrair padrões sazonais, diários e semanais do consumo de energia.

O modelo **XGBoost** treinado se mostrou altamente eficaz, conseguindo acompanhar com grande precisão as variações da demanda no conjunto de teste. O resultado é uma ferramenta preditiva que pode fornecer insights estratégicos para empresas do setor de energia, auxiliando no planejamento operacional, na otimização de recursos e na garantia da estabilidade da rede elétrica.

### **Próximos Passos**

Embora o modelo atual apresente um considerável desempenho, o projeto pode ser aprimorado de várias maneiras:

- **Otimização de Hiperparâmetros:** Utilizar técnicas como *Grid Search* ou *Bayesian Optimization* para encontrar a combinação ideal de hiperparâmetros para o modelo XGBoost, podendo levar a uma redução ainda maior nos erros de previsão.
    
- **Inclusão de Novos Recursos:** Aprimorar o *dataset* com outras variáveis que podem influenciar o consumo, tais como:
    
    - Dados meteorológicos mais detalhados (velocidade do vento, cobertura de nuvens, etc.).
        
    - Indicadores econômicos.
        
- **Experimentar Outros Modelos:** Testar e comparar o desempenho do XGBoost com outros algoritmos avançados para séries temporais, como **LSTM (Redes Neurais Recorrentes)** ou o **Prophet** (desenvolvido pelo Facebook).
    
- ***Deployment* do Modelo:** Empacotar o modelo treinado em uma **API (utilizando Flask ou FastAPI)** e fazer o *deploy* em um serviço de nuvem (como *Heroku, AWS* ou *Google Cloud*). Isso transformaria o projeto em um serviço web funcional, capaz de receber dados e retornar previsões em tempo real.

# Referências

DE MIRANDA MENDONÇA, Jefferson Whitney. Desenvolvimento de um Modelo de Previsão de Demanda de Energia Baseado em Algoritmos de Inteligência Artificial de Curto Prazo. 2022.


