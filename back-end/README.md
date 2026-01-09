# Flight On Time API

A Flight on Time é uma aplicação Back-End (REST) desenvolvida em Java com o framework Spring Boot. O objetivo principal
é fornecer previsões sobre o status de voos (atrasado ou pontual) utilizando o modelo de Data Science integrados via 
microserviço.

## Processo de Previsão (Dados -> Modelo -> Previsão)

O fluxo da aplicação segue três etapas principais:

1. **Entrada de Dados**: A API Java recebe via JSON os detalhes do voo (companhia, aeroportos, data e distância).
2. **Integração DS:** O serviço (`vooPrevisaoService`) comunica-se via `RestClient` com o microserviço de Data Science (Python/FastAPI).
3. **Inteligência Artifical:** O motor de IA utiliza um modelo **Random Forest Classifier**(Scikit-Learn) que 
   considera, além dos dados do voo, o impacto de **feriados nacionais** para calcular o risco.
4. **Resposta:** A API padroniza o retorno com a previsão, a probabilidade decimal e o nível de risco.

## Ferramentas e Dependências

- **Linguagem:** Java 21 (Versão utilizada no projeto)
- **Framework:** Spring Boot 3.5.4 
- **Banco de Dados**: MySQL com migrações via Flyway.
- **Documentação**: SpringDoc OpenAPI (Swagger).
- **Integração DS:** Python 3.10+, FastAPI e Scikit-Learn

## Como Executar o Projeto Localmente

**Pré-requisitos**

- Java 21 e Maven (ou use o ./mvnw incluso).
- MySQL rodando localmente.
- O microserviço de DataScience

**Passos**
1. **Clonar o repositório**:

```bash
git clone (decidir em qual repo vai ficar o projeto final)
cd flight on time
```

2. **Configurar o Banco de Dados**: Execute as migrações presentes em `src/main/resources/db/migrations` para criar 
   as tabelas de usuários e perfis.

3. **Configurar as variáveis de ambiente**: Defina as credenciais do banco e a URL dos serviços de DS:

- `FLIGHTONTIME_DATASOURCE_DEV`: URL do MySQL.
- `FLIGHTONTIME_DATASCIENCE_BASEURL`: URL do motor de IA.

4. **Executar a API**:

```bash
./mvnw spring-boot:run
```
5. **Acesso:** A documentação interativo estará disponível em `/swagger-ui.html`.

## Dataset Utilizado

O modelo foi treinado com o dataset [BrFlights2.csv](https://www.kaggle.com/datasets/ramirobentes/flights-in-brazil), 
que contém registros históricos de operações aéreas no 
Brasil. O target do modelo define como "Atrasado" voos com diferença superior a 15 minutos.

## Exemplos de Uso (Endpoint `/predict`)

O serviço expõe um endpoint `POST` que valida a presença de todos os campos obrigatórios antes de processar a consulta.

1. **Exemplo de Voo Pontual (Risco Baixo)**

**Requisição:**

```JSON
{
  "companhia": "AZUL",
  "origem": "GIG",
  "destino": "GRU",
  "data_partida": "2025-11-10T14:30:00",
  "distancia_km": 350
}
```
**Resposta (Probabilidade < 0.40):**

```JSON
{
  "previsao": "Pontual",
  "probabilidade": 0.15,
  "nivelRisco": "BAIXO",
  "mensagem": "Voo com alta probabilidade de pontualidade.",
  "detalhes": {"isFeriado":  false, "distanciaKm": 350.0}
}
```

2. **Exemplo de Voo Atrasado (Risco Alto)**
**Requisição (Feriado de Natal):**

```JSON
{
  "companhia": "AZUL",
  "origem": "GRU",
  "destino": "REC",
  "data_partida": "2025-12-25T14:30:00",
  "distancia_km": 2100.5
}
```

**Resposta (Probabilidade > 0.60):**

```JSON
{
  "previsao": "Atrasado",
  "probabilidade": 0.72,
  "nivelRisco": "ALTO",
  "mensagem": "Alta probabilidade de atraso devido ao feriado.",
  "detalhes": {"isFeriado":  true, "distanciaKm": 2689.0}
}
```

3. **Exemplo de Erro de Validação:**

Se um campo obrigatório como `data_partida` for omitido, a API retorna um erro padronizado:
**Resposta (400 Bad Request):**

```JSON
[
  {
    "campo": "data_partida",
    "mensagem": "não deve ser nulo"
  }
]
```