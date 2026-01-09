package com.flightontime.flightapi.infra.client.datascience.dto;

public record DataScienceApiResponse(
   String previsao,
   Double probabilidade,
   String cor,
   DataScienceResponseDetails dados_utilizados
) {}