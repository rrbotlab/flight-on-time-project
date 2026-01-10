package com.flightontime.flightapi.infra.client.datascience.dto;

public record DataScienceResponseDetails(
        Double distancia,
        Double chuva,
        Double vento,
        String fonte_clima
) {}