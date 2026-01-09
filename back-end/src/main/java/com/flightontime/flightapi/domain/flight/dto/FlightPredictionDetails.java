package com.flightontime.flightapi.domain.flight.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record FlightPredictionDetails(
        @JsonProperty("distancia")
        Double distance,

        @JsonProperty("chuva")
        Double rain,

        @JsonProperty("vento")
        Double wind,

        @JsonProperty("fonte_clima")
        String weatherSource
) {
}
