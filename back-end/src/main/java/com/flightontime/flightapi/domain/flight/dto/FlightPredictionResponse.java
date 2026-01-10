package com.flightontime.flightapi.domain.flight.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record FlightPredictionResponse(
        @JsonProperty("previsao")
        String prediction,

        @JsonProperty("probabilidade")
        Double probability,

        @JsonProperty("cor")
        String color,

        @JsonProperty("detalhes")
        FlightPredictionDetails details
){
    
}
