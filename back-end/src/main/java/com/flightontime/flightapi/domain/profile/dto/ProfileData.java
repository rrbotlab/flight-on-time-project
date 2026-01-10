package com.flightontime.flightapi.domain.profile.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

public record ProfileData(

        Long id,

        @NotBlank
        @JsonProperty("nome")
        String name

){
}
