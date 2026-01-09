package com.flightontime.flightapi.domain.user.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

public record UserRegistrationRequest(
        @NotBlank
        @JsonProperty("nome")
        String name,

        @NotBlank
        String email,

        @NotBlank
        @JsonProperty("senha")
        String password
){
}