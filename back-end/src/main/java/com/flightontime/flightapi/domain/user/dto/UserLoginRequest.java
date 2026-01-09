package com.flightontime.flightapi.domain.user.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record UserLoginRequest(
        String email,
        @JsonProperty("senha")
        String password
) {
}
