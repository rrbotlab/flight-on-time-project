package com.flightontime.flightapi.domain.user.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.flightontime.flightapi.domain.user.User;

public record UserDataList(

        Long id,

        @JsonProperty("nome")
        String name

) {

    public UserDataList(User user) {
        this(
                user.getId(),
                user.getName()
                );
    }

}
