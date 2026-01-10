package com.flightontime.flightapi.domain.user.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.flightontime.flightapi.domain.profile.dto.ProfileData;
import com.flightontime.flightapi.domain.user.User;

import java.util.Set;
import java.util.stream.Collectors;

public record UserDetailsResponse(

        Long id,

        @JsonProperty("nome")
        String name,

        String email,

        @JsonProperty("ativo")
        boolean active,

        @JsonProperty("perfis")
        Set<ProfileData> profiles

) {

    public UserDetailsResponse(User user) {
        this(
                user.getId(),
                user.getName(),
                user.getEmail(),
                user.isActive(),
                user.getProfiles()
                        .stream()
                        .map(p -> new ProfileData(p.getId(), p.getName()))
                        .collect(Collectors.toSet())
                );
    }

}
