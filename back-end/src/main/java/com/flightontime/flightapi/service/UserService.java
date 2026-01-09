package com.flightontime.flightapi.service;

import com.flightontime.flightapi.domain.exception.ValidationException;
import com.flightontime.flightapi.domain.profile.Profile;
import com.flightontime.flightapi.domain.profile.dto.ProfileData;
import com.flightontime.flightapi.domain.profile.ProfileRepository;
import com.flightontime.flightapi.domain.user.*;
import com.flightontime.flightapi.domain.user.dto.UserDetailsResponse;
import com.flightontime.flightapi.domain.user.dto.UserRegistrationRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Set;
import java.util.stream.Collectors;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private ProfileRepository profileRepository;

    public UserDetailsResponse createUser(UserRegistrationRequest createUserRequest) {
        if (userRepository.existsByEmail(createUserRequest.email())) {
            throw new ValidationException("Já existe um usuário com o mesmo email");
        }

        String encodedPassword = passwordEncoder.encode(createUserRequest.password());
        User user = new User();
        user.setName(createUserRequest.name());
        user.setEmail(createUserRequest.email());
        user.setPassword(encodedPassword);
        user.setProfiles(Set.of(findUserProfile()));

        User createdUser = userRepository.save(user);
        Set<ProfileData> createdUserProfileData = createdUser
                .getProfiles()
                .stream()
                .map(p -> new ProfileData(p.getId(),p.getName()))
                .collect(Collectors.toSet());
        UserDetailsResponse response = new UserDetailsResponse(createdUser.getId(), createdUser.getName(), createdUser.getEmail(),createdUser.isActive(),createdUserProfileData);
        return response;
    }

    private Profile findUserProfile() {
        return profileRepository.findByName("ROLE_USER")
                .orElseThrow(() -> new RuntimeException("Perfil não encontrado"));
    }

}
