package com.flightontime.flightapi.controller;

import com.flightontime.flightapi.domain.user.dto.UserLoginRequest;
import com.flightontime.flightapi.infra.security.UserDetailsImpl;
import com.flightontime.flightapi.domain.user.dto.UserDetailsResponse;
import com.flightontime.flightapi.domain.user.dto.UserRegistrationRequest;
import com.flightontime.flightapi.infra.security.TokenDataResponse;
import com.flightontime.flightapi.infra.security.TokenService;
import com.flightontime.flightapi.service.UserService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    private AuthenticationManager manager;

    @Autowired
    private TokenService tokenService;

    @Autowired
    private UserService userService;

    @PostMapping("/login")
    public ResponseEntity<TokenDataResponse> login(@RequestBody @Valid UserLoginRequest data){
        var authenticationToken = new UsernamePasswordAuthenticationToken(data.email(), data.password());
        var authentication = manager.authenticate(authenticationToken);
        var jwtToken = tokenService.generateToken((UserDetailsImpl) authentication.getPrincipal());
        return ResponseEntity.ok(new TokenDataResponse(jwtToken));
    }

    @PostMapping("/create-user")
    public ResponseEntity<UserDetailsResponse> createUser(@RequestBody @Valid UserRegistrationRequest userRegistrationRequest) {
        UserDetailsResponse response = userService.createUser(userRegistrationRequest);
        return ResponseEntity.ok(response);
    }
}
