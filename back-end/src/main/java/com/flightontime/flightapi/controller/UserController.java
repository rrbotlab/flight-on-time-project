package com.flightontime.flightapi.controller;

import com.flightontime.flightapi.domain.user.UserRepository;
import com.flightontime.flightapi.domain.user.dto.UserDetailsResponse;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("users")
@SecurityRequirement(name = "bearer-key")
public class UserController {

    @Autowired
    private UserRepository repository;

    @GetMapping
    public ResponseEntity<Page<UserDetailsResponse>> listUsers(@PageableDefault(size = 10, sort = {"name"}) Pageable pagination){
        var page = repository.findAll(pagination).map(UserDetailsResponse::new);
        return ResponseEntity.ok(page);
    }

    @GetMapping("/ativos")
    public ResponseEntity<Page<UserDetailsResponse>> listActiveUsers(@PageableDefault(size = 10, sort = {"name"}) Pageable pagination){
        var page = repository.findAllByActiveTrue(pagination).map(UserDetailsResponse::new);
        return ResponseEntity.ok(page);
    }

    @GetMapping("/{id}")
    public ResponseEntity<UserDetailsResponse> getUser(@PathVariable(name = "id") Long id){
        var user = repository.getReferenceById(id);
        return ResponseEntity.ok(new UserDetailsResponse(user));
    }

}
