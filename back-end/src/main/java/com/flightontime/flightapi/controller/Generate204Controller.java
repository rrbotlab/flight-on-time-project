package com.flightontime.flightapi.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("generate_204")
public class Generate204Controller {

    @GetMapping
    public ResponseEntity<?> check(){
        return ResponseEntity.noContent().build();
    }
}
