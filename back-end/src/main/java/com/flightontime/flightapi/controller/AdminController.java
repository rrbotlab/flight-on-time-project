package com.flightontime.flightapi.controller;

import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/admin")
@SecurityRequirement(name = "bearer-key")
public class AdminController {

    @GetMapping
    public ResponseEntity<?> admin(){
        Map<String,String> map = new HashMap<>();
        map.put("msg","admin endpoint");
        return ResponseEntity.ok(map);
    }

}
