package com.flightontime.flightapi.controller;

import com.flightontime.flightapi.domain.airline.dto.AirlineResponse;
import com.flightontime.flightapi.service.AirlineService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/airlines")
public class AirlineController {

    @Autowired
    private AirlineService service;

    @GetMapping
    public ResponseEntity<List<AirlineResponse>> getAirlines() {
        List<AirlineResponse> response = service.getAllAirlines();
        return ResponseEntity.ok(response);
    }
}
