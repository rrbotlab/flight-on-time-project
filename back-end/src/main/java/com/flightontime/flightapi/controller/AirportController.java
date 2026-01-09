package com.flightontime.flightapi.controller;

import com.flightontime.flightapi.domain.airport.dto.AirportResponse;
import com.flightontime.flightapi.service.AirportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
public class AirportController {

    @Autowired
    private AirportService service;

    @GetMapping("/airports")
    public ResponseEntity<List<AirportResponse>> getAllAirports() {
        List<AirportResponse> response = service.getAllAirports();
        return ResponseEntity.ok(response);
    }
}
