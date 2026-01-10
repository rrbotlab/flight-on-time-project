package com.flightontime.flightapi.controller;

import com.flightontime.flightapi.domain.flight.dto.FlightPredictionResponse;
import com.flightontime.flightapi.domain.flight.dto.FlightRequest;
import com.flightontime.flightapi.service.FlightPredictionService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/predict")
public class FlightPredictionController {

    @Autowired
    private FlightPredictionService flightPredictionService;

    @PostMapping
    public ResponseEntity<FlightPredictionResponse> predictFlightDelay(@RequestBody @Valid FlightRequest flightRequest) {
        FlightPredictionResponse response = flightPredictionService.predictDelay(flightRequest);
        return ResponseEntity.ok(response);
    }

}

