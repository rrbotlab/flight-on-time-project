package com.flightontime.flightapi.service;

import com.flightontime.flightapi.domain.airline.Airline;
import com.flightontime.flightapi.domain.airline.AirlineRepository;
import com.flightontime.flightapi.domain.airline.dto.AirlineResponse;
import com.flightontime.flightapi.domain.exception.AirlineNotFoundException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class AirlineService {

    @Autowired
    private AirlineRepository repository;

    @Cacheable(value = "airlineResponseList")
    public List<AirlineResponse> getAllAirlines() {
        return repository.findAll()
                .stream()
                .map(a -> new AirlineResponse(a.getName()))
                .toList();
    }

    public AirlineResponse getAirlineByName(String name) {
        return repository.findByName(name)
                .map(a -> new AirlineResponse(a.getName()))
                .orElseThrow(() -> new AirlineNotFoundException("Companhia aérea não encontrada"));
    }

}
