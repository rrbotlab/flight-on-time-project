package com.flightontime.flightapi.service;

import com.flightontime.flightapi.domain.exception.AirportNotFoundException;
import com.flightontime.flightapi.domain.airport.AirportRepository;
import com.flightontime.flightapi.domain.airport.dto.AirportResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class AirportService {

    @Autowired
    private AirportRepository repository;

    @Cacheable(value = "airportResponseList")
    public List<AirportResponse> getAllAirports() {
        return repository.findAll()
                .stream()
                .map(a -> new AirportResponse(a.getIataCode(), a.getFullName()))
                .toList();
    }

    @Cacheable(value = "airportResponse", key = "#iataCode")
    public AirportResponse getAirportByIataCode(String iataCode) {
        return repository.findByIataCode(iataCode)
                .map(a -> new AirportResponse(a.getIataCode(), a.getFullName()))
                .orElseThrow(() -> new AirportNotFoundException("Aeroporto não encontrado: " + iataCode));
    }
}
