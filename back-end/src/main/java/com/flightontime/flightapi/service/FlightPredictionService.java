package com.flightontime.flightapi.service;

import com.flightontime.flightapi.domain.airline.Airline;
import com.flightontime.flightapi.domain.exception.AirlineNotFoundException;
import com.flightontime.flightapi.domain.exception.AirportNotFoundException;
import com.flightontime.flightapi.domain.exception.DataScienceApiOfflineException;
import com.flightontime.flightapi.domain.flight.dto.FlightPredictionResponse;
import com.flightontime.flightapi.domain.flight.dto.FlightRequest;
import com.flightontime.flightapi.infra.client.datascience.DataScienceClientInterface;
import com.flightontime.flightapi.infra.client.datascience.FlightPredictionMapper;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiRequest;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiResponse;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class FlightPredictionService {

    @Autowired
    private DataScienceClientInterface client;

    @Autowired
    private AirportService airportService;

    @Autowired
    private AirlineService airlineService;

    @CircuitBreaker(name = "externalService", fallbackMethod = "fallbackPredictDelay")
    public FlightPredictionResponse predictDelay(FlightRequest flightRequest) {
        String airlineName = airlineService.getAirlineByName(flightRequest.airline()).name();
        String airportOriginFullName = airportService.getAirportByIataCode(flightRequest.origin()).fullName();
        String airportDestinationFullName = airportService.getAirportByIataCode(flightRequest.destination()).fullName();
        DataScienceApiRequest apiRequest = FlightPredictionMapper.
                toDsApiRequest(flightRequest, airlineName, airportOriginFullName, airportDestinationFullName);
        DataScienceApiResponse apiResponse = client.getFlightPrediction(apiRequest);
        return FlightPredictionMapper.toFlightPredictionResponse(apiResponse);
    }

    public FlightPredictionResponse fallbackPredictDelay(FlightRequest flightRequest, Throwable t) {
        if(t instanceof AirportNotFoundException) {
            throw (AirportNotFoundException) t;
        }

        if(t instanceof AirlineNotFoundException) {
            throw (AirlineNotFoundException) t;
        }

        if(t instanceof CallNotPermittedException) {
            throw new DataScienceApiOfflineException("O circuito está aberto. Chamada ao serviço de Data Science interrompida");
        }

        throw new DataScienceApiOfflineException("O serviço de Data Science está fora do ar");
    }

}
