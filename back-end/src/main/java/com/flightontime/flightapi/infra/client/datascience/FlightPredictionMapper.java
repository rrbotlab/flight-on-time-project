package com.flightontime.flightapi.infra.client.datascience;

import com.flightontime.flightapi.domain.flight.dto.FlightPredictionDetails;
import com.flightontime.flightapi.domain.flight.dto.FlightPredictionResponse;
import com.flightontime.flightapi.domain.flight.dto.FlightRequest;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiRequest;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiResponse;

public class FlightPredictionMapper {

    public static DataScienceApiRequest toDsApiRequest(
            FlightRequest flightRequest, String airlineName, String airportOriginName, String airportDestinationName
    ) {
        DataScienceApiRequest apiRequest = new DataScienceApiRequest(
                airlineName,
                airportOriginName,
                airportDestinationName,
                flightRequest.departureDate()
        );
        return apiRequest;
    }

    public static FlightPredictionResponse toFlightPredictionResponse(DataScienceApiResponse apiResponse) {
        FlightPredictionDetails flightPredictionDetails = null;

        if (apiResponse.dados_utilizados() != null) {
            flightPredictionDetails = new FlightPredictionDetails(
                    apiResponse.dados_utilizados().distancia(),
                    apiResponse.dados_utilizados().chuva(),
                    apiResponse.dados_utilizados().vento(),
                    apiResponse.dados_utilizados().fonte_clima()
            );
        }

        FlightPredictionResponse flightPredictionResponse = new FlightPredictionResponse(
                apiResponse.previsao(),
                apiResponse.probabilidade(),
                apiResponse.cor(),
                flightPredictionDetails
        );
        return flightPredictionResponse;
    }

}
