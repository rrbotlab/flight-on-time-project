package com.flightontime.flightapi.infra.client.datascience;

import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiRequest;
import com.flightontime.flightapi.infra.client.datascience.dto.DataScienceApiResponse;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.service.annotation.HttpExchange;
import org.springframework.web.service.annotation.PostExchange;

@HttpExchange
public interface DataScienceClientInterface {

    @PostExchange("/predict")
    DataScienceApiResponse getFlightPrediction(@RequestBody DataScienceApiRequest request);

}
