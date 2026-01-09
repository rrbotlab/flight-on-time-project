package com.flightontime.flightapi.infra.client.datascience;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpStatusCode;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.support.RestClientAdapter;
import org.springframework.web.service.invoker.HttpServiceProxyFactory;

import java.util.Optional;

@Configuration
public class DataScienceClientConfig {

    @Value("${datascience.api.flightontime-url}")
    private String url;

    @Bean
    public DataScienceClientInterface dataScienceClient(RestClient.Builder restClientBuilder) {
        RestClient restClient = restClientBuilder
                .baseUrl(url)
                .defaultStatusHandler(HttpStatusCode::is4xxClientError,
                        (request, response) -> Optional.empty())
                .build();
        RestClientAdapter adapter = RestClientAdapter.create(restClient);
        HttpServiceProxyFactory factory = HttpServiceProxyFactory.builderFor(adapter).build();
        return factory.createClient(DataScienceClientInterface.class);
    }

}
