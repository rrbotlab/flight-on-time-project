package com.flightontime.flightapi.infra.springdoc;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.security.SecurityScheme;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SpringDocConfigurations {

    @Value("${application.version}")
    private String version;

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .components(new Components()
                        .addSecuritySchemes("bearer-key",
                                new SecurityScheme()
                                        .type(SecurityScheme.Type.HTTP)
                                        .scheme("bearer")
                                        .bearerFormat("JWT")))
                        .info(new Info()
                                .title("Flight On Time - Java backend API")
                                .version(version)
                                .summary("summary: to do")
                                .description("description: to do")
                                .contact(new Contact()
                                        .name("Github")
                                        .url("https://github.com/rrbotlab/flight-on-time-api"))
                        .license(new License()
                                .name("MIT")));
    }

}
