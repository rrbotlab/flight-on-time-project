package com.flightontime.flightapi.domain.airline;

import jakarta.persistence.*;
import lombok.Getter;

@Entity
@Table(name = "airlines")
@Getter
public class Airline {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "name")
    private String name;
}
