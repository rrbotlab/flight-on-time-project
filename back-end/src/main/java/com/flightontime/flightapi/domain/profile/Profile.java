package com.flightontime.flightapi.domain.profile;

import com.flightontime.flightapi.domain.user.User;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.util.Set;

@Setter
@Getter
@Entity
@Table(name = "perfis")
public class Profile {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "nome", nullable = false, length = 100)
    private String name;

    @ManyToMany(mappedBy = "profiles")
    private Set<User> users;


}
