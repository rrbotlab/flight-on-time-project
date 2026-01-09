package com.flightontime.flightapi.domain.user;

import jakarta.validation.constraints.NotBlank;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Page<User> findAllByActiveTrue(Pageable pagination);
    Optional<User> findByEmail(String email);
    Optional<User> findByEmailAndActiveTrue(String email);
    boolean existsByEmail(String email);
}
