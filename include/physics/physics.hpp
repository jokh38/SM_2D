#pragma once
#include "physics/step_control.hpp"
#include "physics/highland.hpp"
#include "physics/nuclear.hpp"

// Physics constants
constexpr float m_p = 938.272f;          // Proton mass [MeV/c²]
constexpr float X0_water = 360.8f;       // Radiation length [mm]
constexpr float E_cutoff = 0.1f;         // Energy cutoff [MeV]
constexpr float weight_epsilon = 1e-12f; // Weight underflow threshold

// Component state for transport
struct ComponentState {
    float theta;  // Polar angle [rad]
    float E;      // Kinetic energy [MeV]
    float w;      // Statistical weight
    float x;      // Position x [mm]
    float z;      // Position z [mm]
    float mu;     // Direction cosine = cos(theta)
    float eta;    // Direction sine = sin(theta)
};
