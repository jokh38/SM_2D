#pragma once
#include "physics/step_control.hpp"
#include "physics/highland.hpp"
#include "physics/nuclear.hpp"
#include "physics/energy_straggling.hpp"

// Physics constants
constexpr float m_p = 938.272f;          // Proton mass [MeV/c²]
// X0_water is defined in highland.hpp (360.8f) to avoid circular dependency
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

// Physics process configuration for transport
// Allows selective enabling/disabling of physics processes for testing
struct PhysicsConfig {
    bool enable_straggling = true;   // Energy straggling (Vavilov model)
    bool enable_nuclear = true;      // Nuclear interactions (ICRU 63)
    bool enable_mcs = true;          // Multiple Coulomb Scattering (Highland)

    // Convenience helper for energy-loss-only mode (Bethe-Bloch only)
    static PhysicsConfig energy_loss_only() {
        return PhysicsConfig{false, false, false};
    }

    // Convenience helper for full physics (default)
    static PhysicsConfig full_physics() {
        return PhysicsConfig{true, true, true};
    }
};
