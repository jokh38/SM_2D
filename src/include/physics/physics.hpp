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
//
// IMPORTANT: This is NOT a Monte Carlo code!
// Lateral spreading is ALWAYS enabled (deterministic, using Gaussian weight distribution)
// The enable_mcs flag has been removed - lateral spreading cannot be disabled.
struct PhysicsConfig {
    bool enable_straggling = true;   // Energy straggling (Vavilov model)
    bool enable_nuclear = true;      // Nuclear interactions (ICRU 63)
    // Lateral spreading is ALWAYS enabled (deterministic, not Monte Carlo)

    // Convenience helper for energy-loss-only mode (Bethe-Bloch only)
    // NOTE: Lateral spreading is still applied even in this mode
    static PhysicsConfig energy_loss_only() {
        return PhysicsConfig{false, false};
    }

    // Convenience helper for full physics (default)
    static PhysicsConfig full_physics() {
        return PhysicsConfig{true, true};
    }
};
