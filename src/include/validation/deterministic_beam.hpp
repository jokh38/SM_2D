#pragma once
#include "core/grids.hpp"
#include "core/psi_storage.hpp"
#include <vector>

/**
 * @file deterministic_beam.hpp
 * @brief Deterministic S-Matrix Transport Configuration
 *
 * ALGORITHM: Hierarchical Deterministic Transport
 * ================================================
 * This is NOT Monte Carlo! This uses deterministic Boltzmann equation solving.
 */

struct PencilBeamConfig {
    float E0 = 150.0f;
    float x0 = 0.0f;
    float z0 = 0.0f;
    float theta0 = 0.0f;
    float sigma_theta0 = 0.0f;  // Initial angular divergence (1D RMS) [rad]
    int Nx = 100;
    int Nz = 200;
    float dx = 1.0f;
    float dz = 1.0f;
    int max_steps = 100;
    float sigma_x0 = 0.0f;  // Initial Gaussian beam width (sigma at z=0) [mm]
    float W_total = 1.0f;
    unsigned random_seed = 42;
};

// Type alias for clarity
using DeterministicConfig = PencilBeamConfig;

struct SimulationResult {
    int Nx, Nz;
    float dx, dz;
    std::vector<std::vector<double>> edep;
    std::vector<float> x_centers;
    std::vector<float> z_centers;
};

SimulationResult run_pencil_beam(const PencilBeamConfig& config);
int find_bragg_peak_z(const SimulationResult& result);
std::vector<double> get_depth_dose(const SimulationResult& result);
