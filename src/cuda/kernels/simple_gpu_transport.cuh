#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "device/device_lut.cuh"

/**
 * @brief Simple GPU particle transport for dose calculation
 *
 * This module provides a simplified GPU-based particle transport
 * that uses proper energy straggling (Vavilov model) and MCS (Highland).
 * It's designed to replace the analytical deterministic_beam.cpp
 * with actual GPU kernel execution.
 */

// Input particle state
struct ParticleInput {
    float x;       // Position X [mm]
    float z;       // Position Z [mm]
    float E;       // Energy [MeV]
    float theta;   // Angle [rad]
    float w;       // Weight
};

/**
 * @brief Run simple GPU particle transport
 *
 * @param x0, z0      Initial position [mm]
 * @param theta0      Initial angle [rad]
 * @param E0          Initial energy [MeV]
 * @param W_total     Total weight
 * @param n_particles Number of particles to simulate
 * @param Nx, Nz      Grid dimensions
 * @param dx, dz      Grid spacing [mm]
 * @param x_min, z_min Grid origin
 * @param dlut        Device lookup table for energy loss
 * @param edep        Output: 2D energy deposition grid [Gy]
 */
void run_simple_gpu_transport(
    float x0, float z0, float theta0, float E0, float W_total,
    int n_particles,
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    const DeviceRLUT& dlut,
    std::vector<std::vector<double>>& edep
);
