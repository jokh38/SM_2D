#pragma once
#include "validation/pencil_beam.hpp"
#include <vector>

float get_lateral_sigma_at_z(const SimulationResult& result, float z_mm);
std::vector<double> get_lateral_profile(const SimulationResult& result, float z_mm);
float compute_fermi_eyges_sigma(float E0_MeV, float z_mm);
float fit_gaussian_sigma(const std::vector<double>& profile, float dx);
