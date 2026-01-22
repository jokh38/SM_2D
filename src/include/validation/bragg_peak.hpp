#pragma once
#include "validation/pencil_beam.hpp"

float find_bragg_peak_position_mm(const SimulationResult& result);
float compute_bragg_peak_fwhm(const SimulationResult& result);
float find_R80(const SimulationResult& result);
float find_R20(const SimulationResult& result);
float compute_distal_falloff(const SimulationResult& result);
