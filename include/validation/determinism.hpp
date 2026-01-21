#pragma once
#include "validation/pencil_beam.hpp"
#include <cstdint>

uint32_t compute_checksum(const SimulationResult& result);
bool verify_determinism(const PencilBeamConfig& config);
