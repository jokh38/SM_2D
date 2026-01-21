#include "validation/determinism.hpp"
#include <cstdint>
#include <cstring>

uint32_t compute_checksum(const SimulationResult& result) {
    // Simple checksum for energy deposition grid
    uint32_t checksum = 0;

    for (int j = 0; j < result.Nz; ++j) {
        for (int i = 0; i < result.Nx; ++i) {
            // Convert double to uint32_t representation for checksum
            double val = result.edep[j][i];
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(double));

            // Fold 64-bit to 32-bit
            checksum += static_cast<uint32_t>(bits ^ (bits >> 32));
        }
    }

    return checksum;
}

bool verify_determinism(const PencilBeamConfig& config) {
    // Run simulation twice with identical config
    SimulationResult result1 = run_pencil_beam(config);
    SimulationResult result2 = run_pencil_beam(config);

    // Compare checksums
    uint32_t checksum1 = compute_checksum(result1);
    uint32_t checksum2 = compute_checksum(result2);

    return checksum1 == checksum2;
}
