#pragma once
#include <string>
#include <vector>

struct NistDataRow {
    float energy_MeV;           // Kinetic energy [MeV]
    float stopping_power;       // dE/dx [MeV cm²/g]
    float csda_range_g_cm2;     // CSDA range [g/cm²]
};

// Load NIST PSTAR data file
std::vector<NistDataRow> LoadNistData(const std::string& path);
bool FileExists(const std::string& path);
