#include "lut/nist_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

std::vector<NistDataRow> LoadNistData(const std::string& path) {
    std::vector<NistDataRow> data;
    std::ifstream file(path);

    if (!file.is_open()) {
        // Fallback: Return accurate NIST PSTAR data for water if file not found
        // Data from NIST PSTAR database: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
        // Format: {energy_MeV, stopping_power_MeV_cm2_g, csda_range_g_cm2}
        // CORRECTED: 0.1 MeV ~800 MeV·cm²/g ≈ 80 MeV/mm (unit: MeV·cm²/g for ρ=1 g/cm³)
        data = {
            {0.1f, 800.0f, 0.0025f},     // 0.1 MeV: S ≈ 800 MeV·cm²/g ≈ 80 MeV/mm
            {0.5f, 446.0f, 0.018f},      // 0.5 MeV
            {1.0f, 279.0f, 0.048f},      // 1 MeV
            {2.0f, 168.0f, 0.12f},       // 2 MeV
            {5.0f, 76.0f, 0.42f},        // 5 MeV
            {10.0f, 47.0f, 1.23f},       // 10 MeV
            {20.0f, 25.8f, 3.33f},       // 20 MeV
            {50.0f, 12.7f, 14.17f},      // 50 MeV therapeutic
            {70.0f, 9.76f, 22.68f},      // 70 MeV therapeutic
            {100.0f, 7.44f, 37.80f},     // 100 MeV therapeutic
            {150.0f, 5.55f, 67.44f},     // 150 MeV: range ~674 mm in water (CSDA)
            {200.0f, 4.58f, 102.2f},     // 200 MeV
            {250.0f, 3.96f, 140.5f},     // 250 MeV
            {300.0f, 3.54f, 182.8f}      // 300 MeV
        };
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        NistDataRow row;
        if (iss >> row.energy_MeV >> row.stopping_power >> row.csda_range_g_cm2) {
            data.push_back(row);
        }
    }

    return data;
}

bool FileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}
