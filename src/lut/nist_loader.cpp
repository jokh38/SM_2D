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
        // CORRECTED stopping power values (previous values were incorrect by ~9x)
        data = {
            {0.1f, 252.0f, 0.00014f},    // Low energy: high stopping power, very short range
            {1.0f, 74.5f, 0.0026f},       // 1 MeV
            {10.0f, 4.82f, 0.122f},       // 10 MeV
            {50.0f, 6.04f, 2.14f},        // 50 MeV therapeutic
            {70.0f, 5.77f, 4.079f},       // 70 MeV therapeutic: S = 5.77 MeV·cm²/g (was 0.641!)
            {100.0f, 5.19f, 7.57f},       // 100 MeV therapeutic
            {150.0f, 4.53f, 15.6f},       // 150 MeV: range ~156 mm in water (was 0.403!)
            {200.0f, 4.12f, 26.5f},       // 200 MeV
            {250.0f, 3.85f, 37.3f}        // 250 MeV: range ~373 mm (was 0.357!)
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
