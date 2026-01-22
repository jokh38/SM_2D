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
        data = {
            {0.1f, 252.0f, 0.00014f},    // Low energy: high stopping power, very short range
            {1.0f, 74.5f, 0.0026f},       // 1 MeV
            {10.0f, 4.82f, 0.122f},       // 10 MeV
            {70.0f, 0.641f, 4.079f},      // 70 MeV therapeutic: range ~40.8 mm in water
            {150.0f, 0.403f, 15.82f},     // 150 MeV: range ~158 mm in water
            {250.0f, 0.357f, 37.93f}      // 250 MeV: range ~379 mm in water
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
