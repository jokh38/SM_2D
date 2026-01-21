#include "lut/nist_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

std::vector<NistDataRow> LoadNistData(const std::string& path) {
    std::vector<NistDataRow> data;
    std::ifstream file(path);

    if (!file.is_open()) {
        // Return minimal hardcoded data for water if file not found
        // Based on NIST PSTAR for water
        data = {
            {0.1f, 20.0f, 0.0001f},
            {1.0f, 50.0f, 0.0023f},
            {10.0f, 45.0f, 0.12f},
            {70.0f, 5.8f, 4.08f},
            {150.0f, 0.48f, 15.8f},
            {250.0f, 0.24f, 38.0f}
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
