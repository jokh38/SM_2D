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
        //
        // Key reference values (from official NIST PSTAR):
        //   0.1 MeV:  S = 814.5 MeV·cm²/g,  R = 0.0001607 g/cm² → 0.0016 mm
        //   1 MeV:    S = 260.6 MeV·cm²/g,  R = 0.002458 g/cm²   → 0.025 mm
        //   10 MeV:   S = 45.64 MeV·cm²/g,  R = 0.1230 g/cm²     → 1.23 mm
        //   70 MeV:   S = 9.555 MeV·cm²/g,  R = 4.080 g/cm²      → 40.8 mm
        //   100 MeV:  S = 7.286 MeV·cm²/g,  R = 7.718 g/cm²      → 77.2 mm
        //   150 MeV:  S = 5.443 MeV·cm²/g,  R = 15.77 g/cm²      → 157.7 mm
        //   200 MeV:  S = 4.491 MeV·cm²/g,  R = 25.96 g/cm²      → 259.6 mm
        //
        // Unit conversion for water (rho=1.0 g/cm³):
        //   Range[mm] = Range[g/cm²] * 10 / rho
        data = {
            {0.001f, 133.7f, 0.000006319f},    // 1 keV
            {0.01f, 422.9f, 0.00003599f},     // 10 keV
            {0.1f, 814.5f, 0.0001607f},       // 0.1 MeV: S ≈ 815 MeV·cm²/g, R ≈ 0.0016 mm
            {0.5f, 412.8f, 0.0008869f},       // 0.5 MeV
            {1.0f, 260.6f, 0.002458f},        // 1 MeV: R = 0.025 mm
            {2.0f, 158.6f, 0.007555f},        // 2 MeV
            {5.0f, 79.11f, 0.03623f},         // 5 MeV
            {10.0f, 45.64f, 0.1230f},         // 10 MeV: R = 1.23 mm
            {20.0f, 26.05f, 0.4260f},         // 20 MeV: R = 4.26 mm
            {30.0f, 18.76f, 0.8853f},         // 30 MeV: R = 8.85 mm
            {50.0f, 12.45f, 2.227f},          // 50 MeV: R = 22.3 mm
            {70.0f, 9.555f, 4.080f},          // 70 MeV: R = 40.8 mm (NIST PSTAR)
            {80.0f, 8.622f, 5.184f},          // 80 MeV: R = 51.8 mm
            {90.0f, 7.884f, 6.398f},          // 90 MeV: R = 64.0 mm
            {100.0f, 7.286f, 7.718f},         // 100 MeV: R = 77.2 mm (NIST PSTAR)
            {150.0f, 5.443f, 15.77f},         // 150 MeV: R = 157.7 mm (NIST PSTAR)
            {200.0f, 4.491f, 25.96f},         // 200 MeV: R = 259.6 mm (NIST PSTAR)
            {250.0f, 3.910f, 37.94f},         // 250 MeV: R = 379.4 mm
            {300.0f, 3.519f, 51.45f}          // 300 MeV: R = 514.5 mm
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
