#include "validation/pencil_beam.hpp"
#include "source/pencil_source.hpp"
#include "core/grids.hpp"
#include "core/psi_storage.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

int main() {
    // Configure 70 MeV pencil beam simulation
    PencilBeamConfig config;
    config.E0 = 70.0f;      // 70 MeV protons
    config.Nx = 60;         // Transverse bins
    config.Nz = 100;        // Depth bins (100mm for 70 MeV range ~41mm)
    config.dx = 1.0f;       // 1 mm transverse spacing
    config.dz = 1.0f;       // 1 mm depth spacing (100 mm total depth)
    config.W_total = 1.0f;  // 1 Gy normalized dose
    config.x0 = 30.0f;      // Center of transverse grid
    config.z0 = 0.0f;       // Start at surface
    config.random_seed = 42;

    std::cout << "Running 70 MeV proton pencil beam simulation..." << std::endl;
    std::cout << "  Grid: " << config.Nx << " x " << config.Nz << std::endl;
    std::cout << "  Spacing: " << config.dx << " mm x " << config.dz << " mm" << std::endl;

    auto result = run_pencil_beam(config);

    // Get depth-dose distribution
    auto depth_dose = get_depth_dose(result);

    // Find Bragg peak
    int peak_z = find_bragg_peak_z(result);
    double peak_dose = depth_dose[peak_z];
    double peak_depth = peak_z * result.dz;

    std::cout << "Simulation complete." << std::endl;
    std::cout << "  Bragg Peak: " << peak_depth << " mm depth, " << peak_dose << " Gy" << std::endl;

    // Save PDD to file
    std::ofstream out("pdd_70MeV.txt");
    out << "# 70 MeV Proton Pencil Beam Depth-Dose Distribution\n";
    out << "# Depth(mm)    Dose(Gy)    Normalized\n";
    out << "# Bragg Peak at: " << peak_depth << " mm, " << peak_dose << " Gy\n";

    for (size_t i = 0; i < depth_dose.size(); ++i) {
        double depth = i * result.dz;
        double dose = depth_dose[i];
        double norm_dose = (peak_dose > 0) ? dose / peak_dose : 0.0;
        out << std::fixed << std::setprecision(4)
            << depth << "\t" << dose << "\t" << norm_dose << "\n";
    }
    out.close();

    std::cout << "PDD saved to: pdd_70MeV.txt" << std::endl;

    // Also save full 2D dose distribution
    std::ofstream out2d("dose_70MeV_2D.txt");
    out2d << "# 70 MeV Proton Pencil Beam 2D Dose Distribution\n";
    out2d << "# x(mm) z(mm) dose(Gy)\n";
    for (int iz = 0; iz < result.Nz; ++iz) {
        for (int ix = 0; ix < result.Nx; ++ix) {
            double x = ix * result.dx;
            double z = iz * result.dz;
            double dose = result.edep[iz][ix];
            out2d << x << "\t" << z << "\t" << dose << "\n";
        }
        out2d << "\n";  // Blank line for gnuplot pm3d
    }
    out2d.close();

    std::cout << "2D dose distribution saved to: dose_120MeV_2D.txt" << std::endl;

    return 0;
}
