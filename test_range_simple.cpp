#include <cstdio>
#include <cmath>
#include "src/core/grids.cpp"
#include "src/lut/r_lut.cpp"
#include "src/lut/nist_loader.cpp"

// Simple energy deposition function
float compute_energy_deposition(RLUT& lut, float E, float step_length) {
    float E_new = lut.lookup_E_inverse(lut.lookup_R(E) - step_length);
    return E - E_new;
}

int main() {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    
    // Test 150 MeV
    float E_test = 150.0f;
    float R = lut.lookup_R(E_test);
    
    printf("Range lookup test:\n");
    printf("  E = %.1f MeV\n", E_test);
    printf("  R = %.2f mm (expected ~158 mm)\n", R);
    
    // Test energy after step
    float step = 0.5f;  // mm
    float dE = compute_energy_deposition(lut, E_test, step);
    float E_after = E_test - dE;
    
    printf("\nEnergy loss test:\n");
    printf("  E_initial = %.1f MeV\n", E_test);
    printf("  step = %.3f mm\n", step);
    printf("  dE = %.5f MeV\n", dE);
    printf("  E_after = %.3f MeV\n", E_after);
    
    // Multiple steps
    printf("\nMultiple step simulation:\n");
    float E = E_test;
    float total_distance = 0;
    for (int i = 1; i <= 350; i++) {
        float dE = compute_energy_deposition(lut, E, step);
        E -= dE;
        total_distance += step;
        
        if (i % 50 == 0 || i <= 10) {
            printf("  Step %3d: E = %7.3f MeV, dE = %7.5f MeV, dist = %6.1f mm\n", 
                   i, E, dE, total_distance);
        }
        if (E <= 0.1f) {
            printf("  Step %3d: E = %7.3f MeV (STOPPED)\n", i, E);
            break;
        }
    }
    printf("\nTotal distance to stop: %.1f mm (expected ~158 mm)\n", total_distance);
    
    return 0;
}
