#include <cstdio>
#include <cmath>

// Simulate the energy lookup from range table
// Approximating the NIST data for water

float lookup_energy_from_range(float R_mm) {
    // Approximate NIST PSTAR data for protons in water
    // E(MeV) vs R(mm)
    // Key points: (0.001, 0.000063), (0.1, 0.0016), (1, 0.025), (10, 1.23), (100, 77.2), (150, 157.7)
    
    if (R_mm <= 0) return 0;
    if (R_mm < 0.01) return 0.01;   // ~10 keV
    if (R_mm < 0.1) return 0.1 * (R_mm / 0.01);  // rough approximation
    if (R_mm < 1.0) return 1.0 * (R_mm / 0.025);  // rough
    if (R_mm < 10) return 10 + (R_mm - 1.23) * (90 - 10) / (77.2 - 1.23);  // rough
    if (R_mm < 80) return 100 + (R_mm - 77.2) * (150 - 100) / (157.7 - 77.2);
    if (R_mm >= 80 && R_mm < 158) {
        // Linear approximation in this range
        return 100 + (R_mm - 77.2) * 50 / 80.5;
    }
    
    return 200;  // above 150 MeV
}

int main() {
    float R_initial = 157.7f;  // mm for 150 MeV
    float step = 0.25f;        // mm per step
    
    printf("Energy loss simulation:\n");
    printf("Initial: E = 150 MeV, R = %.1f mm\n", R_initial);
    printf("Step size: %.3f mm\n\n", step);
    
    float R = R_initial;
    for (int i = 1; i <= 20; i++) {
        R -= step;
        float E = lookup_energy_from_range(R);
        printf("Step %2d: R = %6.2f mm, E = %6.2f MeV\n", i, fmaxf(0, R), E);
        if (R <= 0) break;
    }
    
    return 0;
}
