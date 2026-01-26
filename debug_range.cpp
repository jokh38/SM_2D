#include <cstdio>
#include <cmath>

// Simulate the range-based energy update
int main() {
    // For 150 MeV protons in water
    float R_initial = 157.7f;  // mm (from NIST data)
    float cell_size = 0.5f;    // mm
    float step_size = 0.5f;    // mm per iteration
    
    printf("Proton transport simulation (CSDA approximation):\n");
    printf("Initial energy: 150 MeV, Initial range: %.1f mm\n", R_initial);
    printf("Step size: %.3f mm per iteration\n\n", step_size);
    
    float R = R_initial;
    int iter = 0;
    while (R > 0 && iter < 400) {
        float R_before = R;
        R -= step_size;
        iter++;
        
        if (iter % 50 == 0 || R <= 0) {
            printf("Iteration %3d: Range = %6.2f mm (traveled %6.2f mm)\n", 
                   iter, fmaxf(0, R), iter * step_size);
        }
    }
    
    printf("\nResult: After %d iterations, range exhausted\n", iter);
    printf("Total distance traveled: %.1f mm\n", iter * step_size);
    
    return 0;
}
