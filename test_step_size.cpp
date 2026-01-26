#include <cstdio>
#include <cmath>

// Calculate step to boundary
float get_step_to_boundary(float z_cell, float half_dz, float mu) {
    float step_to_z_plus = (mu > 0) ? (half_dz - z_cell) / mu : 1e30f;
    float step_to_z_minus = (mu < 0) ? (z_cell - (-half_dz)) / (-mu) : 1e30f;
    return fminf(step_to_z_plus, step_to_z_minus);
}

int main() {
    float half_dz = 0.25f;  // centered coordinates
    float mu = 1.0f;        // normal incidence
    
    // For each sub-bin
    float z_offsets[] = {-0.1875f, -0.0625f, 0.0625f, 0.1875f};
    
    printf("Step size calculation for centered coordinates [-0.25, +0.25]:\n");
    printf("mu = 1.0 (normal incidence)\n\n");
    
    float total_steps = 0;
    for (int i = 0; i < 4; i++) {
        float z_cell = z_offsets[i];
        float step = get_step_to_boundary(z_cell, half_dz, mu);
        total_steps += step;
        printf("Bin %d: z_cell = %+.4f, step_to_boundary = %.4f mm\n", i, z_cell, step);
    }
    printf("\nAverage step per bin: %.4f mm\n", total_steps / 4);
    
    return 0;
}
