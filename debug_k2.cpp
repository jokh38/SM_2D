#include <cstdio>

// Simulate the K2 transport with centered coordinates
int main() {
    float dz = 0.5f;
    float step = 0.5f;
    
    // Sub-bin offsets (centered coordinates)
    float z_offsets[] = {-0.1875f, -0.0625f, 0.0625f, 0.1875f};
    
    printf("Simulating K2 boundary detection (CENTERED coordinates):\n");
    printf("dz = %.3f, step = %.3f\n", dz, step);
    printf("Boundary threshold (dz * 0.5) = %.3f\n\n", dz * 0.5f);
    
    for (int i = 0; i < 4; i++) {
        float z_cell = z_offsets[i];
        float z_new = z_cell + 1.0f * step;  // mu = 1.0
        
        bool boundary = (z_new >= dz * 0.5f);
        
        printf("Bin %d: z_cell = %.4f, z_new = %.4f, boundary = %s\n", 
               i, z_cell, z_new, boundary ? "YES" : "NO");
    }
    
    return 0;
}
