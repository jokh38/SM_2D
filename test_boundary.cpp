#include <cstdio>
#include <cmath>

// Face definitions
#define FACE_Z_PLUS 0
#define FACE_Z_MINUS 1
#define FACE_X_PLUS 2
#define FACE_X_MINUS 3

inline int device_determine_exit_face(
    float x_old, float z_old,
    float x_new, float z_new,
    float dx, float dz
) {
    if (z_new >= dz * 0.5f) return FACE_Z_PLUS;
    if (z_new < -dz * 0.5f) return FACE_Z_MINUS;
    if (x_new >= dx * 0.5f) return FACE_X_PLUS;
    if (x_new < -dx * 0.5f) return FACE_X_MINUS;
    return -1;
}

int main() {
    float dx = 0.5f;
    float dz = 0.5f;
    float step = 0.5f;
    
    // Test with particle at different starting positions
    printf("Testing boundary detection:\n");
    printf("dx = %.3f, dz = %.3f, step = %.3f\n", dx, dz, step);
    printf("Boundary threshold = %.3f\n\n", dz * 0.5f);
    
    // Test 1: particle at cell center
    float z_cell = 0.0f;
    float z_new = z_cell + 1.0f * step;
    printf("Test 1: z_cell = %.3f\n", z_cell);
    printf("  z_new = %.3f\n", z_new);
    printf("  Exit face = %d (expected: 0 for FACE_Z_PLUS)\n\n", device_determine_exit_face(0, z_cell, 0, z_new, dx, dz));
    
    // Test 2: particle at bottom bin
    z_cell = -0.1875f;  // bin 0
    z_new = z_cell + step;
    printf("Test 2: z_cell = %.4f (bin 0)\n", z_cell);
    printf("  z_new = %.4f\n", z_new);
    printf("  Exit face = %d (expected: 0 for FACE_Z_PLUS)\n\n", device_determine_exit_face(0, z_cell, 0, z_new, dx, dz));
    
    // Test 3: particle at top bin
    z_cell = 0.1875f;  // bin 3
    z_new = z_cell + step;
    printf("Test 3: z_cell = %.4f (bin 3)\n", z_cell);
    printf("  z_new = %.4f\n", z_new);
    printf("  Exit face = %d (expected: 0 for FACE_Z_PLUS)\n", device_determine_exit_face(0, z_cell, 0, z_new, dx, dz));
    
    return 0;
}
