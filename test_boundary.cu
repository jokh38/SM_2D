#include <cstdio>
#include <cmath>

// Face definitions
#define FACE_Z_PLUS 0
#define FACE_Z_MINUS 1
#define FACE_X_PLUS 2
#define FACE_X_MINUS 3

__device__ inline int device_determine_exit_face(
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
    
    // Test with particle at cell center
    float x_cell = 0.0f;
    float z_cell = 0.0f;
    float mu = 1.0f;  // cos(0) = 1
    
    float z_new = z_cell + mu * step;
    
    printf("dx = %.3f, dz = %.3f, step = %.3f\n", dx, dz, step);
    printf("z_cell = %.3f, z_new = %.3f\n", z_cell, z_new);
    printf("Boundary threshold = %.3f\n", dz * 0.5f);
    printf("z_new >= threshold? %s\n", z_new >= dz * 0.5f ? "YES" : "NO");
    
    int face = device_determine_exit_face(x_cell, z_cell, x_cell, z_new, dx, dz);
    printf("Exit face = %d\n", face);
    
    return 0;
}
