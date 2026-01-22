#include "kernels/k6_swap.cuh"
#include "core/psi_storage.hpp"

void K6_SwapBuffers(PsiC*& in, PsiC*& out) {
    PsiC* temp = in;
    in = out;
    out = temp;
}
