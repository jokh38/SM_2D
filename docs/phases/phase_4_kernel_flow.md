# Kernel Pipeline Data Flow

This document describes the data flow between the 6 CUDA kernels (K1-K6).

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PSI_C (Input Buffer)                        │
│                    [Nx × Nz cells × 32 slots × 32 bins]            │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  K1: ActiveMask                                                    │
│  - Scans all cells                                                 │
│  - Identifies cells with E > E_trigger and weight > threshold      │
│  - Output: ActiveMask[Nx × Nz] (uint8_t)                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  (Optional) K2: CompactActive                                      │
│  - Generates compact list of active cell indices                   │
│  - Output: ActiveList[n_active] (uint32_t)                         │
│  Status: DEFERRED for MVP                                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  K3: FineTransport (CORE KERNEL)                                   │
│  For each active cell:                                             │
│    1. Iterate over slots/bins with weight > epsilon                │
│    2. Decode (theta, E) from block ID + local index                │
│    3. Transport components:                                        │
│       - Compute step size (ds)                                      │
│       - Apply MCS scattering (Highland formula)                     │
│       - Check energy cutoff                                         │
│       - Apply nuclear attenuation                                   │
│       - Detect boundary crossing                                    │
│    4. Emit to outflow buckets if crossing boundary                 │
│    5. Otherwise, update and keep in cell                           │
│                                                                     │
│  Outputs:                                                           │
│    - EdepC[Nx × Nz] (double) - Energy deposition per cell          │
│    - AbsorbedWeight_cutoff[Nx × Nz] (float)                        │
│    - AbsorbedWeight_nuclear[Nx × Nz] (float)                       │
│    - AbsorbedEnergy_nuclear[Nx × Nz] (double)                      │
│    - OutflowBuckets[Nx × Nz × 4] - Boundary emissions              │
│    - BoundaryLoss_weight, BoundaryLoss_energy                      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  K4: BucketTransfer                                                │
│  - For each cell and 4 faces:                                      │
│    - Read bucket contents                                          │
│    - Find neighbor cell (or boundary)                              │
│    - Transfer weights to neighbor's PsiC output buffer             │
│  - Accumulate weights into existing blocks                         │
│                                                                     │
│  Output: Updates to PSI_C (Output Buffer)                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  K5: ConservationAudit                                             │
│  - For each cell:                                                  │
│    W_in = sum(input buffer weights)                                │
│    W_out = sum(output buffer weights)                              │
│    W_cutoff = AbsorbedWeight_cutoff[cell]                          │
│    W_nuclear = AbsorbedWeight_nuclear[cell]                        │
│                                                                     │
│    error = |W_in - W_out - W_cutoff - W_nuclear|                   │
│    pass = (error / W_in < 1e-6)                                    │
│                                                                     │
│  Output: AuditReport[Nx × Nz]                                      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  K6: SwapBuffers                                                    │
│  - Exchange input/output buffer pointers                           │
│  - Prepares for next iteration                                     │
│                                                                     │
│  Output: Swapped PsiC* pointers                                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Next Iteration  │
                └──────────────────┘
```

## Memory Layout

### PsiC Buffer (Input/Output)
```
PsiC = {
    block_id[Nx × Nz][32]      // uint32_t
    value[Nx × Nz][32][32]     // float (LOCAL_BINS)
}
```

### OutflowBuckets
```
OutflowBuckets[Nx × Nz × 4] = {
    block_id[Kb_out]           // uint32_t
    local_count[Kb_out]        // uint16_t
    value[Kb_out][LOCAL_BINS]  // float
}
```

### Per-Cell Outputs
```
EdepC[Nx × Nz]                    // double
AbsorbedWeight_cutoff[Nx × Nz]    // float
AbsorbedWeight_nuclear[Nx × Nz]   // float
AbsorbedEnergy_nuclear[Nx × Nz]   // double
```

## Kernel Launch Parameters

### K1: ActiveMask
```cuda
dim3 threads(256);
dim3 blocks((Nx * Nz + 255) / 256);
K1_ActiveMask<<<blocks, threads>>>(...);
```

### K3: FineTransport
```cuda
dim3 threads(128);  // Limited by shared memory usage
dim3 blocks((n_active + 127) / 128);
K3_FineTransport<<<blocks, threads>>>(...);
```

### K4: BucketTransfer
```cuda
dim3 threads(256);
dim3 blocks((Nx * Nz + 255) / 256);
K4_BucketTransfer<<<blocks, threads>>>(...);
```

### K5: ConservationAudit
```cuda
dim3 threads(256);
dim3 blocks((Nx * Nz + 255) / 256);
K5_WeightAudit<<<blocks, threads>>>(...);
```

### K6: SwapBuffers
```cpp
// Host-side operation, no kernel launch
K6_SwapBuffers(psi_in, psi_out);
```

## Performance Considerations

1. **Memory Coalescing**: All kernels use thread IDs that map to contiguous cell indices
2. **Shared Memory**: K3 uses shared memory for reduction within thread blocks
3. **Atomic Operations**: K4 uses atomicAdd for weight accumulation in output buffer
4. **Divergence**: K3 has conditional branching on boundary crossing (minimized by sorting)

## Conservation Laws

### Weight Conservation
```
W_total = W_input + W_source
W_total = W_output + W_cutoff + W_nuclear + W_boundary
```

### Energy Conservation
```
E_total = E_input + E_source
E_total = E_output + E_dep + E_nuclear + E_boundary_loss
```

## Dependencies Between Kernels

| Kernel | Reads | Writes | Dependencies |
|--------|-------|--------|--------------|
| K1 | PsiC_in | ActiveMask | None |
| K2 | ActiveMask | ActiveList | K1 (deferred) |
| K3 | PsiC_in, ActiveList | EdepC, buckets, absorbed | K1 |
| K4 | buckets | PsiC_out | K3 |
| K5 | PsiC_in, PsiC_out, absorbed | AuditReport | K3, K4 |
| K6 | psi_in, psi_out | (swaps pointers) | K5 |

## Testing Strategy

1. **Unit Tests**: Test each kernel independently with CPU stubs
2. **Integration Tests**: Run full pipeline with known inputs
3. **Conservation Tests**: Verify K5 reports < 1e-6 error
4. **Determinism Tests**: Check identical results across runs
5. **Performance Tests**: Profile kernel execution times

## Known Limitations (Current Implementation)

1. **K3 Stub**: Only basic CPU test stub implemented
2. **No Device LUT**: K3 cannot access R(E) tables on device
3. **No K2**: CompactActive kernel deferred
4. **No Angular Quadrature**: 7-point splitting not integrated
5. **No Boundary Detection**: Geometry calculations incomplete

## Path to Full Implementation

1. Port Phase 1 R_LUT to device memory
2. Implement device-side physics functions
3. Complete K3 boundary crossing logic
4. Add angular quadrature integration
5. Implement variance-based MCS splitting
6. Full integration testing
