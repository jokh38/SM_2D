#set text(font: "Times New Roman", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

// Define custom box elements
#let tip-box(body) = block(
  fill: rgb("#e6fff2"),
  inset: 10pt,
  radius: 5pt,
  body
)

#let warning-box(body) = block(
  fill: rgb("#fff0cc"),
  inset: 10pt,
  radius: 5pt,
  body
)

#show math.equation: set text(weight: "regular")

= CUDA Kernel Pipeline Documentation

== GPU Programming Fundamentals

=== What is a GPU?

*KEY CONCEPT*
===

A GPU (Graphics Processing Unit) is a specialized processor designed for parallel computation. Unlike a CPU which has a few powerful cores optimized for sequential tasks, a GPU has thousands of smaller cores optimized for doing many simple operations simultaneously.

*In Plain English*:
Imagine you need to paint 10,000 fence posts.
- CPU approach: One master painter paints each post one at a time (fast per post, but serial)
- GPU approach: 10,000 apprentice painters each paint one post simultaneously (slower per painter, but massively parallel)

For particle transport simulations, we need to process millions of particles. A GPU can process thousands of particles at the same time.

=== What is a CUDA Kernel?

*KEY CONCEPT*
===

A CUDA kernel is a function that runs on the GPU. When you launch a kernel, it executes on multiple GPU threads in parallel. The same code runs on different data elements - this is called *Single Instruction, Multiple Data* (SIMD).

*In Plain English*:
Think of a kernel as a recipe that multiple chefs follow simultaneously. Each chef (thread) has the same recipe (kernel code) but works on different ingredients (data).

```cpp
// CPU function - runs once
void cpu_function(int* data, int size) {
    for (int i = 0; i \< size; i++) {
        data[i] = data[i] * 2;
    }
}

// GPU kernel - runs on thousands of threads simultaneously
__global__ void gpu_kernel(int* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx \< size) {
        data[idx] = data[idx] * 2;
    }
}
```

=== What are Threads and Blocks?

*KEY CONCEPT*
===

CUDA organizes threads into a two-level hierarchy:

= Threads
The smallest unit of execution. Each thread executes the kernel code independently.

= Blocks
Groups of threads (typically 32-1024 threads) that can cooperate through shared memory and synchronization.

= Grid
A collection of blocks that execute the same kernel.

#figure(
  table(
    columns: (auto, 2fr, 2fr),
    inset: 8pt,
    align: (left, center, center),
    table.header([*Component*], [*Description*], [*Example*]),
    [Grid], [All blocks executing kernel], [All cells in simulation],
    [Block], [Group of threads (32-1024)], [Threads 0-255 process Cell 0],
    [Thread], [Single execution unit], [Thread 0 processes Cell 0, Slot 0],
  ),
  caption: [CUDA Thread Hierarchy],
)

Each thread represents one worker that processes particles in one cell. All threads in a block work together and can share fast memory.

=== What is Shared Memory vs Global Memory?

*KEY CONCEPT*
===

= Global Memory
Large but slow (400-800 clock cycles latency). All threads can access any location. Like main RAM in a computer.

= Shared Memory
Small but fast (~1 clock cycle latency). Only threads within the same block can access it. Like CPU L1 cache.

*In Plain English*:
- Global Memory: A library where books are stored in the basement (takes time to fetch)
- Shared Memory: A small reading table where your discussion group keeps frequently-used books (instant access)

#figure(
  table(
    columns: (auto, 3fr, 2fr),
    inset: 8pt,
    align: (left, left, right),
    table.header([*Memory Type*], [*Characteristics*], [*Latency*]),
    [Shared Memory], [Only threads in same block can access. Used for temp variables, partial sums.], [~1 cycle],
    [L2 Cache], [Shared by all blocks. Intermediate caching layer.], [~100 cycles],
    [Global Memory], [All threads can access any location. Stores all particle data, phase-space arrays (GB scale).], [400+ cycles],
  ),
  caption: [GPU Memory Hierarchy],
)

== Overview

SM_2D implements a 6-stage CUDA kernel pipeline for deterministic proton transport. The pipeline processes particles through hierarchical refinement, with coarse transport for high-energy particles and fine transport for the critical Bragg peak region.

== Pipeline Architecture

=== Kernel Sequence

The simulation loop iterates through six kernels per step:

#figure(
  table(
    columns: (auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*Kernel*], [*Purpose*]),
    [K1: ActiveMask], [Detect cells needing fine transport],
    [K2: CoarseTransport], [High-energy transport (E > 10 MeV)],
    [K3: FineTransport], [Low-energy transport (E ≤ 10 MeV)],
    [K4: BucketTransfer], [Inter-cell particle transfer],
    [K5: WeightAudit], [Validate weight conservation],
    [K6: SwapBuffers], [Exchange in/out pointers],
  ),
  caption: [CUDA Kernel Pipeline],
)

=== Visual Pipeline Flow

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    align: (left, left),
    table.header([*Stage*], [*Description*]),

    [Input], [Particles in cells (t = 0)],
    [↓], [],

    [K1: Active Mask], [
      *Task*: Scan all cells and mark which ones need fine transport
      \
      *For each cell*:
      1. Check particle energies in cell
      2. If any E < 10 MeV → mark as "active"
      3. Create list of active cells for K3
      \
      *Output*: ActiveMask[0..N-1] (0 or 1 for each cell), ActiveList (compressed list)
    ],

    [↓], [],

    [K2: Coarse Transport], [
      *Task*: Fast approximate transport for INACTIVE cells (E > 10 MeV)
      \
      *For each INACTIVE cell*:
      1. Use mean energy loss only (no straggling)
      2. Track variance, don't sample MCS
      3. Larger step sizes
      4. Fast (3-5x speedup)
      \
      *Output*: Transported particles, energy deposition, outflow buckets
    ],

    [K3: Fine Transport], [
      *Task*: Accurate Monte Carlo for ACTIVE cells (E ≤ 10 MeV)
      \
      *For each ACTIVE cell*:
      1. Sample energy straggling (Vavilov)
      2. Sample MCS scattering (random angles)
      3. Smaller step sizes
      4. Accurate (1 percent error)
      \
      *Output*: Transported particles, energy deposition, outflow buckets
    ],

    [↓], [],

    [K4: Bucket Transfer], [
      *Task*: Move particles that left their cell
      \
      *For each cell*:
      1. Check 4 neighbors (±x, ±z)
      2. For each neighbor's outflow bucket: read particles heading to this cell, add them to this cell's phase space
      \
      *Output*: All particles now in correct cells
    ],

    [↓], [],

    [K5: Conservation Audit], [
      *Task*: Verify no particles or energy were lost
      \
      *For each cell*:
      1. Count total weight IN (from start)
      2. Count total weight OUT (after transport)
      3. Count absorbed weight (cutoff + nuclear)
      4. Verify: IN = OUT + ABSORBED
      5. Report any violations
      \
      *Output*: AuditReport for each cell (pass/fail + error)
    ],

    [↓], [],

    [K6: Swap Buffers], [
      *Task*: Prepare for next time step
      \
      *Action*: Exchange input and output pointers (CPU-side)
      - Previous output becomes next input
      - Previous input becomes next output (overwritten)
      \
      *Why*: Avoid copying 2.2 GB of data per iteration
      \
      *Output*: Ready for next iteration
    ],

    [↓], [],

    [Output], [Particles at t = Δt (ready for next iteration)],
  ),
  caption: [CUDA Pipeline Flow (K1-K6)],
)

== K1: ActiveMask Kernel

=== File

`src/cuda/kernels/k1_activemask.cu`

=== What It Does

K1 scans every cell in the simulation grid and answers the question: "Does this cell contain low-energy particles that need accurate simulation?" It creates a mask (a list of 0s and 1s) where 1 means "process this cell carefully" and 0 means "fast processing is OK."

=== Why It Exists

Not all cells need expensive, accurate simulation. High-energy particles (E > 10 MeV) are far from the Bragg peak and don't affect the dose distribution much. Low-energy particles (E ≤ 10 MeV) are in or near the Bragg peak where small errors cause large dose errors. K1 identifies which cells matter.

=== How It Works

#figure(
  table(
    columns: (2fr, 4fr),
    inset: 10pt,
    table.header([*Stage*], [*Process*]),

    [Input], [
      *All cells with particles*:
      - Cell 0: [E=150 MeV] [E=120 MeV] [E=180 MeV] → All high energy
      - Cell 1: [E=8 MeV] [E=12 MeV] [E=200 MeV] → Has low energy
      - Cell 2: [E=250 MeV] [E=300 MeV] → All high energy
      - Cell 3: [E=5 MeV] [E=7 MeV] → All low energy
    ],

    [Processing], [
      *Each thread checks one cell*:
      \
      *Thread 0 (Cell 0)*:
      - Minimum E = 120 MeV
      - 120 > 10, so NO low energy
      - Set ActiveMask[0] = 0
      \
      *Thread 1 (Cell 1)*:
      - Minimum E = 8 MeV
      - 8 ≤ 10, so YES low energy present
      - Set ActiveMask[1] = 1
      \
      *Thread 2 (Cell 2)*:
      - Minimum E = 250 MeV
      - 250 > 10, so NO low energy
      - Set ActiveMask[2] = 0
      \
      *Thread 3 (Cell 3)*:
      - Minimum E = 5 MeV
      - 5 ≤ 10, so YES low energy present
      - Set ActiveMask[3] = 1
    ],

    [Output], [
      *ActiveMask array*: [0, 1, 0, 1, 0, 0, 1, ...]
      \
      *ActiveList*: [1, 3, 6, ...] ← Compressed list of active cells
      \
      (Cells 1, 3, and 6 are active)
    ],
  ),
  caption: [K1 Active Mask Generation],
)

=== Thread Configuration

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Parameter*], [*Value*]),
    [Grid Size], [`(Nx * Nz + 255) / 256`],
    [Block Size], [256 threads],
    [Threads per Cell], [1 thread processes 1 cell],
    [Memory Access], [Coalesced reads from block_ids_in, values_in],
  ),
  caption: [K1 Thread Configuration],
)

=== Memory Access Pattern

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Concept*], [*Description*]),

    [Global Memory Layout], [
      *Row-major organization*:
      \
      Cell 0, Cell 1, Cell 2, Cell 3, Cell 4, Cell 5, Cell 6, Cell 7, ...
      \
      Coalesced access: Adjacent threads read adjacent memory locations
    ],

    [Access Pattern], [
      Thread 0 reads Cell 0
      \
      Thread 1 reads Cell 1
      \
      Thread 2 reads Cell 2
      \
      ...
      \
      GPU can service all 256 threads in one memory transaction → 20-30x faster than scattered access
    ],
  ),
  caption: [K1 Memory Access Pattern (Coalesced)],
)

=== Signature

```cpp
__global__ void K1_ActiveMask(
    // Input phase-space
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,

    // Grid parameters
    const int Nx, const int Nz,

    // Thresholds
    const int b_E_trigger,         // Energy block index threshold (pre-computed)
    const float weight_active_min, // Minimum weight (default: 1e-12)

    // Output
    uint8_t* __restrict__ ActiveMask
);
```

#warning-box[
*Important:* `b_E_trigger` is an *integer block index*, not a float in MeV. It's pre-computed from the energy threshold and represents the coarse energy block index below which fine transport is activated.
]

=== Algorithm

```cpp
__global__ void K1_ActiveMask(...) {
    // STEP 1: Calculate which cell this thread handles
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;  // Boundary check

    // STEP 2: Initialize accumulator variables
    float total_weight = 0.0f;
    bool needs_fine_transport = false;

    // STEP 3: Scan all particle slots in this cell
    for (int slot = 0; slot < 32; ++slot) {
        // Read block ID (energy/angle bin information)
        uint32_t bid = block_ids[cell * 32 + slot];
        if (bid == 0xFFFFFFFF) continue;  // Skip empty slots

        // Decode block ID to get energy bin (direct bit extraction)
        uint32_t b_E = (bid >> 12) & 0xFFF;

        // STEP 4: Check if low energy present
        // Compare b_E directly against trigger (block index comparison)
        if (b_E < static_cast<uint32_t>(b_E_trigger)) {
            needs_fine_transport = true;
        }

        // STEP 5: Accumulate particle weight
        for (int lidx = 0; lidx < 32; ++lidx) {
            total_weight += values[(cell * 32 + slot) * 32 + lidx];
        }
    }

    // STEP 6: Write output
    // Mark active if: (low energy present) AND (sufficient weight)
    ActiveMask[cell] = (needs_fine_transport && total_weight > weight_active_min) ? 1 : 0;
}
```

=== In Plain English

Think of K1 as a quality control inspector. Imagine you have a warehouse with thousands of boxes (cells), each containing items (particles) of different values (energies). The inspector needs to identify which boxes contain valuable items that need special handling.

  K1 quickly checks each box:
   - If any item has value > $1000$ ($E > 10$ "MeV"), for standard handling
   - Items at $1000$ or below: careful handling

This way, expensive "careful handling" (K3 fine transport) is only used where it matters.

== K2: Coarse Transport Kernel

=== File

`src/cuda/kernels/k2_coarsetransport.cu`

=== What It Does

K2 transports high-energy particles quickly using approximations. Instead of simulating every physics detail, it uses average values. It's like taking a highway instead of city streets - faster but less detailed.

=== Why It Exists

High-energy particles (far from the Bragg peak) don't affect the final dose distribution much. A 5% error in a 150 MeV particle's position doesn't matter because it will deposit most of its energy far away. By using approximations, we get 3-5x speedup with acceptable accuracy.

=== How It Works

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Component*], [*Description*]),

    [Input], [
      *Inactive cells* (ActiveMask = 0):
      - Cell 0: Particles at E=150 MeV, E=180 MeV
      - Cell 2: Particles at E=200 MeV, E=250 MeV
    ],

    [Transport Approximations], [
      1. *Energy Loss*: Mean value only
         - Standard: E_new = E_old - dE - straggling
         - Coarse: E_new = E_old - dE_mean
         - Result: ~3% error, 2x faster
      \
      2. *Multiple Coulomb Scattering*: Variance only
         - Standard: Sample random angle θ ~ N(0, sigma^2)
         - Coarse: Accumulate sigma^2, don't sample yet
         - Result: ~5% error in spread, 3x faster
      \
      3. *Step Size*: Larger steps
         - Standard: ds = min(physics_limit, boundary_distance)
         - Coarse: ds = 2-3x larger
         - Result: Fewer steps, faster execution
      \
      4. *Nuclear Reactions*: Same as fine (can't approximate)
         - w *= exp(-sigma_nuclear * ds)
    ],

    [Output], [
      - Approximate particle positions and energies
      - Energy deposition: EdepC array (used for dose calculation)
      - Outflow buckets: Particles that left cells
    ],
  ),
  caption: [K2 Coarse Transport],
)

=== Thread Configuration

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Parameter*], [*Value*]),
    [Grid Size], [`(Nx * Nz + 255) / 256`],
    [Block Size], [256 threads],
    [Processing], [1 thread per cell (skips active cells)],
    [Speedup], [3-5x vs fine transport],
  ),
  caption: [K2 Thread Configuration],
)

=== Key Differences from K3

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Feature*], [*K2 (Coarse)*], [*K3 (Fine)*], [*Difference*]),
    [Energy straggling], [No (mean only)], [Yes (Vavilov)], [~3% accuracy impact],
    [MCS sampling], [No (variance only)], [Yes (random sampling)], [~5% spread impact],
    [Step size], [Larger], [Smaller], [2-3x speedup],
      [Accuracy], [~5%], [1 percent], [Clinical acceptable],
  ),
  caption: [K2 vs K3 Comparison],
)

=== Signature

```cpp
__global__ void K2_CoarseTransport(
    // Input phase-space
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint8_t* __restrict__ ActiveMask,

    // Grid & physics
    const int Nx, const int Nz, const float dx, const float dz,
    const int n_coarse,
    const DeviceRLUT dlut,

    // Grid edges
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,

    // Config
    K2Config config,

    // Output
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,

    // Outflow buckets
    DeviceOutflowBucket* __restrict__ OutflowBuckets
);
```

=== Simplified Physics

```cpp
__device__ void coarse_transport_step(
    float& E, float& theta, float& x, float& z, float& w,
    float ds, const RLUT& lut
) {
    // ENERGY LOSS (mean only, no straggling)
    // Standard: Would sample from Vavilov distribution
    // Coarse:   Just use mean value
    float E_new = lut.lookup_E_inverse(lut.lookup_R(E) - ds);

    // MULTIPLE COULOMB SCATTERING (accumulate variance, don't sample)
    // Standard: theta += sample_gaussian(0, sigma_theta²)
    // Coarse:   theta_variance += sigma_theta²  (save for later)
    float sigma_theta = highland_sigma(E, ds, X0_water);
    theta_variance += sigma_theta * sigma_theta;

    // NUCLEAR ATTENUATION (same as fine - can't approximate)
    float sigma_nuc = Sigma_total(E);
    w *= exp(-sigma_nuc * ds);

    // POSITION UPDATE
    x += ds * sin(theta);
    z += ds * cos(theta);

    E = E_new;
}
```

=== In Plain English

Think of K2 as calculating travel time using average speed vs. actual traffic conditions.

*Fine transport (K3)*: "I'll drive through the city and stop at every red light. This will take 23 minutes and 42 seconds."

*Coarse transport (K2)*: "Average speed in the city is 25 mph, so it will take about 24 minutes."

The coarse calculation is:
- Faster (no need to track every detail)
- Close enough (error is small for high-energy particles)
- Scales better (can process thousands of particles quickly)

== K3: Fine Transport Kernel (MAIN PHYSICS)

=== File

`src/cuda/kernels/k3_finetransport.cu`

=== What It Does

K3 is the heart of the simulation. It performs accurate Monte Carlo transport for low-energy particles in or near the Bragg peak. Every physics effect is simulated with random sampling to ensure accurate dose distribution.

=== Why It Exists

The Bragg peak is where most energy is deposited. Small errors here cause large dose errors. For example, a 1 mm position error at 150 MeV doesn't matter much, but at 10 MeV it can change the dose by 20%. K3 ensures 1 percent error where it matters.

=== How It Works

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Stage*], [*Process*]),

    [Input], [
      *Active cells* (from K1):
      \
      ActiveList = [1, 3, 6, ...]
      \
      - Cell 1: Particles at E=8 MeV, E=12 MeV → Bragg peak region
      - Cell 3: Particles at E=5 MeV, E=7 MeV → End of range
      - Cell 6: Particles at E=9 MeV, E=15 MeV → Bragg peak region
    ],

    [Thread Assignment], [
      Thread 0 → processes Cell 1 (ActiveList[0])
      \
      Thread 1 → processes Cell 3 (ActiveList[1])
      \
      Thread 2 → processes Cell 6 (ActiveList[2])
    ],

    [Per-Particle Transport], [
      *For each particle in active cell*:
      \
      1. *Decode Phase Space*:
         - Get position (x, z) within cell
         - Get energy (E) and angle (theta)
         - Get weight (w) - probability of this path
      \
      2. *Main Physics Loop*:
         a) Calculate step size ds
            - Physics limit: dE/dE constraint
            - Boundary: distance to cell edge
            - ds = min(physics_limit, boundary)
         \
            b) Energy loss with straggling
               - Mean loss: mean_dE = stopping_power times ds
              - Sample straggling: dE ~ Vavilov(kappa, beta)
              - E_new = E - mean_dE - straggle_dE
               - Deposit: Edep += w times (E - E_new)
         \
         c) Multiple Coulomb Scattering
            - Calculate sigma: sigma = Highland_formula(E, ds)
            - Sample angle: dtheta ~ N(0, sigma^2)
            - Update: theta += dtheta
         \
          d) Nuclear reactions
             - Probability: P = 1 - exp(-sigma\_nuclear \* ds)
            - Sample: random number in [0, 1]
            - If absorbed: remove particle, record energy
            - Else: update weight
         \
           e) Move particle
            - x += ds \* sin(theta)
            - z += ds \* cos(theta)
         \
         f) Check boundaries
            - If left cell: add to outflow bucket, break
            - If E < cutoff: record absorption, break
      \
      3. *Output*:
         - Update particle phase space
         - Add to energy deposition (EdepC)
         - Record outflow (for K4 transfer)
    ],

    [Output], [
      - EdepC[active_cells] ← Energy deposited (for dose)
      - OutflowBuckets[active_cells] ← Particles leaving cells
      - AbsorbedWeight ← Particles that stopped
      \
       *Accuracy*: 1 percent error in dose distribution
    ],
  ),
  caption: [K3 Fine Transport Process],
)

=== Thread Configuration

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Parameter*], [*Value*]),
    [Grid Size], [`(n_active + 255) / 256`],
    [Block Size], [256 threads],
    [Processing], [1 thread per active cell],
    [RNG States], [1 per thread (independent random numbers)],
    [Shared Memory], [4 KB for local bin accumulation],
  ),
  caption: [K3 Thread Configuration],
)

=== Signature

```cpp
__global__ void K3_FineTransport(
    // Input: Active cell list
    const uint32_t* __restrict__ ActiveList,
    const int n_active,

    // Input phase-space
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,

    // Grid & physics
    const int Nx, const int Nz, const float dx, const float dz,
    const int n_active,
    const DeviceRLUT dlut,

    // Grid edges for bin finding
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,

    // Output
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,

    // Outflow buckets
    DeviceOutflowBucket* __restrict__ OutflowBuckets
);
```

#warning-box[
*RNG Implementation Note:* K3 uses a deterministic seed-based RNG, not CUDA curand. Random numbers are generated using hash functions based on cell/slot/bin indices for reproducibility.
]

=== Intra-Bin Sampling

For variance preservation, particles are sampled uniformly within bins:

```cpp
__device__ void sample_intra_bin(
    float& theta,
    int theta_bin, int cell, int slot, int lidx,
    float theta_edges[], int N_theta
) {
    // Deterministic seed from cell/slot/lidx
    unsigned seed = static_cast<unsigned>(
        (cell * 7 + slot * 13 + lidx * 17) ^ 0x5DEECE66DL
    );

    // Generate uniform offset within bin [0, 1)
    float theta_frac = (seed & 0xFFFF) / 65536.0f;

    // Add to bin edge for intra-bin position
    float dtheta = (theta_edges[N_theta] - theta_edges[0]) / N_theta;
    theta = theta_edges[theta_bin] + theta_frac * dtheta;
}
```

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Approach*], [*Effect*]),

    [WITHOUT intra-bin sampling], [
      - All particles in same bin get SAME position
      - → Artificial clustering (bad statistics)
      - → Underestimated variance
    ],

    [WITH intra-bin sampling], [
      - Each particle gets random offset within bin
      - → Preserves continuous distribution
      - → Correct variance
    ],
  ),
  caption: [Intra-Bin Sampling Impact],
)

=== In Plain English

Think of K3 as a detailed weather simulation vs. K2's simple forecast.

*K2 (coarse)*: "Temperature will be around 20°C today." (Uses averages)

*K3 (fine)*: "Temperature will be 18.7°C at 9:23 AM, 19.2°C at 9:24 AM, ..." (Simulates every fluctuation)

For the Bragg peak, we need K3's detail because:
- Small energy changes → large position changes
- Small position errors → large dose errors
- Clinical accuracy requires 1 percent uncertainty

== K4: Bucket Transfer Kernel

=== File

`src/cuda/kernels/k4_transfer.cu`

=== What It Does

K4 moves particles that left their cell during transport (K2/K3) to their new cells. Each cell has 4 "buckets" (one for each direction: ±x, ±z) that catch outgoing particles. K4 reads these buckets and deposits particles in the correct neighboring cells.

=== Why It Exists

Particles move! After K2/K3 transport, particles may have crossed cell boundaries. We need to collect all these "refugees" and put them in their new homes. This maintains spatial locality for the next iteration.

=== How It Works

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Stage*], [*Process*]),

    [Input], [
      *Outflow buckets from all cells* (after K2/K3):
      \
      Each cell has 4 buckets:
      - Bucket 0: Particles leaving +z direction
      - Bucket 1: Particles leaving -z direction
      - Bucket 2: Particles leaving +x direction
      - Bucket 3: Particles leaving -x direction
    ],

    [Neighbor Discovery], [
      *For Cell 5 at position (ix=5, iz=3)*:
      \
      *Neighbors*:
      - +z: Cell 5 + Nx = 5 + 200*1 = 205 (if iz+1 < Nz)
      - -z: Cell 5 - Nx = 5 - 200 = -195 (out of bounds)
      - +x: Cell 5 + 1 = 6 (if ix+1 < Nx)
      - -x: Cell 5 - 1 = 4 (if ix-1 >= 0)
    ],

    [Transfer Process], [
      *For each receiving cell*:
      \
      1. Check 4 neighbors' buckets
      \
      2. For neighbor Cell 6's bucket 3 (particles going -x):
         - Read bucket contents
         - Each entry: (block_id, weight array)
         - Check if block_id already exists in this cell
           - If YES: add weights to existing block
           - If NO: allocate new slot, copy weights
      \
      3. Atomic slot allocation (thread-safe)
         - Multiple threads may try to allocate simultaneously
         - atomicCAS() ensures only one succeeds
    ],

    [Output], [
      All particles now in:
      - Correct spatial cell (based on position)
      - Correct phase-space bin (based on E, theta, x, z)
      \
      Ready for next iteration!
    ],
  ),
  caption: [K4 Bucket Transfer Process],
)

=== Thread Configuration

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Parameter*], [*Value*]),
    [Grid Size], [`(Nx * Nz + 255) / 256`],
    [Block Size], [256 threads],
    [Processing], [1 thread per receiving cell],
    [Atomic Operations], [Yes (slot allocation, weight addition)],
    [Shared Memory], [1 KB for transfer buffer],
  ),
  caption: [K4 Thread Configuration],
)

=== Signature

```cpp
__global__ void K4_BucketTransfer(
    // Input: Outflow buckets from all cells
    const DeviceOutflowBucket* __restrict__ OutflowBuckets,

    // Grid
    const int Nx, const int Nz,

    // Output phase-space
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out
);
```

=== Atomic Slot Allocation

```cpp
__device__ int find_or_allocate_slot(
    uint32_t* block_ids,
    int cell,
    uint32_t bid
) {
    // First pass: check if exists
    for (int slot = 0; slot \< Kb; ++slot) {
        if (block_ids[cell * Kb + slot] == bid) {
            return slot;
        }
    }

    // Second pass: allocate empty slot
    for (int slot = 0; slot \< Kb; ++slot) {
        uint32_t expected = EMPTY_BLOCK_ID;
        uint32_t* ptr = &block_ids[cell * Kb + slot];
        if (atomicCAS(ptr, expected, bid) == expected) {
            return slot;  // Successfully allocated
        }
    }

    return -1;  // No space available
}
```

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Approach*], [*Behavior*]),

    [WITHOUT atomic operations (RACE CONDITION)], [
      - Thread A reads: slot[5] = EMPTY
      - Thread B reads: slot[5] = EMPTY ← Both think it's free!
      - Thread A writes: slot[5] = BLOCK_42
      - Thread B writes: slot[5] = BLOCK_99 ← Overwrites A!
      - → Thread A's data lost
    ],

    [WITH atomicCAS (THREAD-SAFE)], [
      - Thread A: atomicCAS(slot[5], EMPTY, BLOCK_42) → SUCCESS
      - Thread B: atomicCAS(slot[5], EMPTY, BLOCK_99) → FAIL
        (because slot[5] now contains BLOCK_42)
      - Thread B: tries slot[6], ...
      - → All data preserved
    ],
  ),
  caption: [Atomic Operations for Thread Safety],
)

=== In Plain English

Think of K4 as a postal service sorting mail. After transport (K2/K3), all the "mail" (particles) is in the wrong sorting bins. K4 is like postal workers who:

1. Check their assigned bin (cell)
2. Look at mail from 4 neighboring sorting centers
3. Find mail addressed to their bin
4. Sort it into the correct slots

The atomic operations are like two workers trying to use the same sorting slot at the same time. Only one can succeed - the other must find a different slot.

== K5: Weight Audit Kernel

=== File

`src/cuda/kernels/k5_audit.cu`

=== Purpose

Verify weight conservation per cell.

=== What It Does

K5 is the accountant of the simulation. It checks that no particles or energy were lost during transport. For each cell, it verifies: "What came in = What went out + What was absorbed."

=== Why It Exists

Conservation laws are fundamental. If weight is not conserved, the simulation has a bug. K5 catches:
- Particles disappearing (bug in transport logic)
- Energy not deposited (leak in accounting)
- Numerical errors accumulating over time

=== How It Works

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Stage*], [*Process*]),

    [Input], [
      *Phase-space before and after transport*:
      \
      BEFORE (in arrays):
      - Cell 0: weight_in = 1.0
      - Cell 1: weight_in = 0.8
      - Cell 2: weight_in = 0.5
      \
      AFTER (out arrays):
      - Cell 0: weight_out = 0.7
      - Cell 1: weight_out = 0.6
      - Cell 2: weight_out = 0.4
      \
      ABSORBED:
      - Cell 0: cutoff = 0.2, nuclear = 0.1
      - Cell 1: cutoff = 0.15, nuclear = 0.05
      - Cell 2: cutoff = 0.08, nuclear = 0.02
    ],

    [Conservation Check], [
      *For Cell 0*:
      \
      - W_in = 1.0
      - W_out = 0.7
      - W_cut = 0.2
      - W_nuc = 0.1
      \
      Expected = W_out + W_cut + W_nuc = 0.7 + 0.2 + 0.1 = 1.0
      \
      Actual = W_in = 1.0
      \
      Difference = |Expected - Actual| = 0.0
      \
      Relative = Difference / W_in = 0.0 / 1.0 = 0.0
      \
      Pass? YES (0.0 < 1e-6)
      \
      *For Cell 1*:
      \
      Expected = 0.6 + 0.15 + 0.05 = 0.8
      \
      Actual = 0.8
      \
      Difference = 0.0
      \
      Pass? YES
    ],

    [Output], [
      reports[0] = {W_in: 1.0, W_out: 0.7, error: 0.0, pass: true}
      \
      reports[1] = {W_in: 0.8, W_out: 0.6, error: 0.0, pass: true}
      \
      reports[2] = {W_in: 0.5, W_out: 0.4, error: 0.0, pass: true}
      \
      ...
      \
      *Summary*: 100% cells passed conservation check
    ],
  ),
  caption: [K5 Conservation Audit Process],
)

=== Thread Configuration

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Parameter*], [*Value*]),
    [Grid Size], [`(Nx * Nz + 255) / 256`],
    [Block Size], [256 threads],
    [Processing], [1 thread per cell],
    [Memory Access], [Read-only (no atomics needed)],
  ),
  caption: [K5 Thread Configuration],
)

=== Signature

```cpp
__global__ void K5_WeightAudit(
    // Input phase-space (both in and out)
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ block_ids_out,
    const float* __restrict__ values_out,

    // Absorption arrays
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    const double* __restrict__ AbsorbedEnergy_nuclear,

    // Grid
    const int Nx, const int Nz,

    // Output report
    AuditReport* __restrict__ reports
);

struct AuditReport {
    float W_error;  // Weight conservation error
    bool W_pass;    // Pass/fail flag
};
```

=== In Plain English

Think of K5 as balancing your checkbook. You want to verify:

*Starting balance + Deposits = Ending balance + Withdrawals*

If the numbers don't match, something went wrong:
- Maybe you forgot to record a withdrawal (bug in absorption)
- Maybe a deposit was lost (bug in transport)
- Maybe someone stole money (numerical error)

K5 checks this balance for every cell, every iteration. If any cell fails, you know there's a bug to fix.

== K6: Swap Buffers

=== File

`src/cuda/kernels/k6_swap.cu`

=== Purpose

Exchange input/output buffers for next iteration (CPU-side pointer swap).

=== What It Does

K6 prepares the simulation for the next time step by swapping the input and output arrays. The output from this iteration becomes the input for the next iteration.

=== Why It Exists

After K2-K4, particles are in the "out" arrays. For the next iteration, these should be the "in" arrays. Instead of copying 2.2 GB of data, we just swap the pointers (addresses) of the arrays.

=== How It Works

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Stage*], [*Process*]),

    [Before Swap], [
      *Memory Layout*:
      \
      in → [PsiC A] (old input, now obsolete)
      \
      out → [PsiC B] (new output, just computed)
      \
      *For next iteration*:
      - PsiC B should become input
      - PsiC A should become output (will be overwritten)
    ],

    [Pointer Swap], [
      *CPU executes*:
      \
      temp = in
      \
      in = out ← Now points to PsiC B
      \
      out = temp ← Now points to PsiC A
      \
      *No data copied!* Just pointer addresses exchanged
    ],

    [After Swap], [
      *Memory Layout* (ready for next iteration):
      \
      in → [PsiC B] ← Previous output, now input
      \
      out → [PsiC A] ← Previous input, now output
      \
      Next K1-K5 kernels will read from "in" and write to "out", effectively overwriting the old input data
    ],
  ),
  caption: [K6 Buffer Swap Process],
)

=== Implementation

```cpp
// Host-side function (no kernel launch)
void K6_SwapBuffers(PsiC*& in, PsiC*& out) {
    // Three-way XOR swap (no temporary needed)
    PsiC* temp = in;
    in = out;
    out = temp;
}
```

#warning-box[
*Implementation Note:* K6 swaps `PsiC*` pointers (struct containing all arrays), NOT individual array pointers. This is a simpler interface than swapping individual arrays.
]

=== Why No Kernel?

Pointer swap is a CPU operation - GPU memory doesn't need to be modified. This avoids ~2.2 GB of memory copy per iteration.

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Approach*], [*Performance*]),

    [WITHOUT POINTER SWAP (SLOW)], [
      Option 1: Copy data
      \
      cudaMemcpy(new_in, old_out, 2.2 GB, cudaMemcpyDeviceToDevice)
      \
      → Takes ~100 ms per iteration
      \
      → Wastes memory bandwidth
    ],

    [WITH POINTER SWAP (FAST)], [
      Option 2: Swap pointers
      \
      swap(in_ptr, out_ptr)
      \
      → Takes ~0.001 microseconds per iteration
      \
      → Zero memory bandwidth used
      \
      → 100,000x faster!
    ],
  ),
  caption: [Pointer Swap Performance],
)

=== In Plain English

Think of K6 as relabeling boxes instead of moving contents.

*Slow way (copying)*: Take all documents out of Box A, put them in Box B. Takes hours.

*Fast way (swapping)*: Just swap the labels on Box A and Box B. Takes 1 second.

K6 does the fast way. The data stays in memory - we just change what we call "input" and "output."

== Memory Access Patterns

=== Coalesced Access Strategy

#figure(
  table(
    columns: (auto, 4fr),
    inset: 10pt,
    table.header([*Concept*], [*Description*]),

    [Global Memory Layout], [
      *Memory is organized contiguously*:
      \
      Cell 0: Slot 0, Bins 0-511 → Thread 0 reads
      \
      Cell 0: Slot 1, Bins 0-511
      \
      ...
      \
      Cell 0: Slot Kb-1, Bins 0-511
      \
      Cell 1: Slot 0, Bins 0-511 → Thread 1 reads
      \
      Cell 1: Slot 1, Bins 0-511
      \
      ...
    ],

    [Coalesced Access], [
      Thread 0 reads: block_ids_in[0], values_in[0:31]
      \
      Thread 1 reads: block_ids_in[1], values_in[32:63]
      \
      Thread 2 reads: block_ids_in[2], values_in[64:95]
      \
      ...
      \
      Thread 255 reads: block_ids_in[255], values_in[8160:8191]
      \
      GPU combines these into ONE memory transaction:
      \
      "Read block_ids_in[0:255] and values_in[0:8191]"
      \
      → 20-30x faster than scattered access
    ],

    [Scattered Access (BAD)], [
      Thread 0 reads: block_ids_in[0]
      \
      Thread 1 reads: block_ids_in[1000] ← Jump!
      \
      Thread 2 reads: block_ids_in[2000] ← Jump!
      \
      GPU must make 3 separate memory transactions
      \
      → 20-30x slower
    ],
  ),
  caption: [Memory Access Patterns],
)

=== Shared Memory Usage

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*Kernel*], [*Shared Memory*], [*Purpose*]),
    [K1], [256 B], [Partial reduction for weight sum],
    [K3], [4 KB], [Local bin accumulation],
    [K4], [1 KB], [Bucket transfer buffer],
  ),
  caption: [Shared Memory Usage],
)

== Performance Optimization Summary

#figure(
  table(
    columns: (2fr, auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*Technique*], [*Kernel(s)*], [*Benefit*]),
    [Active cell processing], [K2, K3], [Skip empty cells (60-90% savings)],
    [Coarse/fine split], [K2, K3], [3-5x speedup for high-energy],
    [Atomic operations], [K4], [Thread-safe slot allocation],
    [Intra-bin sampling], [K3], [Variance preservation],
    [Pointer swap], [K6], [Avoid 2.2 GB memory copy],
    [Coalesced access], [All], [Max memory bandwidth],
  ),
  caption: [Performance Optimizations],
)

== Launch Configuration Example

```cpp
// Grid dimensions
dim3 grid( (Nx * Nz + 255) / 256 );
dim3 block(256);

// K1: ActiveMask
K1_ActiveMask<<<grid, block>>>(...);

// K3: Fine transport (smaller grid for active cells)
dim3 grid_fine( (n_active + 255) / 256 );
K3_FineTransport<<<grid_fine, block>>>(...);

// Synchronization
cudaDeviceSynchronize();
```

== Summary

The SM_2D CUDA pipeline uses a 6-stage kernel sequence to efficiently transport protons through matter:

- *K1 (ActiveMask)*: Identifies cells needing accurate simulation
- *K2 (Coarse)*: Fast transport for high-energy particles
- *K3 (Fine)*: Accurate Monte Carlo for Bragg peak region
- *K4 (Transfer)*: Moves particles between cells
- *K5 (WeightAudit)*: Verifies weight conservation
- *K6 (Swap)*: Prepares for next iteration

Each kernel is optimized for GPU parallel execution with coalesced memory access, minimal thread divergence, and efficient use of shared memory.

---
#align(center)[
  *SM_2D CUDA Pipeline Documentation*

  #text(size: 9pt)[Version 2.0 - Enhanced with Tutorials]
]
