#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")

= SM_2D: Deterministic Proton Transport Solver

#set align(center)
*Complete Code Documentation*

_Version 1.0.0_

#set align(left)

== Abstract

SM_2D is a deterministic proton transport solver for radiotherapy dose calculation. The system uses GPU acceleration (CUDA) for clinical-speed computation, implementing a hierarchical S-matrix method with comprehensive physics models including Highland multiple Coulomb scattering, Vavilov energy straggling, and nuclear interactions.

#v(1em)

=== Project Statistics

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    align: left,
    table.header([*Parameter*], [*Value*]),
    [Language], [C++17 with CUDA],
    [Lines of Code], [~15,000],
    [GPU Memory], [~4.3 GB per simulation],
    [Accuracy], [Bragg peak <1%, Lateral spread <15%],
    [Compute], [RTX 2080+ (Compute Capability 7.5+)],
  ),
  caption: [Project Overview],
)

#v(1em)

=== Table of Contents

#outline()

== Quick Start

=== What is SM_2D?

SM_2D implements:

* GPU acceleration (CUDA) for clinical-speed calculation
* Hierarchical S-matrix method for deterministic transport
* Comprehensive physics (Highland MCS, Vavilov straggling, nuclear interactions)
* Conservation auditing for numerical accuracy validation

=== Directory Structure

#figure(
  table(
    columns: (auto, 2fr),
    inset: 6pt,
    align: (x, y) => (left, center).at(x),
    table.header([*Directory*], [*Description*]),
    [`run_simulation.cpp`], [Main entry point],
    [`sim.ini`], [Configuration file],
    [`src/core/`], [Data structures (grids, storage, encoding)],
    [`src/physics/`], [Physics models (MCS, straggling, nuclear)],
    [`src/cuda/kernels/`], [CUDA kernels (K1-K6 pipeline)],
    [`src/lut/`], [NIST data & range-energy tables],
    [`src/source/`], [Beam sources (pencil, Gaussian)],
    [`src/boundary/`], [Boundary conditions & loss tracking],
    [`src/audit/`], [Conservation checking],
    [`src/validation/`], [Physics validation],
    [`src/utils/`], [Logging, memory tracking],
    [`tests/`], [Unit tests (GoogleTest)],
  ),
  caption: [Directory Structure],
)

== System Overview

=== CUDA Kernel Pipeline

The simulation implements a 6-stage CUDA kernel pipeline:

#figure(
  table(
    columns: (auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*Kernel*], [*Purpose*]),
    [K1], [Find active cells (E < E_sub.trigger)],
    [K2], [Coarse transport for high-energy particles],
    [K3], [Fine transport with full physics],
    [K4], [Bucket transfer between cells],
    [K5], [Conservation audit],
    [K6], [Buffer swap for next iteration],
  ),
  caption: [CUDA Kernel Pipeline],
)

=== Key Concepts

==== Phase-Space Representation

Particles are represented in 4D phase space:

* $ theta $ (Angle): 512 bins from -90 deg to +90 deg
* $ E $ (Energy): 256 bins from 0.1 to 250 MeV (log-spaced)
* $ x_"sub" $: 4 sub-bins within each cell (transverse)
* $ z_"sub" $: 4 sub-bins within each cell (depth)

==== Block-Sparse Storage

```cpp
// 24-bit block ID = (b_E << 12) | b_theta
uint32_t block_id = encode_block(theta_bin, energy_bin);

// 512 local bins per block for variance preservation
uint16_t local_idx = encode_local_idx_4d(theta_local, E_local, x_sub, z_sub);
```

==== Hierarchical Transport

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*Energy Range*], [*Transport Method*], [*Reason*]),
    [E > 10 MeV], [Coarse (K2)], [Fast, approximate physics],
    [E <= 10 MeV], [Fine (K3)], [Full physics for Bragg peak],
  ),
  caption: [Transport Methods by Energy],
)

== Physics Summary

=== Multiple Coulomb Scattering (Highland)

$ sigma_"theta" = (13.6 " MeV" / (beta c p)) * sqrt(x / X_0) * [1 + 0.038 * ln(x / X_0)] / sqrt(2) $

* X_."0" (water): 360.8 mm
* 2D correction: $ 1 / sqrt(2) $ for proper variance

=== Energy Straggling (Vavilov)

Three regimes based on $ kappa = xi / T_"max" $:

* $ kappa > 10 $: Bohr (Gaussian)
* $ 0.01 < kappa < 10 $: Vavilov (interpolation)
* $ kappa < 0.01 $: Landau (asymmetric)

=== Nuclear Attenuation

$ W * exp(-sigma(E) * dif s) $

Energy-dependent cross-section from ICRU 63.

=== Step Control (R-based)

$ dif s = min(0.02 * R, 1 " mm", cell_"size") $

Uses range-energy LUT instead of stopping power for stability.

== Memory Layout

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Buffer*], [*Size*], [*Purpose*]),
    [PsiC_in/out], [1.1 GB each], [Phase-space storage],
    [EdepC], [0.5 GB], [Energy deposition],
    [AbsorbedWeight_*], [0.5 GB], [Cutoff/nuclear tracking],
    [AbsorbedEnergy_*], [0.25 GB], [Nuclear energy budget],
    [BoundaryLoss], [0.1 GB], [Boundary losses],
    [ActiveMask/List], [0.5 GB], [Active cell tracking],
  ),
  caption: [GPU Memory Layout],
)

#text(size: 10pt)[*Total: ~4.3 GB GPU memory*]

== Accuracy Targets

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Observable*], [*Target*], [*Status*]),
    [Bragg peak position], [±2%], [✅ Pass],
    [Lateral sigma (mid-range)], [±15%], [✅ Pass],
    [Lateral sigma (Bragg)], [±20%], [✅ Pass],
    [Weight conservation], [<1e-6], [✅ Pass],
    [Energy conservation], [<1e-5], [✅ Pass],
  ),
  caption: [Validation Results],
)

== Key Classes

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*Class*], [*Module*], [*Purpose*]),
    [EnergyGrid], [core], [Log-spaced energy bins],
    [AngularGrid], [core], [Uniform angle bins],
    [PsiC], [core], [Hierarchical phase-space storage],
    [RLUT], [lut], [Range-energy interpolation],
    [PencilSource], [source], [Deterministic beam source],
    [GaussianSource], [source], [Stochastic beam source],
    [GlobalAudit], [audit], [Conservation tracking],
    [BraggPeakResult], [validation], [Peak analysis],
  ),
  caption: [Key Classes],
)

== Further Reading

* #link("architecture.typ")[Architecture Overview] - Complete system design
* #link("physics.typ")[Physics Models] - Complete physics reference
* #link("data_structures.typ")[Data Structures] - Storage and encoding details
* #link("cuda_pipeline.typ")[CUDA Pipeline] - Detailed kernel documentation
* #link("api.typ")[API Reference] - Function-by-function documentation

== References

#figure(
  table(
    columns: (auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*Source*], [*Topic*]),
    [NIST PSTAR], [Stopping powers & ranges],
    [PDG 2024], [Highland formula],
    [ICRU 63], [Nuclear cross-sections],
    [Vavilov 1957], [Energy straggling],
  ),
  caption: [References],
)

---
#set align(center)
*Generated for SM_2D Proton Therapy Transport Solver*

#text(size: 9pt)[MIT License - Version 1.0.0]
