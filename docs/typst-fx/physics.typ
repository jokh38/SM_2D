#set text(size: 11pt, lang: "en")
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

// Define custom box elements
#let concept-box(body) = block(
  fill: rgb("#e0f0ff"),
  inset: 10pt,
  radius: 5pt,
  body
)

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

#let analogy-box(body) = block(
  fill: rgb("#f2e6ff"),
  inset: 10pt,
  radius: 5pt,
  body
)

#let clinical-box(body) = block(
  fill: rgb("#ffe6e6"),
  inset: 10pt,
  radius: 5pt,
  body
)

#show math.equation: set text(weight: "regular")

= Proton Therapy Physics: A Visual Guide

== Welcome to SM_2D Physics

This document explains the physics models used in SM_2D in plain English. We'll break down complex formulas into understandable concepts with real-world analogies and clinical significance.

---

== What is Proton Therapy? (The Big Picture)

#concept-box[
*Proton Therapy* is a type of radiation treatment that uses proton beams to destroy cancer cells. Unlike X-rays that pass through the body, protons stop at a specific depth, depositing most of their energy right there.

*Why this matters:* This allows doctors to deliver high doses to tumors while sparing healthy tissue behind them.
]

#analogy-box[
*Real-world Analogy:*

Think of protons like tiny trucks carrying energy. X-rays are like cars that keep driving through your body, dropping packages (radiation) along the way. Protons are like trucks that drive in, drop ALL their packages at the destination (the tumor), then stop. Much less mess on the road behind!
]

---

== Physics Models Overview

SM_2D simulates how protons travel through water (which is very similar to human tissue). We need to model:

1. *Multiple Coulomb Scattering* - Protons bouncing off atoms
2. *Energy Straggling* - Variation in energy loss
3. *Nuclear Attenuation* - Protons disappearing due to nuclear reactions
4. *Step Control* - How we simulate the journey step by step
5. *Fermi-Eyges Theory* - How proton beams spread out sideways

---

== 1. Multiple Coulomb Scattering (MCS)

=== In Plain English

#concept-box[
*Multiple Coulomb Scattering (MCS)* describes how protons get deflected (bounced) when they pass near atomic nuclei. Imagine throwing a tiny ball through a forest - it doesn't go in a perfectly straight line because it keeps brushing against trees.

*Key Point:* Each individual deflection is tiny, but after millions of interactions, the proton's path becomes significantly curved.
]

=== Why This Matters for Treatment

#clinical-box[
*Clinical Significance:*

MCS causes the proton beam to "blur" or spread out as it travels deeper into the body. This means:
- *Good:* Helps cover the entire tumor volume
- *Challenge:* Can cause dose to spill into healthy tissue
- *Solution:* We must account for MCS when planning beam angles and shaping the beam

*Example:* A 150 MeV proton beam might spread from 5 mm to 15 mm wide by the time it reaches a deep-seated tumor.
]

=== Visual Diagram: Scattering Process

#figure(
  table(
    columns: (1fr, 3fr),
    inset: 8pt,
    [*Phase*], [*Process*],
    [Incoming Proton], [arrow.down, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", arrow.down],
    [During Transport], [arrow.down, " ↗ ↘ ↗ ↘ ↗ ↘ ↗ ↘ ↗ ↘", arrow.down],
    [], [Multiple tiny deflections at each atomic encounter],
    [Final Direction], [arrow.down, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", arrow.down],
    [], [Final direction changed (arrows exaggerated for visibility)],
  ),
  caption: [Proton Scattering Journey - Multiple tiny deflections accumulate to change proton's direction],
 )

=== The Highland Formula

#tip-box[
*Simple Explanation:*

The Highland formula tells us HOW MUCH a proton beam will spread out after traveling a certain distance. It's like predicting how wide a spray of water will be after it travels through the air.

The formula uses:
- *Proton energy* - Faster protons scatter less
- *Distance traveled* - More distance = more scattering
- *Material* - Some materials scatter more than others
]

=== The Formula Explained

$ sigma_"theta" = (13.6 " MeV" / (beta c p)) times z times sqrt(x / X_0) times [1 + 0.038 times ln(x / X_0)] / sqrt(2) $

Let's break this down:

#table(
  columns: (auto, 4fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*Symbol*], [*What It Means (Plain English)*], [*Example Value*]),
  [$sigma_"theta"$], [Angular spread - how much the beam spreads out], [~2-5 degrees],
  [$13.6 " MeV"$], [Universal constant from particle physics], [Fixed],
  [$beta c p$], [Proton momentum × velocity - higher = less scattering], [140 MeV/c for 150 MeV proton],
  [$z$], [Proton charge number (always 1 for protons)], [1],
  [$x$], [Distance traveled in the material], [0-300 mm],
  [$X_0$], [Radiation length - material's "scattering power"], [360.8 mm for water],
  [$1 / sqrt(2)$], [Correction factor for 2D vs 3D], [0.707],
)

=== Intuitive Understanding

#analogy-box[
*The "Bowling Alley" Analogy:*

Imagine bowling down a very long alley:
- *High energy (fast protons)* = Heavy bowling ball - harder to deflect
- *Low energy (slow protons)* = Light ping-pong ball - bounces everywhere
- *High Z material (like bone)* = Pins placed very close together
- *Low Z material (like water/fat)* = Pins placed far apart

The Highland formula predicts how much your ball will deviate from a straight path.
]

=== Energy Dependence

#figure(
  table(
    columns: (auto, 4fr),
    inset: 8pt,
    align: left,
    table.header([*Energy*], [*Scattering Behavior*]),
    [High (200 MeV)], [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ (Minimal spread)],
    [Medium (100 MeV)], [━━━━━━━━━━━━━━━━━━━━━━━━━→ (Moderate spread)],
    [Low (50 MeV)], [━━━━━━━━━━━━━━━━━→ (Significant spread)],
    [Very Low (20 MeV)], [━━━━━━━━━→ (Large spread - at Bragg peak)],
  ),
  caption: [Scattering vs Energy - Lower energy protons scatter much more],
)

#tip-box[
*Clinical Takeaway:* Low-energy protons (near the Bragg peak) scatter much more. This is why the dose distribution "blooms" at the end of the proton's range.
]

=== Code Implementation (The Right Way)

#warning-box[
*CRITICAL Implementation Detail:*

For accurate multi-step scattering, we must accumulate VARIANCE, not standard deviations. This is a common mistake!

*WRONG:*
```cpp
// DON'T DO THIS!
sigma_total += sigma_theta;  // Wrong!

// DON'T DO THIS EITHER!
theta_total += sample_mcs(sigma_theta);  // Sample each step separately
```

*CORRECT:*
```cpp
// DO THIS: Accumulate variance
sigma_2_total += sigma_theta * sigma_theta;

// Then sample ONCE from total variance
float theta_scatter = sqrt(sigma_2_total) * sample_normal();
```

*Why?* Variances add for independent random processes. Standard deviations don't!
]

=== The 2D Projection Correction

#concept-box[
*3D vs 2D Scattering:*

Real protons scatter in 3D space (all directions). But our simulation is 2D (x-z plane). The Highland formula gives 3D scattering angles, so we must convert to 2D.

*The math:* $sigma_(2D) = sigma_(3D) / sqrt(2)$

*Why divide by sqrt(2)?* Statistical mechanics - projecting a 3D Gaussian distribution onto a 2D plane.
]

---

== 2. Energy Straggling

=== In Plain English

#concept-box[
*Energy Straggling* is the variation in how much energy different protons lose when traveling the same distance. Even if all protons start with the same energy and travel the same path, they won't all have the same energy at the end.

*Why?* Energy loss is a random process. Some protons happen to have more "close encounters" with atoms, losing more energy. Others have fewer interactions, losing less energy.
]

=== Why This Matters for Treatment

#clinical-box[
*Clinical Significance:*

Energy straggling causes the Bragg peak to "smear out" or broaden:

- *Without straggling:* Sharp, narrow Bragg peak
- *With straggling:* Wider, less pronounced peak

*Impact:* Affects the sharpness of the dose fall-off beyond the tumor. Too much straggling = dose spilling into healthy tissue behind the tumor.

*Clinical Reality:* Straggling sets a fundamental limit on how sharp we can make the dose fall-off, regardless of beam delivery technology.
]

=== Visual Diagram: Straggling Effect

#figure(
  table(
    columns: (1fr, 3fr),
    inset: 8pt,
    [*Phase*], [*Energy Distribution*],
    [Initial], [All protons: 150 MeV],
    [After 150 mm], [$ "●●●●●●●●●●●●●●●●●●●●●●●●●●●●●" $],
    [], [Energy range: 140-160 MeV, Average: 148 MeV, Straggling width: ~7 MeV],
    [Interpretation], [Some protons lost more energy (left), others lost less (right)],
  ),
  caption: [Energy Straggling Visualization - Individual protons have different energies after the same path],
)

=== Three Regimes of Straggling

#table(
  columns: (auto, auto, 2fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*Regime*], [*κ Parameter*], [*Distribution Shape*], [*Clinical Context*]),
  [Bohr], [κ > 10], [Symmetric bell curve (Gaussian)], [High energy, thick absorbers],
  [Vavilov], [0.01 < κ < 10], [Intermediate shape], [Most clinical situations],
  [Landau], [κ < 0.01], [Asymmetric with long tail], [Low energy, thin layers],
)

#tip-box[
*The κ (kappa) parameter tells us which regime we're in:*

$ kappa = xi / T_("max") $

Where:
- $xi$ = "characteristic energy loss" - typical amount of energy lost
- $T_("max")$ = maximum possible energy loss in a single collision

*High κ* = Many small energy transfers (Bohr/Gaussian regime)
*Low κ* = Few large energy transfers possible (Landau regime)
]

=== The Vavilov Parameter Formula

$ kappa = xi / T_("sub").max $

Where:

$ xi = (K / 2) times (Z / A) times (z^2 / beta^2) times rho times x $

$ T_("sub").max = (2 m_e c^2 beta^2 gamma^2) / (1 + 2 gamma m_e / m_p + (m_e / m_p)^2) $

Let's understand each part:

#table(
  columns: (auto, 4fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*Symbol*], [*Meaning*], [*Value for Water*]),
  [$K$], [Universal constant from quantum electrodynamics], [0.307 MeV cm²/g],
  [$Z / A$], [Ratio of atomic number to mass number], [0.555 for water],
  [$z$], [Projectile charge (1 for protons)], [1],
  [$beta$], [Proton velocity / speed of light], [0.1-0.6 for therapy],
  [$rho$], [Material density], [1.0 g/cm³ for water],
  [$x$], [Path length], [0-300 mm],
  [$m_e c^2$], [Electron rest energy], [0.511 MeV],
  [$m_p$], [Proton rest mass], [938.27 MeV],
)

=== Bohr Straggling

#analogy-box[
*The "Random Walk" Analogy:*

Energy straggling is like a random walk. Each proton takes a different number of "steps" (energy transfers) as it travels. The spread in final energy follows statistical rules.

Bohr straggling applies when each step is small and there are MANY steps. This gives a Gaussian (normal) distribution.

*Formula:* $sigma = (kappa_0 / beta) times sqrt(x)$

Notice the $sqrt(x)$ dependence - this is characteristic of random walk processes!
]

=== Bohr Formula Implementation

```cpp
__host__ __device__ float bohr_straggling_sigma(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // Bohr formula (simplified for water)
    float kappa_0 = 0.156f;  // Pre-computed for water
    float sigma = kappa_0 * sqrt(ds) / beta;

    return sigma;
}
```

#concept-box[
*Key Dependencies:*

- *1/beta*: Slower protons (low beta) have MORE straggling because they spend more time near atoms
- *sqrt(ds)*: Spreads as the square root of distance (random walk)
- *Material*: Different materials have different kappa_0 values
]

=== Landau Straggling

#tip-box[
*When Low κ (Landau Regime):*

At low energies or in thin layers, energy loss is dominated by occasional LARGE energy transfers. This creates an asymmetric distribution with:

- *Sharp peak* at low energy loss (most protons lose little)
- *Long tail* to high energy loss (few protons lose a lot)

*Clinical relevance:* This tail creates "range straggling" - different protons stop at different depths, smearing the Bragg peak.
]

=== Most Probable Energy Loss

$ Delta_p = xi [ln(xi / T_("sub").max) + ln(1 + beta^2 gamma^2) + 0.2 - beta^2 - delta / 2]$

#warning-box[
*Important:* The most probable energy loss (peak of Landau distribution) is NOT the same as the mean energy loss (Bethe-Bloch formula)!

- *Most probable:* Where the distribution peaks
- *Mean:* Average energy loss (higher due to tail)

The difference matters for accurate Bragg peak modeling.
]

---

== 3. Nuclear Attenuation

=== In Plain English

#concept-box[
*Nuclear Attenuation* means protons can disappear from the beam due to nuclear reactions with atoms. When a proton gets too close to an atomic nucleus, it might:

1. Get absorbed by the nucleus
2. Cause the nucleus to break apart (nuclear fragmentation)
3. Knock out other particles (secondary particles)

In all cases, the original proton is gone - it no longer contributes to the therapeutic dose.
]

=== Why This Matters for Treatment

#clinical-box[
*Clinical Significance:*

Nuclear attenuation reduces the number of protons reaching the tumor:

- *Dose reduction:* Fewer protons = lower dose than predicted
- *Secondary particles:* Nuclear reactions create secondary radiation (neutrons, alpha particles)
- *Neutron dose:* Can increase cancer risk in healthy tissue
- *Treatment planning:* Must account for proton loss to deliver correct dose

*Typical magnitude:* ~1-3% of protons are lost per 10 cm of water travel
]

=== Visual Diagram: Nuclear Reactions

#figure(
  table(
    columns: (auto, 3fr, 2fr),
    inset: 8pt,
    table.header([*Reaction Type*], [*Process & Outcome*], [*Description*]),
    [1. Elastic Scattering], [$ p + "nucleus" arrow.r "nucleus + p'" $], [Proton survives, changes direction],
    [2. Non-Elastic Scattering], [$ p + "nucleus" arrow.r "secondary" " particles" $], [Proton absorbed, creates fragments],
    [3. Nuclear Fragmentation], [$ p + "nucleus" arrow.r "p' + lighter" " nucleus" $], [Original nucleus breaks apart],
  ),
  caption: [Nuclear Interaction Types - Only elastic scattering keeps the original proton],
)

=== The Cross-Section Model

#concept-box[
*Cross-Section (σ):*

The "cross-section" is the effective target area of a nucleus for nuclear reactions. Larger cross-section = more likely to interact.

*Analogy:* Think of nuclei as circular targets. The cross-section is the area of the target. Larger targets are easier to hit.

*Units:* cm² or mm² (very tiny numbers!)
*Typical values:* 10⁻²⁶ to 10⁻²⁴ cm² per nucleus
]

#tip-box[
*Macroscopic Cross-Section (Σ):*

In simulation, we use the macroscopic cross-section Σ, which is the cross-section per unit volume:

$ Sigma = N times sigma $

Where:
- $N$ = number density of nuclei (nuclei/cm³)
- $sigma$ = microscopic cross-section (cm²)

For water: $Sigma ≈ 0.0012 " to " 0.0016 " mm"^-1$ (energy dependent)
]

=== Energy Dependence

#table(
  columns: (auto, 2fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*Energy Range*], [*Cross-Section Behavior*], [*Why*]),
  [\<= 5 MeV], [Negligible], [Proton can't overcome nuclear barrier],
  [5-20 MeV], [Rising rapidly], [More nuclear reactions become possible],
  [20-100 MeV], [Slowly decreasing], [Higher energy = less interaction time],
  [>= 100 MeV], [Approximately constant], [High-energy limit reached],
)

=== Implementation

```cpp
__host__ __device__ float Sigma_total(float E_MeV) {
    // Below Coulomb barrier for hydrogen/oxygen: negligible nuclear reactions
    if (E_MeV < 5.0f) {
        return 0.0f;
    }

    // Reference values at 100 MeV (ICRU 63)
    constexpr float sigma_100 = 0.0012f;  // mm⁻¹ at 100 MeV
    constexpr float E_ref = 100.0f;       // Reference energy [MeV]

    if (E_MeV >= 20.0f) {
        // Logarithmic energy dependence for therapeutic range (20-250 MeV)
        // σ(E) = σ_100 * [1 - 0.15 * ln(E/100)]
        // The factor 0.15 gives ~30% reduction from 20 to 200 MeV
        float log_factor = 1.0f - 0.15f * logf(E_MeV / E_ref);
        float sigma = sigma_100 * fmaxf(log_factor, 0.4f);  // Minimum at 40% of reference
        return sigma;
    } else {
        // Low energy (5-20 MeV): linear ramp from 0 at 5 MeV to sigma_20 at 20 MeV
        constexpr float sigma_20 = 0.0016f;  // mm⁻¹ at 20 MeV
        float frac = (E_MeV - 5.0f) / 15.0f;  // 0 to 1
        return sigma_20 * frac;
    }
}
```

#tip-box[
*Key Implementation Notes:*

1. The formula uses a logarithmic correction: $1 - 0.15 times ln(E / 100)$
2. Values are clamped at 40% of the reference (minimum cross-section)
3. Below 20 MeV, a linear ramp is used from the Coulomb barrier (5 MeV)
]

#analogy-box[
*The "Filter" Analogy:*

Think of nuclear attenuation as the beam passing through a filter:

- *High energy:* Loose filter (most protons pass through)
- *Low energy:* Tighter filter (more protons get caught)
- *Thick material:* Longer filter (more chances to get caught)

The cross-section Σ tells us how "tight" the filter is at any given energy.
]

=== Survival Probability

#concept-box[
*Survival Probability:*

What fraction of protons DON'T undergo nuclear reactions while traveling a distance ds?

$ P_("survival") = e^(-Sigma times "ds") $

This is the standard exponential attenuation law (also used for X-ray absorption).
]

```cpp
__device__ float survival_probability(float E_MeV, float ds) {
    float sigma = Sigma_total(E_MeV);
    return exp(-sigma * ds);
}
```

#tip-box[
*For small steps (ds ≪ 1/Sigma):*

We can use the linear approximation:

$ P_("survival") ≈ 1 - Sigma times "ds" $

This is what we use in Monte Carlo to determine if a proton is "killed" in each step.
]

=== Energy Conservation

#warning-box[
*CRITICAL: Tracking Removed Energy:*

When protons are removed by nuclear interactions, their energy doesn't just disappear! We must track it for energy conservation audits:

```cpp
__device__ void apply_nuclear_attenuation(
    float& weight,      // Modified: weight *= survival
    double& energy_rem, // Accumulator: energy removed by nuclear
    float E_MeV,
    float ds
) {
    float sigma = Sigma_total(E_MeV);
    float prob_interaction = 1.0f - exp(-sigma * ds);

    float weight_removed = weight * prob_interaction;
    weight -= weight_removed;

    // Track energy for conservation audit
    energy_rem += weight_removed * E_MeV;  // ← CRITICAL!
}
```

*Why?* Total energy in = energy deposited + energy carried away by secondary particles
]

=== Clinical Impact

#clinical-box[
*Secondary Radiation:*

Nuclear reactions create secondary particles:
- *Neutrons:* Highly penetrating, can travel far from the beam path
- *Alpha particles:* Very short range, high LET (potentially more damaging)
- *Heavy fragments:* From target nucleus breaking apart

*Clinical concerns:*
1. *Secondary cancer risk:* Neutrons can cause DNA damage outside the treatment area
2. *Imaging artifacts:* PET activation from nuclear reactions
3. *Dose perturbation:* Secondary particles deposit small doses elsewhere

*Current practice:* Modern treatment planning systems include nuclear interaction models for accuracy better than 2%.
]

---

== 4. R-Based Step Control

=== In Plain English

#concept-box[
*R-Based Step Control* is how we simulate the proton's journey through the body. Instead of calculating every microscopic interaction, we take "steps" - discrete chunks of the path where we calculate the average physics.

The "R" stands for Range - how far the proton can travel before stopping. We use range (not energy) to control step sizes because it's more stable and accurate.
]

=== Why This Matters for Treatment

#clinical-box[
*Clinical Significance:*

Accurate step control is crucial for:
- *Bragg peak position:* Small errors in step size → large errors in where the proton stops
- *Dose calculation:* Energy deposition must be computed correctly each step
- *Simulation speed:* Too many steps = slow; too few = inaccurate

*The challenge:* Protons slow down rapidly near the end of their range, so we need very small steps there. R-based control automatically adjusts step size based on remaining range.
]

=== Visual Diagram: Step Size Adaptation

#figure(
  table(
    columns: (auto, 2fr, 2fr),
    inset: 8pt,
    align: left,
    table.header([*Energy*], [*Step Size*], [*Visualization*]),
    [High (150 MeV)], [Large: 1 mm], [━ ┃ ━ ┃ ━ ┃ ━ ┃ ━ ┃ ━],
    [Medium (70 MeV)], [Medium: 0.5 mm], [━ ┃ ━ ┃ ━ ┃ ━ ┃ ━ ┃ ━],
    [Low (10 MeV)], [Small: 0.05 mm], [┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┃],
  ),
  caption: [Adaptive Step Sizing - Steps get smaller as proton slows down near Bragg peak],
)

=== The CSDA Approximation

#concept-box[
*CSDA = Continuous Slowing Down Approximation*

This is the key idea behind R-based step control:

$ "dR" / "ds" = -1 $

*What it means:* For every millimeter the proton travels (ds), its remaining range decreases by exactly one millimeter (dR = -ds).

*Why this works:* In the CSDA approximation, we assume the proton loses energy continuously (not in discrete jumps). This is a good approximation for our purposes.
]

#analogy-box[
*The "Fuel Gauge" Analogy:*

Think of proton range like a car's fuel gauge:
- *Range R* = "Miles remaining" on the fuel gauge
- *Distance traveled ds* = Miles actually driven

CSDA says: For every mile you drive, the "miles remaining" decreases by exactly 1 mile.

*Reality check:* This isn't perfect (hilly terrain, AC usage, etc.), but it's a good approximation. Similarly, CSDA isn't perfect (straggling, nuclear reactions), but it works well for Monte Carlo.
]

=== Maximum Step Size Formula

```cpp
__host__ __device__ float compute_max_step_physics(float E, const RLUT& lut, float dx = 1.0f, float dz = 1.0f) {
    float R = lut.lookup_R(E);  // CSDA range [mm]

    // Primary limit: fraction of remaining range
    float delta_R_max = 0.02f * R;  // 2% of range

    // Energy-dependent refinement factor near Bragg peak
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        // Very near end of range: extreme refinement
        dS_factor = 0.2f;
        delta_R_max = fminf(delta_R_max, 0.1f);  // Max 0.1 mm
    } else if (E < 10.0f) {
        // Near Bragg peak: high refinement
        dS_factor = 0.3f;
        delta_R_max = fminf(delta_R_max, 0.2f);  // Max 0.2 mm
    } else if (E < 20.0f) {
        // Bragg peak region: moderate refinement
        dS_factor = 0.5f;
        delta_R_max = fminf(delta_R_max, 0.5f);  // Max 0.5 mm
    } else if (E < 50.0f) {
        // Pre-Bragg: light refinement
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.7f);  // Max 0.7 mm
    }

    // Apply refinement factor
    delta_R_max = delta_R_max * dS_factor;

    // Hard limits
    delta_R_max = fminf(delta_R_max, 1.0f);  // Max 1 mm
    delta_R_max = fmaxf(delta_R_max, 0.05f);  // Min 0.05 mm

    // Cell size limit (prevents skipping cells)
    float cell_limit = 0.25f * fminf(dx, dz);
    delta_R_max = fminf(delta_R_max, cell_limit);

    return delta_R_max;
}
```

#tip-box[
*Key Limits Explained:*

1. *2% of range (primary)*: Ensures steps are proportional to remaining journey
2. *Energy-dependent refinement*: Steps get smaller as proton slows down
3. *Hard limits (0.05-1.0 mm)*: Prevents absurdly large or tiny steps
4. *Cell size limit*: Prevents skipping entire grid cells (would miss dose deposition!)
]

=== R-Based vs S-Based Energy Updates

#table(
  columns: (auto, 3fr, 2fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*Method*], [*How It Works*], [*Pros*], [*Cons*]),
  [S-based (stopping power)], [Use Bethe-Bloch: $E_("out") = E_("in") - S(E) times "ds"$], [Simple, direct], [Unstable near Bragg peak - stopping power changes rapidly!],
  [R-based (range)], [Use range lookup: $E_("out") = E^(-1)(R(E) - "ds")$], [Stable everywhere, exact (CSDA)], [Needs range lookup table],
)

#warning-box[
*Why S-Based Fails Near Bragg Peak:*

The stopping power S(E) changes VERY rapidly near the Bragg peak:
- At 10 MeV: S ≈ 100 MeV/cm
- At 5 MeV: S ≈ 200 MeV/cm
- At 2 MeV: S ≈ 500 MeV/cm

Using S(E) × ds with even small step sizes gives wrong answers because S changes significantly during the step!

R-based avoids this problem entirely by using range instead.
]

=== Energy Update Implementation

```cpp
// Compute energy after step (R-based method)
__device__ float compute_energy_after_step(float E_in, float ds, const RLUT& lut) {
    float R_in = lut.lookup_R(E_in);
    float R_out = R_in - ds;  // CSDA: dR/ds = -1
    return lut.lookup_E_inverse(R_out);  // Inverse lookup
}

// Energy deposited in step
__device__ float compute_energy_deposition(float E_in, float ds, const RLUT& lut) {
    float E_out = compute_energy_after_step(E_in, ds, lut);
    return E_in - E_out;  // All energy loss becomes deposition
}
```

#concept-box[
*Lookup Tables (LUT):*

We use pre-computed tables for R(E) and E(R) because:
- *Speed:* Table lookup is faster than calculating Bethe-Bloch
- *Accuracy:* Uses NIST PSTAR data (gold standard)
- *Stability:* Avoids numerical issues with formula evaluation

*Interpolation:* We interpolate between table points for smooth results
]

---

== 5. Fermi-Eyges Lateral Spread Theory

=== In Plain English

#concept-box[
*Fermi-Eyges Theory* predicts how a proton beam spreads out sideways (laterally) as it travels through material. This is different from MCS - instead of tracking individual proton deflections, Fermi-Eyges calculates the statistical spread of the entire beam.

*Key idea:* The beam width at any depth depends on ALL the scattering that happened on the way there, not just the scattering at that point.
]

=== Why This Matters for Treatment

#clinical-box[
*Clinical Significance:*

Lateral beam spread is CRITICAL for treatment planning:

- *Field margins:* Must account for beam spread to ensure tumor coverage
- *Organ sparing:* Knowing spread helps avoid critical structures
- *Beam shaping:* Apertures and compensators depend on knowing lateral penumbra
- *Dose painting:* Modern techniques require precise knowledge of dose distribution

*Clinical example:* A 3 cm radius tumor at 20 cm depth might need a 4 cm radius beam to account for lateral spread (~1 cm expansion).
]

=== Visual Diagram: Beam Spreading

#figure(
  table(
    columns: (2fr, 2fr, 1fr),
    inset: 8pt,
    table.header([*Depth*], [*Visualization*], [*Width*]),
    [Surface (z = 0)], [$ "█████████" $], [5 mm],
    [5 cm], [$ "█████████████████" $], [8 mm],
    [10 cm], [$ "█████████████████████████████" $], [12 mm],
    [15 cm (Bragg peak)], [$ "████████████████████████████████████████████████" $], [18 mm],
  ),
  caption: [Lateral Beam Spread Evolution - Beam widens with depth due to accumulated scattering],
)

=== Scattering Power (T)

#concept-box[
*Scattering Power T:*

The "scattering power" T(z) tells us how rapidly scattering is happening at depth z:

$ T(z) = "d"sigma_theta^2 / "dz" $

*Physical meaning:* Rate of change of angular variance with depth

*Units:* radians² per mm (very small numbers!)

*Analogy:* Think of T as "scatterability" - how much the material at depth z wants to scatter the beam.
]

=== The Fermi-Eyges Moments

To calculate beam spread, we track three "moments" - cumulative measures of scattering along the path:

$ A_0(z) = integral_0^z T(z') dif z' $
$ A_1(z) = integral_0^z z' times T(z') dif z' $
$ A_2(z) = integral_0^z z'^2 times T(z') dif z' $

#table(
  columns: (auto, 3fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*Moment*], [*Physical Meaning*], [*Role in Formula*]),
  [$A_0$], [Total angular variance accumulated], [How much the beam direction has spread],
  [$A_1$], [First spatial moment], [Weighted by position - early scattering matters less],
  [$A_2$], [Second spatial moment], [Weighted by position² - late scattering matters more],
)

#analogy-box[
*The "Weighted History" Analogy:*

Imagine you're tracking your spending habits:
- $A_0$ = Total spending (all transactions count equally)
- $A_1$ = Spending weighted by time (recent spending counts more)
- $A_2$ = Spending weighted by time² (very recent spending counts much more)

Fermi-Eyges says: Scattering near the end of the path matters MORE for final beam position than scattering at the beginning.
]

=== Lateral Variance Formula

$ sigma_x^2(z) = A_0 times z^2 - 2 times A_1 times z + A_2 $

#tip-box[
*Intuitive Understanding:*

This formula combines three effects:

1. *$A_0 z^2$*: Angular spread multiplied by distance²
   - Wide angle × long distance = wide beam

2. *$-2 A_1 z$*: Correction for early scattering
   - Early scattering has less time to affect beam position

3. *$A_2$*: Pure scattering contribution
   - Direct lateral displacement from scattering events

*Result:* Net lateral variance at depth z
]

=== Implementation

```cpp
// Scattering power from Highland formula
__device__ float fermi_eyges_scattering_power(float E_MeV) {
    float sigma_theta = highland_sigma(E_MeV, 1.0f, X0_water);
    return sigma_theta * sigma_theta;  // T = σ² per mm
}

// Moment accumulation during transport
struct FermiEygesMoments {
    double A0 = 0.0;  // Total angular variance
    double A1 = 0.0;  // First spatial moment
    double A2 = 0.0;  // Second spatial moment
};

__device__ void device_update_fermi_eyges_moments(
    FermiEygesMoments& moments,
    float z,
    float ds,
    float E_MeV
) {
    float T = fermi_eyges_scattering_power(E_MeV);

    moments.A0 += T * ds;
    moments.A1 += T * z * ds;
    moments.A2 += T * z * z * ds;
}
```

#concept-box[
*Numerical Integration:*

We're approximating the integrals using the rectangle rule:

$ integral_0^z T(z') "dz'" ≈ sum_i T(z_i) times Delta s $

For each step i:
- Calculate scattering power T at current energy
- Multiply by step size ds
- Add to appropriate moment (A0, A1, or A2)

*Accuracy:* With small step sizes (0.05-1 mm), this is very accurate!
]

=== Three-Component Lateral Spread

#tip-box[
*Total Lateral Spread = Three Components:*

The final beam width comes from three independent sources:

1. *Initial beam spread* ($ sigma_"x0" $): The beam starts with some width
2. *Geometric spread* ($ sigma_"theta0" $ × z): Initial divergence magnifies with distance
3. *MCS scatter* ($ sigma_"mcs" $): Scattering during transport

These add IN QUADRATURE (variances add, not sigmas):

$ sigma_("total")^2 = "sigma_x0"^2 + ("sigma_theta0" times z)^2 + "sigma_mcs"^2 $
]

```cpp
float total_lateral_sigma_squared(
    float sigma_x0,      // Initial beam width
    float sigma_theta0,  // Initial angular spread
    float z,             // Depth
    float sigma_mcs      // MCS contribution
) {
    // Initial beam spread (diverges with distance)
    float sigma_initial = sigma_x0;

    // Geometric spread from initial divergence
    float sigma_geometric = sigma_theta0 * z;

    // MCS contribution (from Fermi-Eyges)
    float sigma_scattering = sigma_mcs;

    // Total variance (in quadrature)
    return sqrt(sigma_initial*sigma_initial +
                sigma_geometric*sigma_geometric +
                sigma_scattering*sigma_scattering);
}
```

#warning-box[
*CRITICAL: Add Variances, Not Sigmas!*

*WRONG:* $sigma_("total") = "sigma_x0" + "sigma_theta0" z + "sigma_mcs"$

*CORRECT:* $sigma_("total")^2 = "sigma_x0"^2 + ("sigma_theta0" z)^2 + "sigma_mcs"^2$

*Why?* These are independent random processes. Their variances add, not their standard deviations!
]

---

== 6. Complete Physics Pipeline

=== Putting It All Together

#concept-box[
*The Complete Step:*

For each step of the proton's journey, we calculate ALL physics effects in sequence:

1. *Step size control* (R-based)
2. *Energy loss* (using R-LUT)
3. *Energy straggling* (Bohr/Vavilov/Landau)
4. *Energy deposition* (dose to patient)
5. *Multiple scattering* (Highland formula)
6. *Nuclear attenuation* (proton survival)
7. *Position update* (where did it go?)
8. *Boundary check* (did it leave the simulation?)

All of this happens in microseconds per step on GPU!
]

=== Complete Step Implementation

```cpp
__device__ void transport_step(
    // Input state
    float theta, float E, float x, float z, float w,
    // Grid parameters
    float dx, float dz,
    // LUT
    const RLUT& lut,
    // Output
    float& E_dep, double& E_nuc_rem, float boundary_flux[4]
) {
    // 1. Step size control (R-based)
    float ds = compute_max_step_physics(E, lut);
    ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

    // 2. Energy loss (deterministic)
    float E_out = compute_energy_after_step(E, ds, lut);

    // 3. Energy straggling (random)
    float dE_straggle = sample_energy_loss_with_straggling(E, ds, seed);
    E_out += dE_straggle;
    E_out = fmaxf(E_out, E_cutoff);

    // 4. Energy deposition (to dose grid)
    E_dep = E - E_out;

    // 5. MCS (random direction change)
    float sigma_theta = highland_sigma(E, ds, X0_water);
    float delta_theta = sample_mcs_angle(sigma_theta, seed);
    theta += delta_theta;

    // 6. Nuclear attenuation (weight reduction)
    apply_nuclear_attenuation(w, E_nuc_rem, E, ds);

    // 7. Position update (move the proton)
    x += ds * sin(theta);
    z += ds * cos(theta);

    // 8. Boundary check (track escaping protons)
    check_boundary_emission(x, z, dx, dz, boundary_flux);
}
```

=== Visual Flow Diagram

#figure(
  table(
    columns: (1fr, 3fr),
    inset: 8pt,
    table.header([*Step*], [*Operation*]),
    [INPUT], [State: $E, x, z, theta, w$],
    [1], [STEP SIZE: $"ds" = "min"(R-"based", "boundary")$],
    [2], [ENERGY LOSS: $E_"out" = E^(-1)(R(E) - "ds")$],
    [3], [STRAGGLING: $E_"out" += dot E_"straggle"$],
    [4], [DEPOSITION: $E_"dep" = E - E_"out"$],
    [5], [MCS: $theta += dot theta_"Highland"$],
    [6], [NUCLEAR: $omega *= P_"survival"$],
    [7], [POSITION: $x += "ds" sin(theta), z += "ds" cos(theta)$],
    [8], [BOUNDARY: Track escaping protons],
    [OUTPUT], [State: $E, x, z, theta, w$],
  ),
  caption: [Single Step Physics Pipeline - All eight physics operations computed sequentially],
)

---

== Physical Constants Reference

#figure(
  table(
    columns: (auto, auto, auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*Constant*], [*Value*], [*Unit*], [*Description*]),
    [$m_p$], [938.272], [MeV/c²], [Proton rest mass - used in beta, gamma calculations],
    [$m_e c^2$], [0.511], [MeV], [Electron rest energy - sets maximum energy transfer per collision],
    [$K$], [0.307], [MeV cm²/g], [Vavilov constant - used in energy straggling calculations],
    [$Z / A$ (water)], [0.555], [-], [Ratio of atomic number to mass number for water],
    [$X_0$ (water)], [360.8], [mm], [Radiation length of water - how far protons travel before significant scattering],
    [$E_("cutoff")$], [0.1], [MeV], [Energy cutoff - protons below this are terminated],
    [$rho_("water")$], [1.0], [g/cm³], [Density of water - same as soft tissue (good approximation!)],
  ),
  caption: [Physical Constants Used in SM_2D],
)

---

== Summary: Clinical Takeaways

=== Key Physics Concepts

#tip-box[
*1. Multiple Coulomb Scattering (MCS)*
- *What:* Protons scatter sideways when passing near nuclei
- *Impact:* Beam spreads laterally - need field margins
- *Dependence:* More scattering at low energy (near Bragg peak)

*2. Energy Straggling*
- *What:* Variation in energy loss between protons
- *Impact:* Bragg peak smears out - less sharp dose fall-off
- *Dependence:* Straggling increases with path length

*3. Nuclear Attenuation*
- *What:* Some protons disappear due to nuclear reactions
- *Impact:* Dose reduction (~1-3% per 10 cm) + secondary radiation
- *Dependence:* More significant at low energies

*4. Fermi-Eyges Spread*
- *What:* Cumulative lateral beam spread from all scattering
- *Impact:* Lateral penumbra - must expand beam to cover tumor
- *Dependence:* Increases with depth (especially near end of range)
]

=== Treatment Planning Considerations

#clinical-box[
*Field Margins:*

Must account for:
- *Setup uncertainty:* Patient positioning errors (typically 3-5 mm)
- *Range uncertainty:* CT calibration, stopping power errors (typically 2-3%)
- *Organ motion:* Breathing, bladder filling (site-dependent)
- *Lateral scatter:* Beam spreading (Fermi-Eyges, typically 5-10 mm at depth)

*Typical margin formula:*

$ "Margin" = 2.5 Sigma + 0.5 "cm" $

where Sigma includes all uncertainties

*Bragg Peak Position:*

Critical to get right!
- *Too shallow:* Underdose tumor, overdose healthy tissue before
- *Too deep:* Underdose tumor, overdose healthy tissue after
- *Clinical practice:* Use 3-4 Sigma range uncertainty margins
]

---

== References

1. *NIST PSTAR Database* - Stopping powers and ranges for protons in water (gold standard)
2. *PDG 2024* - Particle Data Group review of particle physics (Highland formula)
3. *ICRU Report 63* - Nuclear cross-sections for proton therapy
4. *Vavilov (1957)* - Energy straggling theory (Bohr, Vavilov, Landau regimes)
5. *Fermi-Eyges* - Multiple scattering theory and lateral spread
6. *Bethe-Bloch* - Mean energy loss formula (stopping power)
7. *ICRU Report 73* - Stopping powers for electrons and positrons
8. *Gottschalk (2012)* - On the scattering power of radiotherapy protons

---

== Glossary

#table(
  columns: (auto, 4fr),
  inset: 8pt,
  align: left,
  table.header([*Term*], [*Definition*]),
  [MCS], [Multiple Coulomb Scattering - protons bouncing off atomic nuclei],
  [Straggling], [Statistical variation in energy loss between protons],
  [CSDA], [Continuous Slowing Down Approximation - assumes smooth energy loss],
  [Bragg Peak], [Sharp dose peak at end of proton range - key advantage of protons],
  [Lateral Spread], [Beam spreading sideways perpendicular to beam direction],
  [Penumbra], [Region of dose fall-off at beam edge - related to lateral spread],
  [Range], [Distance proton travels before stopping - energy dependent],
  [Stopping Power], [Energy loss per unit distance - depends on material and energy],
  [Cross-section], [Effective target area for nuclear interactions],
  [Variance], [Measure of spread (sigma²) - variances add for independent processes],
  [LET], [Linear Energy Transfer - energy deposited per unit distance],
)

---
#align(center)[
  *SM_2D Physics Models: Enhanced Documentation*

  #text(size: 9pt)[Version 2.0.0 - Beginner-Friendly Edition]

  *For questions or feedback, please refer to the main documentation.*
]
