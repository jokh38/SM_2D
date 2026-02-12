# Phase 6: Conservation Audit and Verification

**Status**: Pending
**Duration**: 2 days
**Dependencies**: Phase 4 (Kernels), Phase 5 (Sources/Boundaries)

---

## Objectives

1. Implement per-cell weight audit
2. Implement per-cell energy audit
3. Implement global budget tracking
4. Implement audit logging and reporting
5. Validate conservation tolerances

---

## Conservation Equations

### Weight Conservation per Cell

```
W_in = W_out + W_cutoff + W_nuclear
```

Where:
- `W_in`: Sum of all input weights in PsiC_in
- `W_out`: Sum of all output weights in PsiC_out
- `W_cutoff`: Weight absorbed at E_cutoff
- `W_nuclear`: Weight removed by nuclear interactions

### Energy Conservation per Cell

```
E_in = E_out + E_dep + E_nuclear
```

Where:
- `E_in`: Sum of (w * E_rep) in PsiC_in
- `E_out`: Sum of (w * E_rep) in PsiC_out
- `E_dep`: Energy deposited locally (EdepC)
- `E_nuclear`: Energy removed by nuclear interactions

### Global Budget

```
W_total_in = W_total_out + W_total_cutoff + W_total_nuclear + W_boundary

E_total_in = E_total_dep + E_total_nuclear + E_boundary
```

---

## TDD Cycle 6.1: Per-Cell Weight Audit

### RED - Write Tests First

Create `tests/audit/test_cell_weight_audit.cpp`:

```cpp
#include <gtest/gtest.h>
#include "audit/conservation.hpp"

TEST(CellWeightAudit, SingleComponentTransport) {
    // Create test scenario
    PsiC psi_in(1, 1, 32);
    PsiC psi_out(1, 1, 32);

    // Inject single component
    uint32_t bid = encode_block(20, 40);
    int slot = psi_in.find_or_allocate_slot(0, bid);
    psi_in.set_weight(0, slot, 0, 1.0f);

    // Simulate transport: some weight deposited, some remains
    float W_in = 1.0f;
    float W_cutoff = 0.2f;
    float W_nuclear = 0.1f;
    float W_out = 0.7f;

    CellWeightAudit audit;
    audit.W_in = W_in;
    audit.W_out = W_out;
    audit.W_cutoff = W_cutoff;
    audit.W_nuclear = W_nuclear;

    bool pass = check_weight_conservation(audit);

    EXPECT_TRUE(pass);
    EXPECT_LT(audit.W_error, 1e-6f);
}

TEST(CellWeightAudit, WeightConservationFail) {
    CellWeightAudit audit;
    audit.W_in = 1.0f;
    audit.W_out = 0.8f;  // Error: 0.1 missing
    audit.W_cutoff = 0.2f;
    audit.W_nuclear = 0.1f;

    bool pass = check_weight_conservation(audit);

    EXPECT_FALSE(pass);
    EXPECT_GT(audit.W_error, 1e-4f);
}

TEST(CellWeightAudit, ZeroWeightTolerance) {
    CellWeightAudit audit;
    audit.W_in = 0.0f;
    audit.W_out = 0.0f;
    audit.W_cutoff = 0.0f;
    audit.W_nuclear = 0.0f;

    bool pass = check_weight_conservation(audit);

    // Should pass with zero denominator handled
    EXPECT_TRUE(pass);
}

TEST(CellWeightAudit, SmallWeight) {
    CellWeightAudit audit;
    float w = 1e-10f;
    audit.W_in = w;
    audit.W_out = 0.7f * w;
    audit.W_cutoff = 0.2f * w;
    audit.W_nuclear = 0.1f * w;

    bool pass = check_weight_conservation(audit);

    EXPECT_TRUE(pass);
}
```

### GREEN - Implementation

Create `include/audit/conservation.hpp`:

```cpp
#pragma once

#include <cstdint>

struct CellWeightAudit {
    float W_in = 0;
    float W_out = 0;
    float W_cutoff = 0;
    float W_nuclear = 0;
    float W_error = 0;
};

struct CellEnergyAudit {
    double E_in = 0;
    double E_out = 0;
    double E_dep = 0;
    double E_nuclear = 0;
    double E_error = 0;
};

// Check weight conservation (tolerance: 1e-6 relative)
bool check_weight_conservation(CellWeightAudit& audit);

// Check energy conservation (tolerance: 1e-5 relative)
bool check_energy_conservation(CellEnergyAudit& audit);

// Compute weight error (returns relative error)
float compute_weight_error(const CellWeightAudit& audit);

// Compute energy error (returns relative error)
double compute_energy_error(const CellEnergyAudit& audit);
```

Create `src/audit/conservation.cpp`:

```cpp
#include "audit/conservation.hpp"
#include <cmath>

bool check_weight_conservation(CellWeightAudit& audit) {
    float W_expected = audit.W_out + audit.W_cutoff + audit.W_nuclear;
    float W_diff = fabsf(audit.W_in - W_expected);
    float W_rel = W_diff / fmaxf(audit.W_in, 1e-20f);

    audit.W_error = W_rel;

    return W_rel < 1e-6f;
}

float compute_weight_error(const CellWeightAudit& audit) {
    float W_expected = audit.W_out + audit.W_cutoff + audit.W_nuclear;
    float W_diff = fabsf(audit.W_in - W_expected);
    return W_diff / fmaxf(audit.W_in, 1e-20f);
}

bool check_energy_conservation(CellEnergyAudit& audit) {
    double E_expected = audit.E_out + audit.E_dep + audit.E_nuclear;
    double E_diff = fabs(audit.E_in - E_expected);
    double E_rel = E_diff / fmax(audit.E_in, 1e-20);

    audit.E_error = E_rel;

    return E_rel < 1e-5;
}

double compute_energy_error(const CellEnergyAudit& audit) {
    double E_expected = audit.E_out + audit.E_dep + audit.E_nuclear;
    double E_diff = fabs(audit.E_in - E_expected);
    return E_diff / fmax(audit.E_in, 1e-20);
}
```

---

## TDD Cycle 6.2: Per-Cell Energy Audit

### RED - Write Tests First

Create `tests/audit/test_cell_energy_audit.cpp`:

```cpp
#include <gtest/gtest.h>
#include "audit/conservation.hpp"

TEST(CellEnergyAudit, SingleComponentTransport) {
    CellEnergyAudit audit;

    // Component at 100 MeV
    double E_in = 100.0;

    // After transport:
    // 70 MeV remains in particle
    // 25 MeV deposited locally
    // 5 MeV removed by nuclear (with weight 0.05)
    audit.E_in = E_in;
    audit.E_out = 70.0;
    audit.E_dep = 25.0;
    audit.E_nuclear = 5.0;

    bool pass = check_energy_conservation(audit);

    EXPECT_TRUE(pass);
    EXPECT_LT(audit.E_error, 1e-5);
}

TEST(CellEnergyAudit, NuclearEnergyTracked) {
    // Simulate nuclear attenuation
    float w_removed = 0.01f;
    float E_particle = 100.0f;

    double E_nuclear = w_removed * E_particle;

    CellEnergyAudit audit;
    audit.E_in = 100.0;
    audit.E_out = 99.0;    // Remaining energy
    audit.E_dep = 0.5;     // Local deposition
    audit.E_nuclear = E_nuclear;  // Nuclear removal

    // Note: E_in = E_out + E_dep is not enough with nuclear
    bool pass = check_energy_conservation(audit);

    // Should account for nuclear energy
    EXPECT_TRUE(pass);
}

TEST(CellEnergyAudit, EnergyDriftDetection) {
    CellEnergyAudit audit;

    audit.E_in = 100.0;
    audit.E_out = 75.0;  // Too high - energy not conserved
    audit.E_dep = 20.0;
    audit.E_nuclear = 4.0;

    bool pass = check_energy_conservation(audit);

    EXPECT_FALSE(pass);
    EXPECT_GT(audit.E_error, 1e-5);
}
```

---

## TDD Cycle 6.3: Global Budget

### RED - Write Tests First

Create `tests/audit/test_global_budget.cpp`:

```cpp
#include <gtest/gtest.h>
#include "audit/global_budget.hpp"

TEST(GlobalBudget, WeightCloses) {
    GlobalAudit audit;

    audit.W_total_in = 1.0f;
    audit.W_total_out = 0.6f;
    audit.W_total_cutoff = 0.2f;
    audit.W_total_nuclear = 0.1f;
    audit.W_boundary = 0.1f;

    bool pass = check_global_weight_conservation(audit);

    EXPECT_TRUE(pass);
}

TEST(GlobalBudget, WeightLeakDetection) {
    GlobalAudit audit;

    audit.W_total_in = 1.0f;
    audit.W_total_out = 0.6f;
    audit.W_total_cutoff = 0.2f;
    audit.W_total_nuclear = 0.1f;
    audit.W_boundary = 0.05f;  // Missing 0.05

    bool pass = check_global_weight_conservation(audit);

    EXPECT_FALSE(pass);
}

TEST(GlobalBudget, EnergyCloses) {
    GlobalAudit audit;

    audit.E_total_in = 100.0;
    audit.E_total_dep = 60.0;
    audit.E_total_nuclear = 10.0;
    audit.E_boundary = 30.0;

    bool pass = check_global_energy_conservation(audit);

    EXPECT_TRUE(pass);
}

TEST(GlobalBudget, MultiCellSum) {
    // Simulate 10 cells, each with 0.1 input weight
    const int n_cells = 10;
    std::vector<CellWeightAudit> cell_audits(n_cells);

    for (int i = 0; i < n_cells; ++i) {
        cell_audits[i].W_in = 0.1f;
        cell_audits[i].W_out = 0.07f;
        cell_audits[i].W_cutoff = 0.02f;
        cell_audits[i].W_nuclear = 0.01f;
    }

    GlobalAudit global = aggregate_cell_audits(cell_audits);

    EXPECT_NEAR(global.W_total_in, 1.0f, 1e-6f);
    EXPECT_NEAR(global.W_total_out, 0.7f, 1e-6f);
    EXPECT_NEAR(global.W_total_cutoff, 0.2f, 1e-6f);
    EXPECT_NEAR(global.W_total_nuclear, 0.1f, 1e-6f);

    bool pass = check_global_weight_conservation(global);
    EXPECT_TRUE(pass);
}
```

### GREEN - Implementation

Create `include/audit/global_budget.hpp`:

```cpp
#pragma once

#include "audit/conservation.hpp"
#include <vector>

struct GlobalAudit {
    // Weight totals
    float W_total_in = 0;
    float W_total_out = 0;
    float W_total_cutoff = 0;
    float W_total_nuclear = 0;
    float W_boundary = 0;
    float W_error = 0;

    // Energy totals
    double E_total_in = 0;
    double E_total_dep = 0;
    double E_total_nuclear = 0;
    double E_boundary = 0;
    double E_error = 0;
};

// Aggregate cell audits into global audit
GlobalAudit aggregate_cell_audits(
    const std::vector<CellWeightAudit>& weight_audits,
    const std::vector<CellEnergyAudit>& energy_audits
);

// Check global weight conservation
bool check_global_weight_conservation(GlobalAudit& audit);

// Check global energy conservation
bool check_global_energy_conservation(GlobalAudit& audit);
```

Create `src/audit/global_budget.cpp`:

```cpp
#include "audit/global_budget.hpp"
#include <cmath>

GlobalAudit aggregate_cell_audits(
    const std::vector<CellWeightAudit>& weight_audits,
    const std::vector<CellEnergyAudit>& energy_audits
) {
    GlobalAudit audit;

    for (const auto& cell : weight_audits) {
        audit.W_total_in += cell.W_in;
        audit.W_total_out += cell.W_out;
        audit.W_total_cutoff += cell.W_cutoff;
        audit.W_total_nuclear += cell.W_nuclear;
    }

    for (const auto& cell : energy_audits) {
        audit.E_total_in += cell.E_in;
        audit.E_total_dep += cell.E_dep;
        audit.E_total_nuclear += cell.E_nuclear;
    }

    return audit;
}

bool check_global_weight_conservation(GlobalAudit& audit) {
    float W_expected = audit.W_total_out + audit.W_total_cutoff +
                      audit.W_total_nuclear + audit.W_boundary;
    float W_diff = fabsf(audit.W_total_in - W_expected);
    audit.W_error = W_diff / fmaxf(audit.W_total_in, 1e-20f);

    return audit.W_error < 1e-6f;
}

bool check_global_energy_conservation(GlobalAudit& audit) {
    double E_expected = audit.E_total_dep + audit.E_total_nuclear +
                       audit.E_boundary;
    double E_diff = fabs(audit.E_total_in - E_expected);
    audit.E_error = E_diff / fmax(audit.E_total_in, 1e-20);

    return audit.E_error < 1e-5;
}
```

---

## TDD Cycle 6.4: Audit Reporting

### RED - Write Tests First

Create `tests/audit/test_reporting.cpp`:

```cpp
#include <gtest/gtest.h>
#include "audit/reporting.hpp"
#include <sstream>

TEST(AuditReport, PrintWeightReport) {
    CellWeightAudit audit;
    audit.W_in = 1.0f;
    audit.W_out = 0.7f;
    audit.W_cutoff = 0.2f;
    audit.W_nuclear = 0.1f;

    std::ostringstream oss;
    print_cell_weight_report(oss, audit, 42);

    std::string report = oss.str();

    EXPECT_NE(report.find("Cell 42"), std::string::npos);
    EXPECT_NE(report.find("W_in"), std::string::npos);
    EXPECT_NE(report.find("PASS"), std::string::npos);
}

TEST(AuditReport, PrintGlobalReport) {
    GlobalAudit audit;
    audit.W_total_in = 1.0f;
    audit.W_total_out = 0.6f;
    audit.W_total_cutoff = 0.2f;
    audit.W_total_nuclear = 0.1f;
    audit.W_boundary = 0.1f;

    std::ostringstream oss;
    print_global_report(oss, audit);

    std::string report = oss.str();

    EXPECT_NE(report.find("Weight"), std::string::npos);
    EXPECT_NE(report.find("Energy"), std::string::npos);
}

TEST(AuditReport, FailedCells) {
    std::vector<int> failed_cells = {5, 10, 15};

    std::ostringstream oss;
    print_failed_cells(oss, failed_cells);

    std::string report = oss.str();

    EXPECT_NE(report.find("5"), std::string::npos);
    EXPECT_NE(report.find("10"), std::string::npos);
    EXPECT_NE(report.find("15"), std::string::npos);
}
```

### GREEN - Implementation

Create `include/audit/reporting.hpp`:

```cpp
#pragma once

#include "audit/conservation.hpp"
#include "audit/global_budget.hpp"
#include <iosfwd>
#include <vector>

// Print cell weight audit report
void print_cell_weight_report(std::ostream& os, const CellWeightAudit& audit, int cell);

// Print global audit report
void print_global_report(std::ostream& os, const GlobalAudit& audit);

// Print list of failed cells
void print_failed_cells(std::ostream& os, const std::vector<int>& failed_cells);

// Print summary statistics
void print_summary(std::ostream& os,
                  int n_cells,
                  int n_passed,
                  int n_failed,
                  const GlobalAudit& global);
```

---

## Exit Criteria Checklist

- [ ] Per-cell weight error < 1e-6 for single step
- [ ] Per-cell energy error < 1e-5 for single step
- [ ] Global budget closes for multi-step run
- [ ] Audit report readable and informative
- [ ] Failed cells identified and reported
- [ ] Boundary losses included in global budget

---

## Next Steps

After completing Phase 6, proceed to **Phase 7 (Physics Validation)** which validates against NIST data.

```bash
# Test audit
./bin/sm2d_tests --gtest_filter="*Audit*"
```
