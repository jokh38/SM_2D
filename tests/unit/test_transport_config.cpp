#include <gtest/gtest.h>
#include "core/config_loader.hpp"
#include "gpu/gpu_transport_runner.hpp"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

TEST(TransportConfigTest, DefaultConfigIsValid) {
    sm_2d::IncidentParticleConfig cfg;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(TransportConfigTest, LoadTransportSectionFromIni) {
    const fs::path tmp_path = fs::temp_directory_path() / "sm2d_transport_config_test.ini";
    {
        std::ofstream out(tmp_path);
        ASSERT_TRUE(out.is_open());
        out << "[grid]\n";
        out << "Nx = 16\n";
        out << "Nz = 32\n";
        out << "dx_mm = 1.0\n";
        out << "dz_mm = 1.0\n";
        out << "max_steps = 123\n\n";
        out << "[transport]\n";
        out << "N_theta = 48\n";
        out << "N_theta_local = 4\n";
        out << "N_E_local = 2\n";
        out << "E_fine_on_MeV = 42.5\n";
        out << "E_fine_off_MeV = 43.5\n";
        out << "weight_active_min = 1e-10\n";
        out << "E_coarse_max_MeV = 260.0\n";
        out << "step_coarse_mm = 3.5\n";
        out << "n_steps_per_cell = 2\n";
        out << "fine_batch_max_cells = 2048\n";
        out << "fine_halo_cells = 2\n";
        out << "preflight_vram_margin = 0.80\n";
        out << "max_iterations = 321\n";
        out << "log_level = 2\n";
        out << "energy_groups = 0.1:2.0:0.1,2.0:30.0:0.5\n";
    }

    sm_2d::IncidentParticleConfig cfg;
    ASSERT_NO_THROW(cfg = sm_2d::load_incident_particle_config(tmp_path.string()));
    EXPECT_EQ(cfg.transport.N_theta, 48);
    EXPECT_EQ(cfg.transport.N_theta_local, 4);
    EXPECT_EQ(cfg.transport.N_E_local, 2);
    EXPECT_FLOAT_EQ(cfg.transport.E_fine_on, 42.5f);
    EXPECT_FLOAT_EQ(cfg.transport.E_fine_off, 43.5f);
    EXPECT_FLOAT_EQ(cfg.transport.E_trigger, 42.5f);
    EXPECT_FLOAT_EQ(cfg.transport.weight_active_min, 1e-10f);
    EXPECT_FLOAT_EQ(cfg.transport.E_coarse_max, 260.0f);
    EXPECT_FLOAT_EQ(cfg.transport.step_coarse, 3.5f);
    EXPECT_EQ(cfg.transport.n_steps_per_cell, 2);
    EXPECT_EQ(cfg.transport.fine_batch_max_cells, 2048);
    EXPECT_EQ(cfg.transport.fine_halo_cells, 2);
    EXPECT_FLOAT_EQ(cfg.transport.preflight_vram_margin, 0.80f);
    EXPECT_EQ(cfg.transport.max_iterations, 321);
    EXPECT_EQ(cfg.transport.log_level, 2);
    ASSERT_EQ(cfg.transport.energy_groups.size(), 2u);
    EXPECT_FLOAT_EQ(cfg.transport.energy_groups[0].E_min_MeV, 0.1f);
    EXPECT_FLOAT_EQ(cfg.transport.energy_groups[0].E_max_MeV, 2.0f);
    EXPECT_FLOAT_EQ(cfg.transport.energy_groups[0].dE_MeV, 0.1f);
    EXPECT_FLOAT_EQ(cfg.transport.energy_groups[1].E_min_MeV, 2.0f);
    EXPECT_FLOAT_EQ(cfg.transport.energy_groups[1].E_max_MeV, 30.0f);
    EXPECT_FLOAT_EQ(cfg.transport.energy_groups[1].dE_MeV, 0.5f);

    fs::remove(tmp_path);
}

TEST(TransportConfigTest, LegacyETriggerPopulatesFineOn) {
    const fs::path tmp_path = fs::temp_directory_path() / "sm2d_transport_config_legacy_trigger.ini";
    {
        std::ofstream out(tmp_path);
        ASSERT_TRUE(out.is_open());
        out << "[transport]\n";
        out << "E_trigger_MeV = 15.0\n";
    }

    sm_2d::IncidentParticleConfig cfg;
    ASSERT_NO_THROW(cfg = sm_2d::load_incident_particle_config(tmp_path.string()));
    EXPECT_FLOAT_EQ(cfg.transport.E_trigger, 15.0f);
    EXPECT_FLOAT_EQ(cfg.transport.E_fine_on, 15.0f);
    EXPECT_FLOAT_EQ(cfg.transport.E_fine_off, 15.0f);

    fs::remove(tmp_path);
}

TEST(TransportConfigTest, RunnerRejectsLocalBinMismatchBeforeGpuCheck) {
    sm_2d::IncidentParticleConfig cfg;
    cfg.transport.N_theta_local = 8;  // compile-time value is 4
    cfg.transport.N_E_local = 2;

    EXPECT_THROW(
        (void)sm_2d::GPUTransportRunner::run(cfg),
        std::invalid_argument
    );
}
