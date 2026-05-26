//
// Catch2 end-to-end test: drive seet::run() in-process against the committed
// 2-site SIAM data (U=4, V=1, mu=U/2, beta=10), then assert on the
// produced sim.h5.
//
#include <cmath>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <green/h5pp/archive.h>
#include <green/params/params.h>

#include "seet_ed_solver.h"

namespace {

  // Analytic GS in the (Nup=1, Ndown=1) sector for the symmetric 2-site SIAM:
  //   E_gs = -(U/2 + sqrt((U/2)^2 + 4 V^2))
  //        = -(2 + 2*sqrt(2))  for U=4, V=1.
  constexpr double GS_REFERENCE = -4.8284271247461903;

  constexpr int EXPECTED_NW     = 8;   // matches test/data/freq.h5
  constexpr int EXPECTED_NSITES = 1;
  constexpr int EXPECTED_NSPINS = 2;

  /// Build a parsed params object with all fixture-specific arguments
  /// applied. Doing this through `parse(argc, argv)` keeps the param
  /// machinery on its happy path (default validation, INI loading, etc.).
  green::params::params make_params() {
    green::params::params p("seet test");
    seet::define_parameters(p);

    std::vector<std::string> args = {
        "test_runner",
        "--NSITES",            "2",
        "--NSPINS",            "2",
        "--siam.NORBITALS",    "1",
        "--lanc.BETA",         "10.0",
        "--lanc.NOMEGA",       "8",
        "--lanc.NLANC",        "50",
        "--arpack.NEV",        "2",
        "--storage.MAX_DIM",   "64",
        "--storage.MAX_SIZE",  "4096",
        "--REAL_FREQ",         "0",
        "--COMPUTE_SIGMA",     "1",
        "--INPUT_FILE",        DATA_INPUT_PATH,
        "--FREQ_FILE",         DATA_FREQ_PATH,
        "--FREQ_PATH",         "fermi/wsample",
        "--OUTPUT_FILE",       SIM_H5_PATH,
    };
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& s : args) argv.push_back(const_cast<char*>(s.c_str()));
    REQUIRE(p.parse(static_cast<int>(argv.size()), argv.data()));
    return p;
  }

}  // namespace

TEST_CASE("ed_solver sim.h5 contents", "[ed_solver][integration]") {
  // Run the solver exactly once per test executable invocation, regardless
  // of how many Catch2 SECTIONs below re-enter the case body.
  static const int rc = [] {
    auto p = make_params();
    return seet::run(p);
  }();
  REQUIRE(rc == 0);

  green::h5pp::archive ar(SIM_H5_PATH, "r");

  SECTION("eigenvalues are present and consistent") {
    int n_eigen = 0;
    ar["results/eigenvalues/N"] >> n_eigen;
    REQUIRE(n_eigen > 0);

    std::vector<double> values;
    ar["results/eigenvalues/data"] >> values;
    REQUIRE(static_cast<int>(values.size()) == n_eigen);
  }

  SECTION("ground-state energy matches analytic 2-site SIAM reference") {
    std::vector<double> values;
    ar["results/eigenvalues/data"] >> values;
    REQUIRE_FALSE(values.empty());
    double gs = values[0];
    for (double v : values) gs = std::min(gs, v);
    REQUIRE_THAT(gs, Catch::Matchers::WithinAbs(GS_REFERENCE, 1e-6));
  }

  SECTION("G_omega has the expected flat size and mesh extents") {
    std::vector<double> g_flat;
    ar["results/G_omega/data"] >> g_flat;
    const std::size_t expected =
        2ULL * EXPECTED_NW * EXPECTED_NSITES * EXPECTED_NSPINS;
    REQUIRE(g_flat.size() == expected);

    int nw_in_file = 0;
    int n2_in_file = 0;
    int n3_in_file = 0;
    ar["results/G_omega/mesh/1/N"] >> nw_in_file;
    ar["results/G_omega/mesh/2/N"] >> n2_in_file;
    ar["results/G_omega/mesh/3/N"] >> n3_in_file;
    CHECK(nw_in_file == EXPECTED_NW);
    CHECK(n2_in_file == EXPECTED_NSITES);
    CHECK(n3_in_file == EXPECTED_NSPINS);
  }
}
