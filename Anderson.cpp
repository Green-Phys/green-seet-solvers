//
// SEET ED impurity solver, ALPSCore-free port: uses green-params for
// command-line parameters, green-h5pp for HDF5 I/O, and the alpscore-free
// edlib:: core for the ED solver itself.
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <green/h5pp/archive.h>
#include <green/params/params.h>

#include <edlib/Hamiltonian.h>
#include <edlib/Mesh.h>
#include <edlib/MeshFactory.h>
#include <edlib/Parameters.h>
#include <edlib/SingleImpurityAndersonModel.h>
#include <edlib/SpinResolvedStorage.h>
#include <edlib/GreensFunction.h>

#include "sparse_matsubara.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace {

  using HType = edlib::SRSSIAMHamiltonian;
  using SparseGF = edlib::GreensFunction<HType, seet::sparse_matsubara_mesh>;
  using RealGF   = edlib::GreensFunction<HType, edlib::RealFreqMesh>;
  using SiamModelData = edlib::SingleImpurityAndersonModel<double>::ModelData;

  template <typename Prec>
  using MatrixX = Eigen::Matrix<Prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcd = MatrixX<std::complex<double>>;

  /**
   * Project the alpscore parameter surface that the original Anderson.cpp
   * relied on onto green::params::params. Parameter names match the legacy
   * input.h5 / INI files so existing inputs keep working.
   */
  void define_parameters(green::params::params& p) {
    // ED-core knobs (formerly defined by EDLib::define_parameters)
    p.define<int>("NSITES",                 "Total number of orbitals (impurity + bath).", 4);
    p.define<int>("NSPINS",                 "Number of electron spins.", 2);
    p.define<std::string>("INPUT_FILE",     "HDF5 file with SIAM model data.", std::string("input.h5"));
    p.define<std::string>("OUTPUT_FILE",    "HDF5 file to write results into.", std::string("sim.h5"));

    p.define<bool>("arpack.SECTOR",         "Restrict diagonalisation to sectors listed in input.h5.", false);
    p.define<int> ("arpack.NEV",            "Number of eigenvalues per sector.", 2);
    p.define<int> ("arpack.NCV",            "Number of ARPACK Lanczos vectors (0 = solver default).", 0);

    p.define<std::size_t>("storage.MAX_DIM",  "Maximum CRS dimension.", std::size_t(5000));
    p.define<std::size_t>("storage.MAX_SIZE", "Maximum CRS non-zero count.", std::size_t(70000));
    p.define<bool>       ("storage.EIGENVALUES_ONLY", "Skip eigenvectors.", false);
    p.define<int>        ("storage.ORBITAL_NUMBER",   "Spin-storage orbital block size.", 1);

    p.define<int>   ("lanc.NOMEGA",            "Number of frequency points.", 32);
    p.define<double>("lanc.EMIN",              "Real-frequency lower bound.", -3.0);
    p.define<double>("lanc.EMAX",              "Real-frequency upper bound.",  3.0);
    p.define<int>   ("lanc.NLANC",             "Lanczos iterations for spectral functions.", 100);
    p.define<double>("lanc.BETA",              "Inverse temperature.", 10.0);
    p.define<double>("lanc.BOLTZMANN_CUTOFF",  "Boltzmann factor cutoff.", 1e-12);

    p.define<int>("siam.NORBITALS",            "Number of impurity orbitals.", 1);

    // SEET-specific knobs
    p.define<int> ("single.NEV",  "Number of eigenvalues to find for single-EV calculations.", 1);
    p.define<int> ("single.NCV",  "Number of convergent values for single-EV calculations.", 7);

    p.define<bool>("COMPUTE_SIGMA", "Compute Self-energy.", true);
    p.define<bool>("REAL_FREQ",     "Compute on real frequency grid.", false);
    p.define<std::string>("FREQ_FILE", "HDF5 file with sparse Matsubara index grid.", std::string("1e5_112.hdf5"));
    p.define<std::string>("FREQ_PATH", "Path inside FREQ_FILE to the index grid.", std::string("fermi/wsample"));
  }

  edlib::Parameters make_edlib_parameters(const green::params::params& p) {
    edlib::Parameters out;
    out.nsites                      = p["NSITES"].as<int>();
    out.nspins                      = p["NSPINS"].as<int>();
    out.arpack_sector               = p["arpack.SECTOR"].as<bool>();
    out.arpack_nev                  = p["arpack.NEV"].as<int>();
    out.arpack_ncv                  = p["arpack.NCV"].as<int>();
    out.storage_max_dim             = p["storage.MAX_DIM"].as<std::size_t>();
    out.storage_max_size            = p["storage.MAX_SIZE"].as<std::size_t>();
    out.eigenvalues_only            = p["storage.EIGENVALUES_ONLY"].as<bool>();
    out.spinstorage_orbital_number  = p["storage.ORBITAL_NUMBER"].as<int>();
    out.lanc_nomega                 = p["lanc.NOMEGA"].as<int>();
    out.lanc_emin                   = p["lanc.EMIN"].as<double>();
    out.lanc_emax                   = p["lanc.EMAX"].as<double>();
    out.lanc_nlanc                  = p["lanc.NLANC"].as<int>();
    out.lanc_beta                   = p["lanc.BETA"].as<double>();
    out.lanc_boltzmann_cutoff       = p["lanc.BOLTZMANN_CUTOFF"].as<double>();
    out.siam_norbitals              = p["siam.NORBITALS"].as<int>();
    return out;
  }

  // ------------------------------------------------------------------ I/O helpers

  std::vector<std::size_t> dataset_shape(green::h5pp::archive& ar, const std::string& path) {
    return green::h5pp::dataset_shape(ar.file_id(), path);
  }

  /// Read a flat HDF5 dataset into a nested vector<vector<T>> with the given
  /// outer/inner dims.
  template <typename T>
  std::vector<std::vector<T>> read_2d(green::h5pp::archive& ar, const std::string& path) {
    auto shape = dataset_shape(ar, path);
    if (shape.size() != 2) {
      throw std::runtime_error("Expected 2-D dataset at " + path);
    }
    std::vector<T> flat;
    ar[path] >> flat;
    if (flat.size() != shape[0] * shape[1]) {
      throw std::runtime_error("Inconsistent flat size for " + path);
    }
    std::vector<std::vector<T>> out(shape[0], std::vector<T>(shape[1]));
    for (std::size_t i = 0; i < shape[0]; ++i) {
      std::copy(flat.begin() + i * shape[1],
                flat.begin() + (i + 1) * shape[1],
                out[i].begin());
    }
    return out;
  }

  /// Read a flat 6-D HDF5 dataset into an edlib::Gf<T,6> with shape pulled from
  /// the dataset itself.
  template <typename T>
  edlib::Gf<T, 6> read_6d_into_gf(green::h5pp::archive& ar, const std::string& path) {
    auto shape = dataset_shape(ar, path);
    if (shape.size() != 6) {
      throw std::runtime_error("Expected 6-D dataset at " + path);
    }
    std::array<int, 6> ishape{};
    std::size_t total = 1;
    for (int d = 0; d < 6; ++d) {
      ishape[d] = static_cast<int>(shape[d]);
      total    *= shape[d];
    }
    std::vector<T> flat;
    ar[path] >> flat;
    if (flat.size() != total) {
      throw std::runtime_error("Inconsistent flat size for " + path);
    }
    edlib::Gf<T, 6> g(ishape);
    std::copy(flat.begin(), flat.end(), g.data().begin());
    return g;
  }

  SiamModelData read_model_data(green::params::params& p) {
    const int Ns = p["NSITES"].as<int>();
    const int ml = p["siam.NORBITALS"].as<int>();
    const int ms = p["NSPINS"].as<int>();
    if (ml > Ns) {
      throw std::invalid_argument("SIAM: siam.NORBITALS exceeds NSITES");
    }
    if (ms != 2) {
      throw std::invalid_argument("SIAM: NSPINS must be 2");
    }

    SiamModelData b;
    b.Vk.assign(ml, std::vector<std::vector<double>>());
    b.H0.assign(ml, std::vector<std::vector<double>>(ml, std::vector<double>(ms, 0.0)));

    const std::string input = p["INPUT_FILE"].as<std::string>();
    green::h5pp::archive ar(input, "r");

    b.Epsk = read_2d<double>(ar, "ModelData/Epsk/values");
    if (static_cast<int>(b.Epsk.size()) != Ns - ml) {
      throw std::invalid_argument(
          "SIAM: bath levels + impurity orbitals != NSITES");
    }

    for (int im = 0; im < ml; ++im) {
      std::stringstream s;
      s << "ModelData/Vk_" << im << "/values";
      b.Vk[im] = read_2d<double>(ar, s.str());
      s.str("");
      s << "H0_" << im << "/values";
      b.H0[im] = read_2d<double>(ar, s.str());
    }

    double mu = 0.0;
    ar["mu"] >> mu;
    b.mu = mu;

    b.U = read_6d_into_gf<double>(ar, "interaction/values");
    if (b.U.shape(2) != ml) {
      throw std::invalid_argument("SIAM: U shape mismatch");
    }

    ar.close();
    return b;
  }

  std::vector<std::array<int, 2>> read_orbital_pairs(green::params::params& p) {
    std::vector<std::array<int, 2>> out;
    const std::string input = p["INPUT_FILE"].as<std::string>();
    green::h5pp::archive ar(input, "r");
    if (ar.is_data("/GreensFunction_orbitals/values")) {
      auto shape = dataset_shape(ar, "/GreensFunction_orbitals/values");
      if (shape.size() == 2 && shape[1] == 2) {
        std::vector<int> flat;
        ar["/GreensFunction_orbitals/values"] >> flat;
        for (std::size_t i = 0; i < shape[0]; ++i) {
          out.push_back({flat[2 * i], flat[2 * i + 1]});
        }
      }
    }
    ar.close();
    return out;
  }

  void save_eigen_pairs(green::h5pp::archive& ar, const HType& ham, const std::string& path) {
    const auto& eps = ham.eigenpairs();
    ar[path + "/eigenvalues/N"] << static_cast<int>(eps.size());
    std::vector<double> values;
    int i = 0;
    for (const auto& e : eps) {
      values.push_back(e.eigenvalue());
      ar[path + "/eigenvalues/sectors/" + std::to_string(i) + "/nup"]   << e.sector().nup();
      ar[path + "/eigenvalues/sectors/" + std::to_string(i) + "/ndown"] << e.sector().ndown();
      ar[path + "/eigenvalues/sectors/" + std::to_string(i) + "/size"]  << static_cast<int>(e.sector().size());
      ++i;
    }
    ar[path + "/eigenvalues/data"] << values;
  }

  /// Save a complex-valued 3-index Gf as two real datasets (.../data and
  /// .../mesh) under `path`, mirroring the legacy alps::gf::three_index_gf
  /// serialisation closely enough for downstream consumers.
  void save_gf(green::h5pp::archive& ar, const std::string& base,
               const edlib::GF3& g,
               const std::vector<double>& mesh_points) {
    const int nw = g.shape(0);
    const int n2 = g.shape(1);
    const int n3 = g.shape(2);
    std::vector<double> flat(2 * std::size_t(nw) * n2 * n3);
    for (int iw = 0; iw < nw; ++iw) {
      for (int i = 0; i < n2; ++i) {
        for (int is = 0; is < n3; ++is) {
          const auto v = g(iw, i, is);
          const std::size_t off = ((std::size_t(iw) * n2 + i) * n3 + is) * 2;
          flat[off]     = v.real();
          flat[off + 1] = v.imag();
        }
      }
    }
    ar[base + "/data"] << flat;
    ar[base + "/mesh/1/points"] << mesh_points;
    ar[base + "/mesh/1/N"]      << nw;
    ar[base + "/mesh/2/N"]      << n2;
    ar[base + "/mesh/3/N"]      << n3;
  }

  void write_text_dump(const std::string& filename, const edlib::GF3& g,
                       const std::vector<double>& mesh_points) {
    std::ofstream f(filename);
    f << std::setprecision(14);
    const int nw = g.shape(0);
    for (int iw = 0; iw < nw; ++iw) {
      f << mesh_points[iw];
      for (int i = 0; i < g.shape(1); ++i) {
        for (int is = 0; is < g.shape(2); ++is) {
          const auto v = g(iw, i, is);
          f << " " << v.real() << " " << v.imag();
        }
      }
      f << "\n";
    }
  }

  // ------------------------------------------------------------------ sector pruning

  /**
   * Replicate the legacy "single-NEV" pass: run a cheap eigenvalue-only diag
   * across all sectors, keep only those with Boltzmann weight above the
   * cutoff, then load the surviving sectors into the main Hamiltonian's
   * symmetry queue.
   */
  void prune_sectors(green::params::params& p, HType& ham,
                     const SiamModelData& model_data) {
    edlib::Parameters p2 = make_edlib_parameters(p);
    p2.arpack_nev        = 1;
    p2.arpack_ncv        = p["single.NCV"].as<int>();
    p2.eigenvalues_only  = true;
    p2.arpack_sector     = false;

#ifdef USE_MPI
    HType ham2(p2, model_data, MPI_COMM_WORLD);
#else
    HType ham2(p2, model_data);
#endif
    ham2.diag();

    const auto& gs   = *ham2.eigenpairs().begin();
    const double beta = p["lanc.BETA"].as<double>();
    const double cutoff = p["lanc.BOLTZMANN_CUTOFF"].as<double>();
    std::queue<edlib::SzSymmetry::Sector> survivors;
    for (const auto& ep : ham2.eigenpairs()) {
      if (std::exp(-(ep.eigenvalue() - gs.eigenvalue()) * beta) > cutoff) {
        survivors.push(ep.sector());
      }
    }
    ham.model().symmetry().sectors() = survivors;
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  // ------------------------------------------------------------------ self-energy

  void compute_and_save_selfenergy(green::params::params& p,
                                   green::h5pp::archive& ar,
                                   const edlib::GF3& G_ij,
                                   const std::vector<double>& wn) {
    const std::string input = p["INPUT_FILE"].as<std::string>();
    green::h5pp::archive in(input, "r");

    auto g0_shape = dataset_shape(in, "G0_imp/data");
    if (g0_shape.size() != 4) {
      throw std::runtime_error("G0_imp/data must be 4-D [nw, ns, nio, 2*nio]");
    }
    std::vector<double> G0_flat;
    in["G0_imp/data"] >> G0_flat;
    in.close();

    const std::size_t nw  = g0_shape[0];
    const std::size_t ns  = g0_shape[1];
    const std::size_t nio = g0_shape[2];
    if (g0_shape[3] != 2 * nio) {
      throw std::runtime_error("G0_imp inner-most axis must be 2*nio (re/im)");
    }

    auto G0_at = [&](std::size_t iw, std::size_t is, std::size_t io, std::size_t jor) -> double& {
      return G0_flat[((iw * ns + is) * nio + io) * (2 * nio) + jor];
    };

    std::vector<double> Sigma_flat(G0_flat.size(), 0.0);
    auto Sigma_at = [&](std::size_t iw, std::size_t is, std::size_t io, std::size_t jor) -> double& {
      return Sigma_flat[((iw * ns + is) * nio + io) * (2 * nio) + jor];
    };

    for (std::size_t iw = 0; iw < nw; ++iw) {
      for (std::size_t is = 0; is < ns; ++is) {
        MatrixXcd G0(nio, nio);
        MatrixXcd G (nio, nio);
        for (std::size_t io = 0; io < nio; ++io) {
          for (std::size_t jo = 0, jor = 0; jo < nio; ++jo, jor += 2) {
            G0(io, jo) = std::complex<double>(G0_at(iw, is, io, jor), G0_at(iw, is, io, jor + 1));
            G (io, jo) = G_ij(static_cast<int>(iw), static_cast<int>(io * nio + jo), static_cast<int>(is));
          }
        }
        MatrixXcd Sigma = G0.inverse().eval() - G.inverse().eval();
        for (std::size_t io = 0; io < nio; ++io) {
          for (std::size_t jo = 0, jor = 0; jo < nio; ++jo, jor += 2) {
            Sigma_at(iw, is, io, jor)     = Sigma(io, jo).real();
            Sigma_at(iw, is, io, jor + 1) = Sigma(io, jo).imag();
          }
        }
      }
    }

    // Subtract the high-frequency tail by fitting a 1/(i wn) expansion on the
    // last three points (identical to the legacy logic).
    std::vector<double> Sigma_inf(ns * nio * nio, 0.0);
    auto Sinf = [&](std::size_t is, std::size_t io, std::size_t jo) -> double& {
      return Sigma_inf[(is * nio + io) * nio + jo];
    };

    MatrixXcd A(3, 3);
    MatrixXcd B(3, 1);
    const std::complex<double> iwn (0.0, -1.0 / wn[nw - 1]);
    const std::complex<double> iwn1(0.0, -1.0 / wn[nw - 2]);
    const std::complex<double> iwn2(0.0, -1.0 / wn[nw - 3]);
    for (std::size_t is = 0; is < ns; ++is) {
      for (std::size_t io = 0; io < nio; ++io) {
        for (std::size_t jo = 0, jor = 0; jo < nio; ++jo, jor += 2) {
          A << 1.0, iwn,  iwn  * iwn,
               1.0, iwn1, iwn1 * iwn1,
               1.0, iwn2, iwn2 * iwn2;
          B(0, 0) = std::complex<double>(Sigma_at(nw - 1, is, io, jor), Sigma_at(nw - 1, is, io, jor + 1));
          B(1, 0) = std::complex<double>(Sigma_at(nw - 2, is, io, jor), Sigma_at(nw - 2, is, io, jor + 1));
          B(2, 0) = std::complex<double>(Sigma_at(nw - 3, is, io, jor), Sigma_at(nw - 3, is, io, jor + 1));
          MatrixXcd X = A.colPivHouseholderQr().solve(B).eval();
          Sinf(is, io, jo) = X(0, 0).real();
          for (std::size_t iw = 0; iw < nw; ++iw) {
            Sigma_at(iw, is, io, jor) -= Sinf(is, io, jo);
          }
        }
      }
    }

    ar["results/Sigma_ij/data"] << Sigma_flat;
    ar["results/Sigma_ij/shape"] << std::vector<std::size_t>(g0_shape.begin(), g0_shape.end());
    ar["results/Sigma_inf_ij/data"]  << Sigma_inf;
    ar["results/Sigma_inf_ij/shape"] << std::vector<std::size_t>{ns, nio, nio};
  }

}  // namespace

int main(int argc, char** argv) {
  std::this_thread::sleep_for(std::chrono::seconds(2));

#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  green::params::params p("SEET ED impurity solver.");
  define_parameters(p);
  if (!p.parse(argc, argv)) {
#ifdef USE_MPI
    if (!rank) p.help();
    MPI_Finalize();
#else
    p.help();
#endif
    return 0;
  }
#ifdef USE_MPI
  if (!rank)
#endif
    p.print();

  std::system(("sync -f " + p["INPUT_FILE"].as<std::string>()).c_str());

  try {
    SiamModelData model_data = read_model_data(p);
    auto orbital_pairs       = read_orbital_pairs(p);

    edlib::Parameters edp = make_edlib_parameters(p);

#ifdef USE_MPI
    HType ham(edp, model_data, MPI_COMM_WORLD);
#else
    HType ham(edp, model_data);
#endif

    if (!edp.arpack_sector) {
      // The "active" Hamiltonian wants arpack.SECTOR = true after pruning
      edp.arpack_sector = true;
      prune_sectors(p, ham, model_data);
    }

    ham.diag();

    green::h5pp::archive out;
#ifdef USE_MPI
    if (!rank)
#endif
      out.open(p["OUTPUT_FILE"].as<std::string>(), "w");

#ifdef USE_MPI
    if (!rank) {
#endif
      save_eigen_pairs(out, ham, "results");
#ifdef USE_MPI
    }
#endif

    auto fmesh = seet::SparseMeshFactory::createMesh(p, edlib::Statistics::Fermionic);
    SparseGF greens(edp, ham, fmesh, orbital_pairs);
    greens.compute();

#ifdef USE_MPI
    if (!rank) {
#endif
      const auto mesh_points = fmesh.points();
      if (greens.G().shape(1) > 0) {
        save_gf(out, "results/G_omega", greens.G(), mesh_points);
        write_text_dump("G_omega", greens.G(), mesh_points);
      }
      if (greens.G_ij().shape(1) > 0) {
        save_gf(out, "results/G_ij_omega", greens.G_ij(), mesh_points);
        write_text_dump("G_ij_omega", greens.G_ij(), mesh_points);
      }
#ifdef USE_MPI
    }
#endif

    if (p["REAL_FREQ"].as<bool>()) {
      edlib::RealFreqMesh rmesh(edp.lanc_emin, edp.lanc_emax, edp.lanc_nomega);
      RealGF greens_r(edp, ham, rmesh, orbital_pairs);
      greens_r.compute();
#ifdef USE_MPI
      if (!rank) {
#endif
        const auto rpoints = rmesh.points();
        if (greens_r.G().shape(1) > 0) {
          save_gf(out, "results/G_omega_r", greens_r.G(), rpoints);
        }
        if (greens_r.G_ij().shape(1) > 0) {
          save_gf(out, "results/G_ij_omega_r", greens_r.G_ij(), rpoints);
        }
#ifdef USE_MPI
      }
#endif
    }

    if (p["COMPUTE_SIGMA"].as<bool>()) {
#ifdef USE_MPI
      if (!rank) {
#endif
        compute_and_save_selfenergy(p, out, greens.G_ij(), fmesh.points());
#ifdef USE_MPI
      }
#endif
    }

#ifdef USE_MPI
    if (!rank)
#endif
      out.close();
  } catch (std::exception& e) {
#ifdef USE_MPI
    if (!rank) std::cerr << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
#else
    std::cerr << e.what() << std::endl;
    return -1;
#endif
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif
  std::system(("sync -f " + p["OUTPUT_FILE"].as<std::string>()).c_str());
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return 0;
}
