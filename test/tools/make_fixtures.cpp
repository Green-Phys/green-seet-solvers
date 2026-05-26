//
// One-shot fixture generator for the green-seet-solvers test suite.
// Writes input.h5 (SIAM model data) and freq.h5 (sparse Matsubara index list)
// in the layout the alpscore-free Anderson.cpp expects.
//
// Build only when EDLIB_BUILD_FIXTURE_TOOLS=ON. Re-run when fixture schema
// changes; commit the regenerated HDF5 files under test/fixtures/.
//
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

namespace {

  // SIAM problem dimensions
  constexpr int Ns = 2;   // total sites (1 impurity + 1 bath)
  constexpr int ml = 1;   // impurity orbitals
  constexpr int Nk = Ns - ml;
  constexpr int ms = 2;   // spins

  // SIAM physical parameters
  constexpr double U   = 4.0;
  constexpr double V   = 1.0;
  constexpr double Eps = 0.0;
  constexpr double H0d = 0.0;
  constexpr double mu  = U * 0.5;   // half filling

  /// Create every parent group in `path` (everything before the trailing
  /// component), no-op if a group already exists. Mirrors green-h5pp's
  /// internal `create_parents`.
  void ensure_parents(hid_t file, const std::string& path) {
    std::string accumulated;
    std::size_t start = 0;
    while (true) {
      std::size_t pos = path.find('/', start);
      if (pos == std::string::npos) return;   // last component is the leaf
      if (pos != start) {
        accumulated += "/" + path.substr(start, pos - start);
        if (H5Lexists(file, accumulated.c_str(), H5P_DEFAULT) <= 0) {
          hid_t g = H5Gcreate2(file, accumulated.c_str(), H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
          if (g < 0) throw std::runtime_error("group create failed: " + accumulated);
          H5Gclose(g);
        }
      }
      start = pos + 1;
    }
  }

  void write_dataset(hid_t file, const std::string& path,
                     const std::vector<hsize_t>& dims,
                     hid_t type, const void* buf) {
    ensure_parents(file, path);
    hid_t space = dims.empty()
        ? H5Screate(H5S_SCALAR)
        : H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t d = H5Dcreate2(file, path.c_str(), type, space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (d < 0) throw std::runtime_error("dataset create failed: " + path);
    H5Dwrite(d, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(d);
    H5Sclose(space);
  }

  void write_double_nd(hid_t file, const std::string& path,
                       std::initializer_list<hsize_t> dims,
                       const std::vector<double>& flat) {
    std::vector<hsize_t> d(dims);
    hsize_t total = 1;
    for (auto v : d) total *= v;
    if (flat.size() != total) {
      throw std::runtime_error("flat size mismatch for " + path);
    }
    write_dataset(file, path, d, H5T_NATIVE_DOUBLE, flat.data());
  }

  void write_double_scalar(hid_t file, const std::string& path, double v) {
    write_dataset(file, path, {}, H5T_NATIVE_DOUBLE, &v);
  }

  void write_int_1d(hid_t file, const std::string& path,
                    const std::vector<int>& vals) {
    std::vector<hsize_t> d{static_cast<hsize_t>(vals.size())};
    write_dataset(file, path, d, H5T_NATIVE_INT, vals.data());
  }

  void write_input(const std::string& path) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("cannot open " + path + " for write");

    // Epsk: shape [Nk][nspins]
    write_double_nd(file, "/ModelData/Epsk/values",
                    {static_cast<hsize_t>(Nk), static_cast<hsize_t>(ms)},
                    std::vector<double>(Nk * ms, Eps));

    // Vk_<im>: shape [Nk][nspins]
    write_double_nd(file, "/ModelData/Vk_0/values",
                    {static_cast<hsize_t>(Nk), static_cast<hsize_t>(ms)},
                    std::vector<double>(Nk * ms, V));

    // H0_<im>: shape [ml][nspins]
    write_double_nd(file, "/H0_0/values",
                    {static_cast<hsize_t>(ml), static_cast<hsize_t>(ms)},
                    std::vector<double>(ml * ms, H0d));

    // mu scalar
    write_double_scalar(file, "/mu", mu);

    // U tensor: shape [nspins][nspins][ml][ml][ml][ml].
    // Density-density only: U(s,s',i,i,i,i) = U if s != s' else 0.
    const std::size_t total = static_cast<std::size_t>(ms) * ms * ml * ml * ml * ml;
    std::vector<double> U_flat(total, 0.0);
    for (int s1 = 0; s1 < ms; ++s1) {
      for (int s2 = 0; s2 < ms; ++s2) {
        if (s1 == s2) continue;
        // i=j=k=l=0 (ml == 1)
        const std::size_t off =
            ((((static_cast<std::size_t>(s1) * ms + s2) * ml + 0) * ml + 0) * ml + 0) * ml + 0;
        U_flat[off] = U;
      }
    }
    write_double_nd(file, "/interaction/values",
                    {static_cast<hsize_t>(ms), static_cast<hsize_t>(ms),
                     static_cast<hsize_t>(ml), static_cast<hsize_t>(ml),
                     static_cast<hsize_t>(ml), static_cast<hsize_t>(ml)},
                    U_flat);

    H5Fclose(file);
  }

  void write_freq(const std::string& path) {
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("cannot open " + path + " for write");
    std::vector<int> ns;
    for (int n = 0; n < 8; ++n) ns.push_back(n);
    write_int_1d(file, "/fermi/wsample", ns);
    H5Fclose(file);
  }

}  // namespace

int main(int argc, char** argv) {
  std::string outdir = ".";
  if (argc >= 2) outdir = argv[1];

  try {
    write_input(outdir + "/input.h5");
    write_freq (outdir + "/freq.h5");
    std::cout << "wrote " << outdir << "/input.h5 and " << outdir << "/freq.h5" << std::endl;
  } catch (std::exception& e) {
    std::cerr << "fixture generation failed: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
