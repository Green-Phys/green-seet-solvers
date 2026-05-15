#ifndef SEET_SPARSE_MATSUBARA_H
#define SEET_SPARSE_MATSUBARA_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <green/h5pp/archive.h>
#include <green/params/params.h>

#include <edlib/Mesh.h>

namespace seet {

  /**
   * Matsubara-style mesh whose frequency grid is a caller-supplied sparse list
   * of Matsubara indices (read from an external HDF5 dataset). Carries no
   * ALPSCore types — uses edlib::Statistics and is consumed by the
   * EDLib::core templates that take a mesh instance directly.
   */
  class sparse_matsubara_mesh {
  public:
    sparse_matsubara_mesh() = default;

    sparse_matsubara_mesh(double beta, std::vector<double> points,
                          edlib::Statistics stat = edlib::Statistics::Fermionic)
        : _beta(beta), _points(std::move(points)), _stat(stat) {
      if (_beta <= 0.0) {
        throw std::invalid_argument("sparse_matsubara_mesh: beta must be positive");
      }
      if (_points.empty()) {
        throw std::invalid_argument("sparse_matsubara_mesh: points must be non-empty");
      }
    }

    double                     beta()       const { return _beta; }
    int                        extent()     const { return static_cast<int>(_points.size()); }
    edlib::Statistics          statistics() const { return _stat; }
    const std::vector<double>& points()     const { return _points; }

  private:
    double                _beta = 0.0;
    std::vector<double>   _points;
    edlib::Statistics     _stat = edlib::Statistics::Fermionic;
  };

  /**
   * Build a sparse_matsubara_mesh from green-params + green-h5pp. The HDF5
   * file referenced by FREQ_FILE stores a 1-D dataset of Matsubara indices at
   * FREQ_PATH; we convert them to angular frequencies for beta from lanc.BETA.
   */
  class SparseMeshFactory {
  public:
    using MeshType = sparse_matsubara_mesh;

    static MeshType createMesh(green::params::params& p, edlib::Statistics stat) {
      const std::string file = p["FREQ_FILE"].as<std::string>();
      const std::string path = p["FREQ_PATH"].as<std::string>();
      const double      beta = p["lanc.BETA"].as<double>();

      std::vector<int> ns;
      green::h5pp::archive ar(file, "r");
      ar[path] >> ns;
      ar.close();

      const int shift = (stat == edlib::Statistics::Fermionic) ? 1 : 0;
      std::vector<double> omegas;
      omegas.reserve(ns.size());
      for (int n : ns) {
        omegas.push_back((2 * n + shift) * M_PI / beta);
      }
      return sparse_matsubara_mesh(beta, std::move(omegas), stat);
    }
  };

}  // namespace seet

#endif  // SEET_SPARSE_MATSUBARA_H
