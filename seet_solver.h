#ifndef SEET_SOLVER_H
#define SEET_SOLVER_H

#include <green/params/params.h>

namespace seet {

  /// Register every parameter the SEET ED impurity solver reads.
  /// Parameter names match the legacy alpscore/input.h5 conventions so
  /// existing INI files keep working.
  void define_parameters(green::params::params& p);

  /// Run the full SEET ED solve described by `p`:
  ///   read model data and frequency grid → diagonalise → compute G and
  ///   optionally Sigma → write OUTPUT_FILE. MPI is NOT initialised here;
  ///   the caller is responsible for MPI_Init/Finalize.
  ///
  /// Returns 0 on success, -1 on an exception caught from the solve.
  int run(green::params::params& p);

}  // namespace seet

#endif  // SEET_SOLVER_H
