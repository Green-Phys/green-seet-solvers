//
// SEET ED impurity solver — command-line entry point. All real work lives
// in seet_solver.{h,cpp}; this file only parses argv, drives MPI, and
// flushes the input/output files around the call.
//
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include <green/params/params.h>

#include "seet_solver.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
  std::this_thread::sleep_for(std::chrono::seconds(2));

#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  green::params::params p("SEET ED impurity solver.");
  seet::define_parameters(p);
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

  const int rc = seet::run(p);

#ifdef USE_MPI
  MPI_Finalize();
#endif
  std::system(("sync -f " + p["OUTPUT_FILE"].as<std::string>()).c_str());
  std::this_thread::sleep_for(std::chrono::seconds(2));
  return rc;
}
