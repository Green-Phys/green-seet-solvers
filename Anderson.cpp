//
// Created by iskakoff on 19/07/16.
//
#include <iostream>
#include <chrono>
#include <thread>

#include <edlib/EDParams.h>
#include <cstdlib>
#include "edlib/Hamiltonian.h"
#include "edlib/SzSymmetry.h"
#include "edlib/SOCRSStorage.h"
#include "edlib/CRSStorage.h"
#include "edlib/HubbardModel.h"
#include "edlib/GreensFunction.h"
#include "edlib/ChiLoc.h"
#include "edlib/HDF5Utils.h"
#include "edlib/SpinResolvedStorage.h"
#include "edlib/StaticObservables.h"
#include "edlib/MeshFactory.h"
#include "sparse_matsubara.h"

#include <Eigen/Dense>

#include <alps/numeric/tensors.hpp>

void define_params(alps::params &params) {
  params.define < int >("single.NEV", 1, "Number of eigenvalues to find for single EV calculations.");
  params.define < int >("single.NCV", 7, "Number of convergent values for single EV calculations.");
}

template <typename prec>
using MMatrixX = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using MMatrixXcd = MMatrixX<std::complex<double>>;
template <typename prec>
using MatrixX = Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXcd = MatrixX<std::complex<double>>;
template <typename prec>
using VectorX = Eigen::Vector<prec, Eigen::Dynamic>;
using VectorXcd = VectorX<std::complex<double>>;

int main(int argc, const char ** argv) {
  std::this_thread::sleep_for (std::chrono::seconds(2));
// Init MPI if enabled
#ifdef USE_MPI
  MPI_Init(&argc, (char ***) &argv);
  alps::mpi::communicator comm;
#endif
// Define and read model parameters
  alps::params params(argc, argv);
  EDLib::define_parameters(params);
  define_params(params);
  params.define<bool>("COMPUTE_SIGMA", true, "Compute Self-energy.");
  params.define<bool>("REAL_FREQ", false, "Compute on real frequency grid.");
  params.define<std::string>("FREQ_FILE", "1e5_112.hdf5", "File with frequency grid.");
  params.define<std::string>("FREQ_PATH", "fermi/wsample", "File with frequency grid.");
  if(params.help_requested(std::cout)) {
    std::exit(0);
  }
  // open output file
  alps::hdf5::archive ar;
#ifdef USE_MPI
  if(!comm.rank())
#endif
    ar.open(params["OUTPUT_FILE"].as<std::string>(), "w");
  std::cout<<params<<std::endl;
  std::system(("sync -f " + params["INPUT_FILE"].as<std::string>()).c_str());
  {
  //  ar.open(params["INPUT_FILE"].as<std::string>(), "r");
  //  ar.close();
  }
  std::cout<<"Let's start"<<std::endl;
  // Start calculations
  try {
    // Construct Hamiltonian object
    typedef EDLib::SRSSIAMHamiltonian HType;
#ifdef USE_MPI
    HType ham(params, comm);
#else
    HType ham(params);
#endif
    // Diagonalize Hamiltonian
    if (!params["arpack.SECTOR"].as<bool>())
    {
      alps::params params2 = params;
      params2["arpack.NEV"] = 1;
      params2["arpack.NCV"] = params["single.NCV"];
      params2["storage.EIGENVALUES_ONLY"] = 1;
      params2["arpack.SECTOR"] = false;
      params["arpack.SECTOR"] = true;
#ifdef USE_MPI
      HType ham2(params2, comm);
#else
      HType ham2(params2);
#endif
      ham2.diag();
      std::queue<EDLib::Symmetry::SzSymmetry::Sector> sectors;
      std::vector<std::vector<int> > sectors_list;
      
      EDLib::Combination _comb(params["NSITES"]);
      const EDLib::EigenPair<double, EDLib::Symmetry::SzSymmetry::Sector> &gs = *ham2.eigenpairs().begin();
      for (auto kkk = ham2.eigenpairs().begin(); kkk != ham2.eigenpairs().end(); kkk++) {
        const EDLib::EigenPair<double, EDLib::Symmetry::SzSymmetry::Sector> &eigenpair = *kkk;
        if(std::exp(-(eigenpair.eigenvalue() - gs.eigenvalue()) * params["lanc.BETA"].as<double>()) > params["lanc.BOLTZMANN_CUTOFF"].as<double>() ) {
          sectors.push(eigenpair.sector());
        }
      }
      ham.model().symmetry().sectors() = sectors;
#ifdef USE_MPI
      MPI_Barrier(comm);
#endif
    }
    ham.diag();
    // Save eigenvalues to HDF5 file
    EDLib::hdf5::save_eigen_pairs(ham, ar, "results");
    // Construct Green's function object
    EDLib::gf::GreensFunction < HType, seet::SparseMeshFactory, alps::gf::statistics::statistics_type> greensFunction(params, ham,alps::gf::statistics::statistics_type::FERMIONIC);
    // Compute and save Green's function
    greensFunction.compute();
    greensFunction.save(ar, "results");

    if (params["REAL_FREQ"].as<bool>() ) {
      EDLib::gf::GreensFunction < HType, EDLib::RealFreqMeshFactory> greensFunction_r(params, ham);
      greensFunction_r.compute();
      greensFunction_r.save(ar, "results");
    }
    if (params["COMPUTE_SIGMA"].as<bool>() )     {

      auto G_ij = greensFunction.G_ij().data();
      auto wn   = greensFunction.G_ij().mesh1().points();
      alps::hdf5::archive ar2(params["INPUT_FILE"].as<std::string>(), "r");
      alps::numerics::tensor<double, 4> G0_ij;
      ar2["G0_imp/data"] >> G0_ij;
      alps::numerics::tensor<double, 4> Sigma_ij(G0_ij.shape());
      size_t nw = G0_ij.shape()[0];
      size_t ns = G0_ij.shape()[1];
      size_t nio = G0_ij.shape()[2];
      for(size_t iw = 0; iw < nw; ++iw) {
        for(size_t is = 0; is < ns; ++is) {
          MatrixXcd G0(nio, nio);
          MatrixXcd G(nio, nio);
          MatrixXcd Sigma(nio, nio);
          for(size_t io = 0; io < nio ; ++io) {
            for(size_t jo = 0, jor = 0; jo < nio ; ++jo, jor += 2) {
              G0(io, jo) = std::complex<double>(G0_ij(iw, is, io, jor), G0_ij(iw, is, io, jor + 1));
              G(io, jo) = G_ij(iw, io* nio + jo, is);
            }
          }
          Sigma = G0.inverse().eval() - G.inverse().eval();
          for(size_t io = 0; io < nio ; ++io) {
            for(size_t jo = 0, jor = 0; jo < nio; ++jo, jor += 2) {
              Sigma_ij(iw, is, io, jor) = Sigma(io, jo).real();
              Sigma_ij(iw, is, io, jor + 1) = Sigma(io, jo).imag();
            }
          }
        }
      }
      alps::numerics::tensor<double, 3> Sigma_inf_ij(ns, nio, nio);
      MatrixXcd A(3,3);
      MatrixXcd B(3,1);
      std::complex<double> iwn (0.0, -1./wn[nw-1]);
      std::complex<double> iwn1(0.0, -1./wn[nw-2]);
      std::complex<double> iwn2(0.0, -1./wn[nw-3]);
      for(size_t is = 0; is < ns; ++is) {
        for(size_t io = 0; io < nio ; ++io) {
          for(size_t jo = 0, jor = 0; jo < nio; ++jo, jor += 2) {
            A << 1.0, iwn, iwn*iwn,  1.0, iwn1, iwn1*iwn1,  1.0, iwn2, iwn2*iwn2;
            B(0,0) = std::complex<double>(Sigma_ij(nw-1, is, io, jor), Sigma_ij(nw-1, is, io, jor + 1));
            B(1,0) = std::complex<double>(Sigma_ij(nw-2, is, io, jor), Sigma_ij(nw-2, is, io, jor + 1));
            B(2,0) = std::complex<double>(Sigma_ij(nw-3, is, io, jor), Sigma_ij(nw-3, is, io, jor + 1));
            MatrixXcd X = A.colPivHouseholderQr().solve(B).eval();
            Sigma_inf_ij(is, io, jo) = X(0,0).real();
            for(size_t iw = 0; iw < nw; ++iw) {
              Sigma_ij(iw, is, io, jor) -= Sigma_inf_ij(is, io, jo);
            }
          }
        }
      }
      ar["results/Sigma_ij"]<<Sigma_ij;
      ar["results/Sigma_inf_ij"]<<Sigma_inf_ij;
    }
    // Init two particle Green's function object
    //EDLib::gf::ChiLoc<HType, alps::gf::real_frequency_mesh> susc(params, ham);
    // Compute and save spin susceptibility
    //susc.compute();
    //susc.save(ar, "results");
    // Compute and save charge susceptibility
    //susc.compute<EDLib::gf::NOperator<double> >();
    //susc.save(ar, "results");
  } catch (std::exception & e) {
#ifdef USE_MPI
    if(comm.rank() == 0) {
      std::cerr<<e.what();
      ar.close();
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
#else
    std::cerr<<e.what();
#endif
  }
#ifdef USE_MPI
  if(!comm.rank())
#endif
  ar.close();

  
#ifdef USE_MPI
  MPI_Finalize();
#endif
  std::system(("sync -f " + params["OUTPUT_FILE"].as<std::string>()).c_str());
  std::this_thread::sleep_for (std::chrono::seconds(2));
  return 0;
}
