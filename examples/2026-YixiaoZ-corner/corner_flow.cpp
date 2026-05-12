#include <athena/athena_arrays.hpp>
#include <athena/bvals/bvals.hpp>
#include <athena/coordinates/coordinates.hpp>
#include <athena/eos/eos.hpp>
#include <athena/field/field.hpp>
#include <athena/hydro/hydro.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/parameter_input.hpp>

// climath
#include <climath/interpolation.h>

// snap
#include <snap/thermodynamics/atm_thermodynamics.hpp>

#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "channel_utils.hpp"
#include "wall_boundary_condition.hpp"

constexpr int i_vapor = 1;
constexpr int i_solid = 2;

constexpr Real x_flow_exit = 0.;

constexpr Real max_nu = 1.;

inline Real get_mu(Real mu, Real rho) {
  return std::min(mu / rho, max_nu);
}

inline Real get_cp(const int n_species) {
  auto pthermo = Thermodynamics::GetInstance();
  return (
      pthermo->GetRd() * (
        (pthermo->GetCvRatio(n_species) / (pthermo->GetGammad() - 1.))
        + pthermo->GetInvMuRatio(n_species)
      )
  );
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  if (pthermo->SpeciesIndex("H2O") != i_vapor) {
    throw std::runtime_error("i_vapor does not match");
  }
  if (pthermo->SpeciesIndex("H2O(s)") != i_solid) {
    throw std::runtime_error("i_solid does not match");
  }

  if (pin->GetOrAddString("mesh", "ix1_bc", "none") == "user") {
    const static Real exit_delta = pin->GetReal(
      "problem", "exit_delta");
    const static Real center_velocity = pin->GetReal(
      "problem", "exit_center_velocity");
    const static Real temperature = pin->GetReal(
      "problem", "exit_temperature");

    auto _bottom_bc = [](MeshBlock *pmb, Coordinates *pco,
                      AthenaArray<Real> &prim, FaceField &b,
                      Real time, Real dt,
                      int il, int iu,
                      int jl, int ju,
                      int kl, int ku, int ngh) -> void {
      for (int k = kl; k <= ku; ++k) {
        for (int j = jl; j <= ju; ++j) {
          for (int ii = 1; ii <= ngh; ++ii) {

            const int i = il - ii;
            auto w = prim.at(k, j, i);

            auto water_ice_eos = WaterIceEOS();
            const Real pressure = water_ice_eos.pres_sat(temperature);
            const Real density = water_ice_eos.gas.density(
              temperature, pressure);
            w[IDN] = density;
            w[i_vapor] = 1.;
            w[i_solid] = 0.;
            w[IVX] = center_velocity;
            w[IVY] = 0.;
            w[IVZ] = 0.;
            w[IPR] = pressure;
          }
        }
      }
    };
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, _bottom_bc);
  }
  if (pin->GetOrAddString("mesh", "ox2_bc", "none") == "user") {

    const static Real temperature = pin->GetReal(
      "problem", "exit_temperature");
    const static Real outerspace_pressure = pin->GetReal(
      "problem", "outerspace_pressure");

    auto _vacuum_bc = [](MeshBlock *pmb, Coordinates *pco,
                      AthenaArray<Real> &prim, FaceField &b,
                      Real time, Real dt,
                      int il, int iu,
                      int jl, int ju,
                      int kl, int ku, int ngh) -> void {
      const Real pressure = outerspace_pressure;
      auto water_ice_eos = WaterIceEOS();
      const Real density = water_ice_eos.gas.density(temperature, pressure);
      for (int k = kl; k <= ku; ++k) {
        for (int jj = 1; jj <= ngh; ++jj) {
          for (int i = il; i <= iu; ++i) {
            const int j = jl + jj;
            auto w = prim.at(k, j, i);
            w[IDN] = density;
            w[i_vapor] = 1.;
            w[i_solid] = 0.;
            w[IVX] = 0.;
            w[IVY] = 0.;
            w[IVZ] = 0.;
            w[IPR] = pressure;
          }
        }
      }
    };
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, _vacuum_bc);
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(3);
  SetUserOutputVariableName(0, "temp");
  SetUserOutputVariableName(1, "mass_flux_1");
  SetUserOutputVariableName(2, "mass_flux_2");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0, k, j, i) = pthermo->GetTemp(w.at(k, j, i));

        user_out_var(1, k, j, i) = get_left_mass_flux(
          phydro->flux, X1DIR, k, j, i);
        user_out_var(2, k, j, i) = get_left_mass_flux(
          phydro->flux, X2DIR, k, j, i);
      }
    }
  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  auto pthermo = Thermodynamics::GetInstance();
  auto water_ice_eos = WaterIceEOS();
  const Real temperature = pin->GetReal(
    "problem", "exit_temperature");
  const Real outerspace_pressure = pin->GetReal(
    "problem", "outerspace_pressure");
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real pressure = outerspace_pressure;
        const Real density = water_ice_eos.gas.density(temperature, pressure);
        phydro->w(IDN, k, j, i) = density;
        phydro->w(i_vapor, k, j, i) = 1.;
        phydro->w(IPR, k, j, i) = pressure;
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                              js, je, ks, ke);
}
