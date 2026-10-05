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

template<class Real>
class DensityForcing {
  public:
    const int n_species;
    const Real cp;

    DensityForcing(const int n_species):
      n_species(n_species), cp(get_cp(n_species)) {}

    inline void apply(StrideIterator<Real*> u, StrideIterator<Real*> w,
                      const Real drho, const Real source_temp) const {
        auto pthermo = Thermodynamics::GetInstance();
        const Real t_exchange = (
          (drho > 0) ? source_temp
          : pthermo->GetTemp(w)
        );
        const Real u_exchange = (drho > 0) ? 0. : w[IVX];
        const Real v_exchange = (drho > 0) ? 0. : w[IVY];
        const Real w_exchange = (drho > 0) ? 0. : w[IVZ];

        u[n_species] += drho;
        u[IEN] += drho * (
          cp * t_exchange + 0.5 * (
            square(u_exchange) + square(v_exchange) + square(w_exchange)
          )
        );
        u[IVX] += drho * u_exchange;
        u[IVY] += drho * v_exchange;
        u[IVZ] += drho * w_exchange;
    }
};

void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  if (pthermo->SpeciesIndex("H2O") != i_vapor) {
    throw std::runtime_error("i_vapor does not match");
  }
  if (pthermo->SpeciesIndex("H2O(s)") != i_solid) {
    throw std::runtime_error("i_solid does not match");
  }

  if (pin->GetOrAddString("mesh", "ix1_bc", "none") == "user") {
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
            const int j = ju + jj;
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

  if (pin->GetOrAddString("problem", "wall_boundary_condition", "none")
      == "rad_geo") {

    const static Real exit_temperature = pin->GetReal(
      "problem", "exit_temperature");

    auto _forcing = [](
                 MeshBlock *pmb, Real const time, Real const dt,
                 AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                 AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                 AthenaArray<Real> &s) -> void {

      auto pthermo = Thermodynamics::GetInstance();
      const auto vapor_density_forcing = DensityForcing<Real>(i_vapor);
      const auto solver = WallBoundaryCondition::build_solver<Real>();
      const Real theta = (0.5 * M_PI) - get_xmax(pmb, X2DIR);

      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        for (int j = pmb->js; j <= pmb->je; ++j) {
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            const Real radius = get_xv(pmb, X1DIR, k, j, i);

            if (is_right_boundary(pmb, X2DIR, k, j, i)) {
              auto w_kji = w.at(k, j, i);

              const Real air_temp = pthermo->GetTemp(w_kji);

              const Real vapor_p = (
                  w_kji[IDN] * w_kji[i_vapor] * air_temp
                  * pthermo->GetRd() * pthermo->GetInvMuRatio(i_vapor)
              );

              auto bc = solver.solve(air_temp, vapor_p,
                theta,
                radius, exit_temperature
              );

              const Real evaporation = bc.evaporation;
              const Real ice_temp = bc.ice_temp;

              const Real drho = (
                dt * evaporation * pmb->pcoord->GetFace2Area(k, j+1, i)
                / pmb->pcoord->GetCellVolume(k, j, i)
              );

              vapor_density_forcing.apply(u.at(k, j, i), w.at(k, j, i),
                drho, ice_temp);

              pmb->user_out_var(3, k, j, i) = evaporation;
              pmb->user_out_var(4, k, j, i) = ice_temp;
            }
          }
        }
      }
    };
    EnrollUserExplicitSourceFunction(_forcing);

  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  std::vector<const char*> names = {
    "temp",
    "mass_flux_1",
    "mass_flux_2",
    "evaporation",
    "ice_temp",
  };

  AllocateUserOutputVariables(names.size());
  for (int n = 0; n < names.size(); ++n) {
    SetUserOutputVariableName(n, names[n]);
  }
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
