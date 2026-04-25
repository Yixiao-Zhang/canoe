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

constexpr int i_vapor = 1;
constexpr int i_solid = 2;
constexpr int i_flow = 1;
constexpr int i_norm = 2;
constexpr int i_span = 3;

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


void WaterVaporConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                     const AthenaArray<Real> &bcc,
                     int is, int ie, int js, int je, int ks, int ke) {
  auto pthermo = Thermodynamics::GetInstance();

  const Real gas_cp = get_cp(i_vapor);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const Real rho = prim(IDN, k, j, i);
        const Real kappa = get_mu(phdif->kappa_iso, rho);
        phdif->kappa(HydroDiffusion::DiffProcess::iso, k, j, i) = (
          kappa * rho * gas_cp
        );
      }
    }
  }
  return;
}

void WaterVaporViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                    const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
                    int ks, int ke) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        const Real rho = prim(IDN, k, j, i);
        const Real nu = get_mu(phdif->nu_iso, rho);
        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = (
          nu
        );
      }
    }
  }
  return;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  if (pthermo->SpeciesIndex("H2O") != i_vapor) {
    throw std::runtime_error("i_vapor does not match");
  }
  if (pthermo->SpeciesIndex("H2O(s)") != i_solid) {
    throw std::runtime_error("i_solid does not match");
  }

  const static bool forcing_wall_interaction = pin->GetBoolean("problem",
    "forcing_wall_interaction");

  if (forcing_wall_interaction) {
    const static Real exit_delta = pin->GetReal(
      "problem", "exit_delta");
    auto _forcing = [](MeshBlock *pmb, Real const time, Real const dt,
                 AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                 AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                 AthenaArray<Real> &s) -> void {

      auto pthermo = Thermodynamics::GetInstance();
      auto water_ice_eos = WaterIceEOS();
      const auto vapor_density_forcing = DensityForcing<Real>(i_vapor);

      for (int k = pmb->ks; k <= pmb->ke; ++k) {
        for (int j = pmb->js; j <= pmb->je; ++j) {
          for (int i = pmb->is; i <= pmb->ie; ++i) {
            const Real x_norm = get_xv(pmb, i_norm, k, j, i);
            if (is_left_boundary(pmb, i_flow, k, j, i)
                && std::abs(x_norm) > exit_delta) {
              auto w_kji = w.at(k, j, i);
              const Real dx = get_dxf(pmb, i_flow, k, j, i);
              const Real mass_flux = (
                  -w_kji[IPR]
                  / (
                    std::sqrt(
                        2 * M_PI * water_ice_eos.gas.gas_constant
                        * pthermo->GetTemp(w_kji)
                    )
                  )
              );

              pmb->user_out_var(3, k, j, i) = mass_flux;

              const Real drho = dt * mass_flux / dx;

              const Real dummy_wall_temp = 0.;

              vapor_density_forcing.apply(u.at(k, j, i), w.at(k, j, i),
                drho, dummy_wall_temp);
            }
          }
        }
      }
    };
    EnrollUserExplicitSourceFunction(_forcing);
  }

  EnrollViscosityCoefficient(WaterVaporViscosity);
  EnrollConductionCoefficient(WaterVaporConduction);

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
            const Real x = get_xv(pmb, i_norm, k, j, i);

            if (std::abs(x) < exit_delta) {
              auto water_ice_eos = WaterIceEOS();
              const Real pressure = water_ice_eos.pres_sat(temperature);
              const Real density = water_ice_eos.gas.density(
                temperature, pressure);
              w[IDN] = density;
              w[i_vapor] = 1.;
              w[i_solid] = 0.;
              for (int n = IVX; n <= IVZ; ++n) {
                w[n] = 0.;
              }
              w[IVX+i_flow-1] += (
                center_velocity * (1. - square(x/exit_delta)));
              w[IPR] = pressure;
            } else {
              auto wi = prim.at(k, j, il + ii);
              for (int n = 0; n < NHYDRO; ++n) {
                const int sign = (IVX <= n && n <= IVZ) ? -1 : 1;
                w[n] = sign * wi[n];
              }
            }
          }
        }
      }
    };
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, _bottom_bc);
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "temp");
  SetUserOutputVariableName(1, "mass_flux_1");
  SetUserOutputVariableName(2, "mass_flux_2");
  SetUserOutputVariableName(3, "mass_flux");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0, k, j, i) = pthermo->GetTemp(w.at(k, j, i));

        user_out_var(1, k, j, i) = get_center_mass_flux(
          phydro->flux, 1, k, j, i);
        user_out_var(2, k, j, i) = get_center_mass_flux(
          phydro->flux, 2, k, j, i);
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
