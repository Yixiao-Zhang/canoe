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

#include "wall_boundary_condition.hpp"
#include "channel_utils.hpp"

constexpr int i_vapor = 1;
constexpr int i_solid = 2;
constexpr int i_wall = 1;
constexpr int i_flow = 2;


enum class ProblemType {
  LongChannel,
  NudgedShortChannel,
};

const ProblemType problem_type = ProblemType::LongChannel;

constexpr int buffer_size = 1000;
Real g_ice_temp[buffer_size];
Real g_evaporation[buffer_size];
Real g_sensible_heat_flux[buffer_size];
Real g_total_energy_flux[buffer_size];


void WallInteraction(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();

  const Real nu_iso = pmb->phydro->hdif.nu_iso;
  const Real kappa_iso = pmb->phydro->hdif.kappa_iso;

  const Real cp = (
      pthermo->GetRd() * (
        (pthermo->GetCvRatio(i_vapor) / (pthermo->GetGammad() - 1.))
        + pthermo->GetInvMuRatio(i_vapor)
      )
  );

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (is_boundary(pmb, i_wall, k, j, i)) {
          const auto w_kji = w.at(k, j, i);
          const Real dx = get_dxf(pmb, i_wall, k, j, i);
          const Real r = -dt * nu_iso / square(0.5 * dx);

          const int nvs[] = {IVX, IVY, IVZ};
          for (auto n: nvs) {
            u(n, k, j, i) += r * w_kji[n];
          }

          const Real distance = (
              (problem_type == ProblemType::LongChannel) ?
              -get_xv(pmb, i_flow, k, j, i) : 500.
          );

          if (distance > 0) {
            auto solver = WallBoundaryCondition::build_solver(
              0.5 * dx, distance, kappa_iso * cp
            );

            Real air_temp = pthermo->GetTemp(w_kji);

            Real vapor_p = (
                w_kji[IDN] * w_kji[i_vapor] * air_temp
                * pthermo->GetRd() * pthermo->GetInvMuRatio(i_vapor)
            );
            auto bc = solver.solve(air_temp, vapor_p, distance);

            u(IEN, k, j, i) -= dt * bc.sensible_heat_flux / dx;
            // const Real drho = dt * bc.evaporation / dx;
            const Real drho = 0.;

            const Real t_exchange = (drho > 0) ? bc.ice_temp : air_temp;
            const Real u_exchange = (drho > 0) ? 0. : w_kji[IVX];
            const Real v_exchange = (drho > 0) ? 0. : w_kji[IVY];
            const Real w_exchange = (drho > 0) ? 0. : w_kji[IVZ];

            u(i_vapor, k, j, i) += drho;
            u(IEN, k, j, i) += drho * (
              cp * t_exchange + 0.5 * (
                square(u_exchange) + square(v_exchange) + square(w_exchange)
              )
            );
            u(IVX, k, j, i) += drho * u_exchange;
            u(IVY, k, j, i) += drho * v_exchange;
            u(IVZ, k, j, i) += drho * w_exchange;

            const int i_out = get_axis_i(i_flow, k, j, i);
            if (i_out > buffer_size - 1) {
              std::cout << "buffer_size is too small." << std::endl;
            }
            g_ice_temp[i_out] = bc.ice_temp;
            g_total_energy_flux[i_out] = bc.total_energy_flux;
            g_evaporation[i_out] = bc.evaporation;
            g_sensible_heat_flux[i_out] = bc.sensible_heat_flux;
          }
        }
      }
    }
  }
}

void BottomInjection(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();
  auto water_ice_eos = WaterIceEOS();

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (is_left_boundary(pmb, i_flow, k, j, i)) {
          const Real p = pmb->phydro->w(IPR, k, j, i);
          const Real d = get_dxf(pmb, i_flow, k, j, i);
          const Real drho= dt * (
              std::max(water_ice_eos.pres3 - p, 0.)
              / (
                sqrt(
                    2 * M_PI * water_ice_eos.gas.gas_constant
                    * water_ice_eos.temp3
                ) * d
              )
          );
          u(i_vapor, k, j, i) += drho;
          u(IEN, k, j, i) += (
              drho * water_ice_eos.gas.specific_enthalpy(water_ice_eos.temp3)
          );
        }
      }
    }
  }
}

Real get_density(StrideIterator<Real*> w) {
  return w[IDN];
}

Real get_massflux(StrideIterator<Real*> w) {
  return w[IDN] * w[IVX + i_flow - 1];
}

Real get_energy(StrideIterator<Real*> w) {
  auto pthermo = Thermodynamics::GetInstance();
  return w[IDN] * (
    pthermo->GetInternalEnergy(w)
    + 0.5 * (
      square(w[IVX])
      + square(w[IVY])
      + square(w[IVZ])
    )
  );
}

void Nudge(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {

  auto pthermo = Thermodynamics::GetInstance();

  const Real mean_density = get_domain_average(get_density, pmb, w);
  const Real mean_massflux = get_domain_average(get_massflux, pmb, w);
  const Real mean_energy = get_domain_average(get_energy, pmb, w);

  const static Real prescribed_density = mean_density;
  const static Real prescribed_massflux = mean_massflux;
  const static Real prescribed_energy = mean_energy;

  const Real relaxation_rate = 0.2;

  const Real src_density = relaxation_rate * (
    prescribed_density - mean_density);
  const Real src_velocity = relaxation_rate * (
    prescribed_massflux - mean_massflux) / mean_density;
  const Real src_energy = relaxation_rate * (
    prescribed_energy - mean_energy) / mean_density;

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const auto w_kji = w.at(k, j, i);
        const Real density = w_kji[IDN];
        // u(i_vapor, k, j, i) += src_density;
        u(IVX + i_flow - 1, k, j, i) += src_velocity * density;
        u(IEN, k, j, i) += src_energy * density;
      }
    }
  }
}


void Forcing(MeshBlock *pmb, Real const time, Real const dt,
             AthenaArray<Real> const &w, AthenaArray<Real> const &r,
             AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
             AthenaArray<Real> &s) {
  // WallInteraction(pmb, time, dt, w, r, bcc, u, s);

  if (problem_type == ProblemType::LongChannel) {
    BottomInjection(pmb, time, dt, w, r, bcc, u, s);
  } else if (problem_type == ProblemType::NudgedShortChannel) {
    Nudge(pmb, time, dt, w, r, bcc, u, s);
  } else {
    throw std::runtime_error("Unknown problem_type");
  }
}

void WaterVaporConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                     const AthenaArray<Real> &bcc,
                     int is, int ie, int js, int je, int ks, int ke) {
  auto pthermo = Thermodynamics::GetInstance();

  const Real cp = (
      pthermo->GetRd() * (
        (pthermo->GetCvRatio(i_vapor) / (pthermo->GetGammad() - 1.))
        + pthermo->GetInvMuRatio(i_vapor)
      )
  );

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phdif->kappa(HydroDiffusion::DiffProcess::iso, k, j, i) = (
          phdif->kappa_iso * cp
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
        phdif->nu(HydroDiffusion::DiffProcess::iso, k, j, i) = (
          phdif->nu_iso / prim(IDN, k, j, i)
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

  EnrollUserExplicitSourceFunction(Forcing);
  EnrollViscosityCoefficient(WaterVaporViscosity);
  EnrollConductionCoefficient(WaterVaporConduction);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(5);
  SetUserOutputVariableName(0, "temp");
  SetUserOutputVariableName(1, "ice_temp");
  SetUserOutputVariableName(2, "evaporation");
  SetUserOutputVariableName(3, "sensible_heat_flux");
  SetUserOutputVariableName(4, "total_energy_flux");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0, k, j, i) = pthermo->GetTemp(w.at(k, j, i));

        const int i_out = get_axis_i(i_flow, k, j, i);
        if (i_out > buffer_size - 1) {
          std::cout << "buffer_size is too small." << std::endl;
        }

        user_out_var(1, k, j, i) = g_ice_temp[i_out];
        user_out_var(2, k, j, i) = g_evaporation[i_out];
        user_out_var(3, k, j, i) = g_sensible_heat_flux[i_out];
        user_out_var(4, k, j, i) = g_total_energy_flux[i_out];
      }
    }
  }
}

void reflecting_inner_x2(MeshBlock *pmb, Coordinates *pco,
                        AthenaArray<Real> &prim, FaceField &b, Real time,
                        Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                        int ngh) {
  for (int n = 0; n < NHYDRO; ++n) {
    const int sign = (IVY <= n && n <= IVZ) ? -1 : 1;
    for (int k = kl; k <= ku; ++k) {
      for (int j = 1; j <= ngh; ++j) {
        for (int i = il; i <= iu; ++i) {
          prim(n, k, jl - j, i) = sign * prim(n, k, jl + j - 1, i);
        }
      }
    }
  }
}

void reflecting_outer_x2(MeshBlock *pmb, Coordinates *pco,
                        AthenaArray<Real> &prim, FaceField &b, Real time,
                        Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                        int ngh) {
  for (int n = 0; n < NHYDRO; ++n) {
    const int sign = (IVY <= n && n <= IVZ) ? -1 : 1;
    for (int k = kl; k <= ku; ++k) {
      for (int j = 1; j <= ngh; ++j) {
        for (int i = il; i <= iu; ++i) {
          prim(n, k, ju + j, i) = sign * prim(n, k, ju - j + 1, i);
        }
      }
    }
  }
}


void reflecting_inner_x1(MeshBlock *pmb, Coordinates *pco,
                        AthenaArray<Real> &prim, FaceField &b, Real time,
                        Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                        int ngh) {
  for (int n = 0; n < NHYDRO; ++n) {
    const int sign = (IVY <= n && n <= IVZ) ? -1 : 1;
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = 1; i <= ngh; ++i) {
          prim(n, k, j, il - i) = sign * prim(n, k, j, il + i - 1);
        }
      }
    }
  }
}

void reflecting_outer_x1(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b, Real time,
                         Real dt, int il, int iu, int jl, int ju, int kl,
                         int ku, int ngh) {
  for (int n = 0; n < NHYDRO; ++n) {
    const int sign = (IVY <= n && n <= IVZ) ? -1 : 1;
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = 1; i <= ngh; ++i) {
          prim(n, k, j, iu + i) = sign * prim(n, k, j, iu - i + 1);
        }
      }
    }
  }
}


template <class T>
auto get_user_boundary_function(T bf) {
  if (bf == BoundaryFace::inner_x2) {
    return reflecting_inner_x2;
  } else if (bf == BoundaryFace::outer_x2) {
    return reflecting_outer_x2;
  } else if (bf == BoundaryFace::inner_x1) {
    return reflecting_inner_x1;
  } else if (bf == BoundaryFace::outer_x1) {
    return reflecting_outer_x1;
  } else {
    throw std::runtime_error("Unknown BoundaryFace");
  }
}


template <class T>
auto get_boundary_center(MeshBlock *pblock, T bf) {
  const auto block_size = pblock->block_size;
  Real x1 = (block_size.x1min + block_size.x1max) / 2;
  Real x2 = (block_size.x2min + block_size.x2max) / 2;

  if (bf == BoundaryFace::inner_x2) {
    x2 = block_size.x2min;
  } else if (bf == BoundaryFace::outer_x2) {
    x2 = block_size.x2max;
  } else if (bf == BoundaryFace::inner_x1) {
    x1 = block_size.x1min;
  } else if (bf == BoundaryFace::outer_x1) {
    x1 = block_size.x1max;
  } else {
    throw std::runtime_error("Unknown BoundaryFace");
  }
  struct BoundaryCenter {
    Real x1;
    Real x2;
  } bc {x1, x2};
  return bc;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  auto pthermo = Thermodynamics::GetInstance();
  auto water_ice_eos = WaterIceEOS();
  const Real temperature = water_ice_eos.temp3;
  const Real pressure = 0.5 * water_ice_eos.pres_sat(temperature);
  const Real density = water_ice_eos.gas.density(temperature, pressure);

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real y = get_xv(this, i_wall, k, j, i);
        phydro->w(IDN, k, j, i) = density;
        phydro->w(i_vapor, k, j, i) = 1.;
        phydro->w(IPR, k, j, i) = pressure;
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                              js, je, ks, ke);

  const Real wall_x1_min = pin->GetReal("problem", "wall_x1_min");
  const Real wall_x1_max = pin->GetReal("problem", "wall_x1_max");
  const Real wall_x2_min = pin->GetReal("problem", "wall_x2_min");
  const Real wall_x2_max = pin->GetReal("problem", "wall_x2_max");

  const int bfs[] = {BoundaryFace::inner_x2, BoundaryFace::outer_x2,
    BoundaryFace::inner_x1, BoundaryFace::outer_x1};
  for (auto bf: bfs) {
    const auto bc = get_boundary_center(this, bf);
    std::cout << "loc: " << bc.x1 << ", " << bc.x2 << std::endl;
    const bool is_wall = (
        (bc.x1 > wall_x1_min) && (bc.x1 < wall_x1_max)
        && (bc.x2 < wall_x2_min) && (bc.x2 < wall_x2_max)
    );
    if (is_wall) {
      pmy_mesh->mesh_bcs[bf] = BoundaryFlag::user;
      pbval->block_bcs[bf] = BoundaryFlag::user;
      pbval->apply_bndry_fn_[bf] = true;

      std::cout << "EnrollUserBoundaryFunction: " << bf << std::endl;

      const auto func = get_user_boundary_function(bf);
      pmy_mesh->EnrollUserBoundaryFunction(bf, func);
    }
  }
}
