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
#include "structured_interpolator.hpp"

constexpr int i_vapor = 1;
constexpr int i_solid = 2;
constexpr int i_norm = 1;
constexpr int i_flow = 2;
constexpr int i_span = 3;

constexpr Real x_flow_exit = 0.;

constexpr Real max_nu = 1.;

Real g_wall_delta;
Real g_wall_x_abstol;
Real g_ice_k;

inline bool is_masked(Real x_norm, Real x_flow) {
  return (
    (std::abs(x_norm) > g_wall_delta - g_wall_x_abstol)
    && (x_flow < x_flow_exit + g_wall_x_abstol)
  );
}

inline bool is_ice_wall_boundary(MeshBlock *pmb,
      const int k, const int j, const int i) {
  const Real xv_norm = get_xv(pmb, i_norm, k, j, i);
  const Real xv_flow = get_xv(pmb, i_flow, k, j, i);
  const Real dxf_norm = get_dxf(pmb, i_norm, k, j, i);
  return (
      (std::abs(xv_norm) < g_wall_delta)
      && (std::abs(xv_norm) + dxf_norm > g_wall_delta)
      && (xv_flow < x_flow_exit)
  );
}

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

void WallInteraction(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();

  const Real kappa_iso = pmb->phydro->hdif.kappa_iso;

  const Real gas_cp = get_cp(i_vapor);

  const auto vapor_density_forcing = DensityForcing<Real>(i_vapor);
  const auto solid_density_forcing = DensityForcing<Real>(i_solid);

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const auto w_kji = w.at(k, j, i);
        auto u_kji = w.at(k, j, i);
        const Real rho = w_kji[IDN];
        if (is_ice_wall_boundary(pmb, k, j, i)) {
          const Real dx = get_dxf(pmb, i_norm, k, j, i);
          const auto solver = WallBoundaryCondition::build_solver(
            0.5 * dx, get_mu(kappa_iso, rho) * rho * gas_cp, g_ice_k
          );

          const Real distance = x_flow_exit - get_xv(pmb, i_flow, k, j, i);

          const Real air_temp = pthermo->GetTemp(w_kji);

          const Real vapor_p = (
              w_kji[IDN] * w_kji[i_vapor] * air_temp
              * pthermo->GetRd() * pthermo->GetInvMuRatio(i_vapor)
          );

          auto bc = solver.solve(air_temp, vapor_p, distance);

          u(IEN, k, j, i) -= dt * bc.sensible_heat_flux / dx;

          const Real drho_vapor = dt * bc.evaporation / dx;
          vapor_density_forcing.apply(u_kji, w_kji, drho_vapor, bc.ice_temp);

          const Real solid_density = w_kji[IDN] * w_kji[i_solid];
          const Real drho_solid = -0.5 * solid_density;
          solid_density_forcing.apply(u_kji, w_kji, drho_solid, bc.ice_temp);

          pmb->user_out_var(1, k, j, i) = bc.ice_temp;
          pmb->user_out_var(2, k, j, i) = bc.evaporation;
          pmb->user_out_var(3, k, j, i) = bc.sensible_heat_flux;
          pmb->user_out_var(4, k, j, i) = bc.total_energy_flux;
          pmb->user_out_var(5, k, j, i) = drho_solid * dx / dt;
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

  const auto vapor_density_forcing = DensityForcing<Real>(i_vapor);

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real x_norm = get_xv(pmb, i_norm, k, j, i);
        const Real x_flow = get_xv(pmb, i_flow, k, j, i);
        if (is_left_boundary(pmb, i_flow, k, j, i)
            && !is_masked(x_norm, x_flow)) {
          const Real p = pmb->phydro->w(IPR, k, j, i);
          const Real d = get_dxf(pmb, i_flow, k, j, i);
          const Real drho = dt * (
              std::max(water_ice_eos.pres3 - p, 0.)
              / (
                sqrt(
                    2 * M_PI * water_ice_eos.gas.gas_constant
                    * water_ice_eos.temp3
                ) * d
              )
          );

          vapor_density_forcing.apply(u.at(k, j, i), w.at(k, j, i),
            drho, water_ice_eos.temp3);
        }
      }
    }
  }
}

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

  g_ice_k = pin->GetReal("problem", "ice_k");

  g_wall_delta = pin->GetReal("problem", "wall_delta");
  g_wall_x_abstol = 1e-8 * std::abs(g_wall_delta);

  const static bool forcing_wall_interaction = pin->GetBoolean("problem",
    "forcing_wall_interaction");
  const static bool forcing_bottom_ejection = pin->GetBoolean("problem",
    "forcing_bottom_ejection");

  auto _forcing = [](MeshBlock *pmb, Real const time, Real const dt,
               AthenaArray<Real> const &w, AthenaArray<Real> const &r,
               AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
               AthenaArray<Real> &s) -> void {
    if (forcing_wall_interaction) {
      WallInteraction(pmb, time, dt, w, r, bcc, u, s);
    }
    if (forcing_bottom_ejection) {
      BottomInjection(pmb, time, dt, w, r, bcc, u, s);
    }
  };

  EnrollUserExplicitSourceFunction(_forcing);
  EnrollViscosityCoefficient(WaterVaporViscosity);
  EnrollConductionCoefficient(WaterVaporConduction);

  if (pin->GetReal("mesh", "x1rat") < 0) {
    const static Real nxc = pin->GetReal("problem", "nx_half_conduit");
    const static Real nxd = 0.5 * pin->GetReal("mesh",
      std::string("nx") + std::to_string(i_norm));
    const static Real rat = pin->GetReal("problem", "xnorm_rat");

    auto my_mesh_spacing_x1 = [](const Real t0, const RegionSize rs) -> Real {
      const Real t1 = 2 * t0 - 1;
      const Real s = std::copysign(static_cast<Real>(1), t1);
      const Real t = s * t1;

      const Real n = t * nxd;
      const Real dx = g_wall_delta / nxc;

      const Real x_abs = (
        (n < nxc) ?
        (n * dx)
        : (g_wall_delta + dx * (std::pow(rat, n - nxc) - 1.) / (rat - 1.)
        )
      );

      return std::copysign(x_abs, s);
    };
    EnrollUserMeshGenerator(X1DIR, my_mesh_spacing_x1);
  }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(7);
  SetUserOutputVariableName(0, "temp");
  SetUserOutputVariableName(1, "ice_temp");
  SetUserOutputVariableName(2, "evaporation");
  SetUserOutputVariableName(3, "sensible_heat_flux");
  SetUserOutputVariableName(4, "total_energy_flux");
  SetUserOutputVariableName(5, "ice_mass_flux");
  SetUserOutputVariableName(6, "mass_flux");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0, k, j, i) = pthermo->GetTemp(w.at(k, j, i));

        Real mass_flux = 0.;
        for (int n = 0; n < IVX; ++n) {
          mass_flux += get_center_flux(phydro->flux, i_flow, n, k, j, i);
        }
        user_out_var(6, k, j, i) = mass_flux;
      }
    }
  }
}

void reflecting_inner_x2(MeshBlock *pmb, Coordinates *pco,
                        AthenaArray<Real> &prim, FaceField &b, Real time,
                        Real dt, int il, int iu, int jl, int ju, int kl, int ku,
                        int ngh) {
  for (int n = 0; n < NHYDRO; ++n) {
    const int sign = (IVX <= n && n <= IVZ) ? -1 : 1;
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
    const int sign = (IVX <= n && n <= IVZ) ? -1 : 1;
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
    const int sign = (IVX <= n && n <= IVZ) ? -1 : 1;
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
    const int sign = (IVX <= n && n <= IVZ) ? -1 : 1;
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
  Real x3 = (block_size.x3min + block_size.x3max) / 2;

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
  const std::tuple xs {x1, x2, x3};
  const Real x_norm = std::get<i_norm-1>(xs);
  const Real x_flow = std::get<i_flow-1>(xs);
  const Real x_span = std::get<i_span-1>(xs);
  struct BoundaryCenter {
    Real x_norm;
    Real x_flow;
    Real x_span;
  } bc {x_norm, x_flow, x_span};
  return bc;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  const bool initialize_with_ncfile = pin->GetOrAddBoolean("problem",
    "initialize_with_ncfile", false);

  if (initialize_with_ncfile) {
    const std::string ncfile("init.nc");
    CoordinateSystem<double, 4> coord("x1", "x2", "x3", "time");

    const std::array<std::string, NHYDRO> variable_names{
      "rho", "H2O", "H2O(s)", "vel1", "vel2", "vel3", "press"
    };

    std::vector<decltype(coord)::Field> fields;
    fields.reserve(variable_names.size());

    for (const auto& name : variable_names) {
      fields.push_back(coord.load(ncfile, name));
    }

    const Real time = 0.;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real x3 = pcoord->x3v(k);
          const Real x2 = pcoord->x2v(j);
          const Real x1 = pcoord->x1v(i);
          const auto loc = coord.location(x1, x2, x3, time);
          for (int n = 0; n < NHYDRO; ++n){
            phydro->w(n, k, j, i) = fields[n].interpolate(loc);
          }
        }
      }
    }
  } else {
    auto pthermo = Thermodynamics::GetInstance();
    auto water_ice_eos = WaterIceEOS();
    const Real temperature = water_ice_eos.temp3;
    const Real bottom_pressure = water_ice_eos.pres_sat(temperature);
    const Real outerspace_pressure = pin->GetReal("problem",
      "outerspace_pressure");
    const Real x_flow_bottom = get_xmin(this, i_flow);
    const Real gamma = (
      std::log(bottom_pressure / outerspace_pressure)
      / (x_flow_exit - x_flow_bottom)
    );

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real x_norm = get_xv(this, i_norm, k, j, i);
          const Real x_flow = get_xv(this, i_flow, k, j, i);

          const Real pressure = (
            bottom_pressure
            * ((is_masked(x_norm, x_flow)) ?
              0.99
              : std::exp(-gamma * (
                std::min(x_flow, x_flow_exit) - x_flow_bottom
              ))
            )
          );
          const Real density = water_ice_eos.gas.density(temperature, pressure);
          phydro->w(IDN, k, j, i) = density;
          phydro->w(i_vapor, k, j, i) = 1.;
          phydro->w(IPR, k, j, i) = pressure;
        }
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                              js, je, ks, ke);

  const int bfs[] = {BoundaryFace::inner_x2, BoundaryFace::outer_x2,
    BoundaryFace::inner_x1, BoundaryFace::outer_x1};

  for (auto bf: bfs) {
    const auto bc = get_boundary_center(this, bf);
    if (is_masked(bc.x_norm, bc.x_flow)) {
      pmy_mesh->mesh_bcs[bf] = BoundaryFlag::user;
      pbval->block_bcs[bf] = BoundaryFlag::user;
      pbval->apply_bndry_fn_[bf] = true;

      const auto func = get_user_boundary_function(bf);
      pmy_mesh->EnrollUserBoundaryFunction(bf, func);
    }
  }
}
