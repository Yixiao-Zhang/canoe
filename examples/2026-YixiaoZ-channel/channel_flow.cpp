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

const int i_vapor = 1;
const int i_wall = 1;
const int i_flow = 2;


enum class ProblemType {
  LongChannel,
  NudgedShortChannel,
};

const ProblemType problem_type = ProblemType::NudgedShortChannel;

const int buffer_size = 1000;
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

  const Real cv = (
      pthermo->GetRd() / (pthermo->GetGammad() - 1.)
      * pthermo->GetCvRatio(i_vapor)
  );

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
              0.5 * dx, distance, kappa_iso * cv
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

void TopSuction(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();
  auto water_ice_eos = WaterIceEOS();

  const Real velocity_scale = 200.;
  const Real rate = velocity_scale / get_xmax(pmb, i_flow);

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (get_xv(pmb, i_flow, k, j, i) > 0.) {
          const auto w_kji = w.at(k, j, i);
          const Real drho = -dt * rate * w_kji[IDN] * w_kji[i_vapor];

          u(i_vapor, k, j, i) += drho;
          const int nvs[] = {IVX, IVY, IVZ};
          Real b = water_ice_eos.gas.specific_enthalpy(pthermo->GetTemp(w_kji));

          for (auto n: nvs) {
            u(n, k, j, i) += drho * w_kji[n];
            b += 0.5 * square(w_kji[n]);
          }
          u(IEN, k, j, i) += drho * b;
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

  const Real cv = (
    pthermo->GetRd() * pthermo->GetCvRatio(i_vapor)
    / (pthermo->GetGammad() - 1.0)
  );

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
  WallInteraction(pmb, time, dt, w, r, bcc, u, s);

  if (problem_type == ProblemType::LongChannel) {
    BottomInjection(pmb, time, dt, w, r, bcc, u, s);
    TopSuction(pmb, time, dt, w, r, bcc, u, s);
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
  const Real cv = (
      pthermo->GetRd() / (pthermo->GetGammad() - 1.)
      * pthermo->GetCvRatio(i_vapor)
  );

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phdif->kappa(HydroDiffusion::DiffProcess::iso, k, j, i) = (
          phdif->kappa_iso * cv
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

  EnrollUserExplicitSourceFunction(Forcing);
  EnrollViscosityCoefficient(WaterVaporViscosity);
  EnrollConductionCoefficient(WaterVaporConduction);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(6);
  SetUserOutputVariableName(0, "temp");
  SetUserOutputVariableName(1, "ice_temp");
  SetUserOutputVariableName(2, "evaporation");
  SetUserOutputVariableName(3, "sensible_heat_flux");
  SetUserOutputVariableName(4, "total_energy_flux");
  SetUserOutputVariableName(5, "energy_density");
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
        user_out_var(5, k, j, i) = get_energy(w.at(k, j, i));
      }
    }
  }
}

#define PG_INIT_USING_TEXT_FILES

#ifdef PG_INIT_USING_ANALYTICAL_SOLUTION
void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  const auto mesh_size = pmy_mesh->mesh_size;
  const Real ymin = get_xmin(this, i_wall);
  const Real ymax = get_xmax(this, i_wall);
  const Real yc = 0.5 * (ymax + ymin);
  const Real yd = 0.5 * (ymax - ymin);

  auto pthermo = Thermodynamics::GetInstance();
  auto water_ice_eos = WaterIceEOS();
  const Real temperature = pin->GetReal("initialcondition", "temperature");
  const Real pressure = water_ice_eos.pres_sat(temperature);
  const Real density = water_ice_eos.vapor_density_sat(temperature);
  const Real uc = pin->GetReal("initialcondition", "center_velocity");

  // populate to 3D mesh
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real y = get_xv(this, i_wall, k, j, i);
        phydro->w(IDN, k, j, i) = density;
        phydro->w(i_vapor, k, j, i) = 1.;
        phydro->w(IVX + i_flow - 1, k, j, i) = uc * (1. - square((y - yc) / yd));
        phydro->w(IPR, k, j, i) = pressure;
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                              js, je, ks, ke);

  std::mt19937 gen(1234 + 17 * get_mpi_rank());
  std::uniform_real_distribution<Real> phi_distribution(0., 2 * M_PI);

  const Real vp_scale = (
    pin->GetReal("initialcondition", "velocity_perturbation_scale")
  );

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real phi = phi_distribution(gen);
        const Real pert = vp_scale * phydro->u(IVX + i_flow - 1, k, j, i);
        phydro->u(IVX + i_wall - 1, k, j, i) += pert * std::cos(phi);
      }
    }
  }
}
#endif

#ifdef PG_INIT_USING_TEXT_FILES

auto read_vector(const std::string &file_name) {
  std::string line;
  Real value;
  std::vector<Real> values;

  std::ifstream file(file_name);

  if (!file.is_open()) {
    throw std::runtime_error("File does not exist");
  }

  while (std::getline(file, line)) {
      std::istringstream ss(line);
      if (ss >> value) {
        values.push_back(value);
      } else {
        throw std::runtime_error("Illegal value");
      }
  }
  file.close();
  return values;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  // {
  //   auto w = phydro->w.at(ks, js, is);
  //   auto water_ice_eos = WaterIceEOS();
  //   Real temp = water_ice_eos.temp3;

  //   w[IDN] = water_ice_eos.vapor_density_sat(temp);
  //   w[pthermo->SpeciesIndex("H2O")] = 1.;
  //   w[pthermo->SpeciesIndex("H2O(s)")] = 0.;
  //   w[IPR] = water_ice_eos.pres_sat(temp);
  //   std::cout << pthermo->GetInternalEnergy(w) << std::endl;

  //   Real ice_frac = 0.999;

  //   w[IDN] = water_ice_eos.vapor_density_sat(temp) / (1. - ice_frac);
  //   w[pthermo->SpeciesIndex("H2O")] = 1. - ice_frac;
  //   w[pthermo->SpeciesIndex("H2O(s)")] = ice_frac;
  //   w[IPR] = water_ice_eos.pres_sat(temp);
  //   std::cout << pthermo->GetInternalEnergy(w) << std::endl;
  //   std::exit(0);
  // }

  auto init_density = read_vector("init_density.txt");
  auto init_icefraction = read_vector("init_icefraction.txt");
  auto init_velocity = read_vector("init_velocity.txt");
  auto init_pressure = read_vector("init_pressure.txt");

  // populate to 3D mesh
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const int l = (
            get_axis_i(i_wall, k, j, i) - get_axis_i(i_wall, ks, js, is)
        );
        phydro->w(IDN, k, j, i) = init_density[l];
        phydro->w(pthermo->SpeciesIndex("H2O"), k, j, i) = 1. - init_icefraction[l];
        phydro->w(pthermo->SpeciesIndex("H2O(s)"), k, j, i) = init_icefraction[l];
        phydro->w(IVX + i_flow - 1, k, j, i) = init_velocity[l];
        phydro->w(IPR, k, j, i) = init_pressure[l];
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                              js, je, ks, ke);
}
#endif
