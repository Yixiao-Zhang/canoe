// athena
#include <athena/athena.hpp>
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

#include "wall_boundary_condition.hpp"

template<typename T>
auto square(const T x) {
  return x * x;
}

template<class Real>
class IdealGas {
  public:
    Real gas_constant;
    Real specific_cv;

    IdealGas(const Real gas_constant, const Real specific_cv):
      gas_constant(gas_constant), specific_cv(specific_cv) {
    }

    template<class R1, class R2>
    inline auto density(const R1 &temp, const R2 &pres) const {
      return pres / (gas_constant * temp);
    }

    template<class R1>
    inline auto specific_internal_energy(const R1 &temp) const {
      return specific_cv * temp;
    }

    template<class R1>
    inline auto specific_enthalpy(const R1 &temp) const {
      return (specific_cv + gas_constant) * temp;
    }
};

template<class Real>
class CondensedMatter {
  public:
    IdealGas<Real> gas;
    Real temp3;
    Real pres3;
    Real beta;
    Real delta;

    CondensedMatter(const IdealGas<Real> &gas,
        const Real temp3, const Real pres3,
        const Real beta, const Real delta):
      gas(gas), temp3(temp3), pres3(pres3), beta(beta), delta(delta) {}

    template<class R1>
    inline auto specific_internal_energy(const R1 &temp) const {
      return (
        gas.specific_enthalpy(temp)
        + gas.gas_constant * (-beta * temp3 + delta * temp)
      );
    }

    template<class R>
    inline auto pres_sat(const R &temp) const {
      auto t3 = temp / temp3;
      return pres3 * exp(beta * (1. - 1./t3) - delta * log(t3));
    }

    template<class R>
    inline auto vapor_density_sat(const R &temp) const {
      return gas.density(temp, pres_sat(temp));
    }
};

inline auto WaterIceEOS() {
  const double Avogadro = 6.02214076e23;
  const double Boltzmann = 1.380649e-23;
  const Real atomic_mass_H = 1.008e-3;
  const Real atomic_mass_O = 15.999e-3;
  const Real universial_gas_constant = Avogadro * Boltzmann;

  const Real water_mw = 2 * atomic_mass_H  + atomic_mass_O;

  const Real water_gas_constant = universial_gas_constant / water_mw;

  const Real water_vapor_cp_mol = 37.4;

  const Real water_vapor_cp = water_vapor_cp_mol / water_mw;

  const Real water_vapor_cv = water_vapor_cp - water_gas_constant;

  const Real temp3 = 273.16;
  const Real pres3 = 611.7;
  const Real beta = 24.845;
  const Real delta = 4.986;

  IdealGas<Real> water_vapor(water_gas_constant, water_vapor_cv);
  CondensedMatter<Real> water_ice(water_vapor, temp3, pres3, beta, delta);
  return water_ice;
}

inline int get_axis_i(const int axis,
      const int k, const int j, const int i) {
  switch (axis) {
    case 1:
      return i;
    case 2:
      return j;
    case 3:
      return k;
    default:
      throw std::runtime_error("Unknown Axis");
  }
}


inline Real get_xv(MeshBlock *pmb, const int axis,
      const int k, const int j, const int i) {
  switch (axis) {
    case 1:
      return pmb->pcoord->x1v(i);
    case 2:
      return pmb->pcoord->x2v(j);
    case 3:
      return pmb->pcoord->x3v(k);
    default:
      throw std::runtime_error("Unknown Axis");
  }
}

inline Real get_dxf(MeshBlock *pmb, const int axis,
      const int k, const int j, const int i) {
  switch (axis) {
    case 1:
      return pmb->pcoord->dx1f(i);
    case 2:
      return pmb->pcoord->dx2f(j);
    case 3:
      return pmb->pcoord->dx3f(k);
    default:
      throw std::runtime_error("Unknown Axis");
  }
}

inline Real get_xmin(MeshBlock *pmb, const int axis) {
  switch (axis) {
    case 1:
      return pmb->pmy_mesh->mesh_size.x1min;
    case 2:
      return pmb->pmy_mesh->mesh_size.x2min;
    case 3:
      return pmb->pmy_mesh->mesh_size.x3min;
    default:
      throw std::runtime_error("Unknown Axis");
  }
}

inline Real get_xmax(MeshBlock *pmb, const int axis) {
  switch (axis) {
    case 1:
      return pmb->pmy_mesh->mesh_size.x1max;
    case 2:
      return pmb->pmy_mesh->mesh_size.x2max;
    case 3:
      return pmb->pmy_mesh->mesh_size.x3max;
    default:
      throw std::runtime_error("Unknown Axis");
  }
}

inline bool is_left_boundary(MeshBlock *pmb, const int axis,
      const int k, const int j, const int i) {
  return get_xv(pmb, axis, k, j, i) < (
    get_xmin(pmb, axis) + get_dxf(pmb, axis, k, j, i)
  );
}

inline bool is_right_boundary(MeshBlock *pmb, const int axis,
      const int k, const int j, const int i) {
  return get_xv(pmb, axis, k, j, i) > (
    get_xmax(pmb, axis) - get_dxf(pmb, axis, k, j, i)
  );
}

inline bool is_boundary(MeshBlock *pmb, const int axis,
      const int k, const int j, const int i) {
  return (
    is_left_boundary(pmb, axis, k, j, i)
    || is_right_boundary(pmb, axis, k, j, i)
  );
}

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
  const int i_vapor = pthermo->SpeciesIndex("H2O");

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
            u(n, k, j, i) += r * w_kji[n] * w_kji[IDN];
          }

          const Real distance = (
              (problem_type == ProblemType::LongChannel) ?
              -get_xv(pmb, i_flow, k, j, i) : 500.
          );

          if (distance > 0) {
            auto solver = WallBoundaryCondition::build_solver(
              0.5 * dx, distance,
              w_kji[IDN] * w_kji[i_vapor], cv, kappa_iso
            );

            Real air_temp = pthermo->GetTemp(w_kji);

            Real vapor_p = (
                w_kji[IDN] * w_kji[i_vapor] * air_temp
                * pthermo->GetRd() * pthermo->GetInvMuRatio(i_vapor)
            );
            auto bc = solver.solve(air_temp, vapor_p, distance);

            u(IEN, k, j, i) -= dt * bc.sensible_heat_flux / dx;
            const Real drho = dt * bc.evaporation / dx;

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
  const int iH2O = pthermo->SpeciesIndex("H2O");

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
          u(iH2O, k, j, i) += drho;
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
  const int iH2O = pthermo->SpeciesIndex("H2O");

  const Real velocity_scale = 200.;
  const Real rate = velocity_scale / get_xmax(pmb, i_flow);

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (get_xv(pmb, i_flow, k, j, i) > 0.) {
          const auto w_kji = w.at(k, j, i);
          const Real drho = -dt * rate * w_kji[IDN] * w_kji[iH2O];

          u(iH2O, k, j, i) += drho;
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

int get_mpi_rank(const MPI_Comm mpi_world = MPI_COMM_WORLD) {
  int rank;
  MPI_Comm_rank(mpi_world, &rank);
  return rank;
}

template<typename F>
double get_domain_average(F f,MeshBlock *pmb, AthenaArray<Real> const &w) {
  double local_sum, global_sum;
  int local_count, global_count;

  local_sum = 0.;
  local_count = 0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        local_sum += f(w.at(k, j, i));
        ++local_count;
      }
    }
  }

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
    MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT,
    MPI_SUM, MPI_COMM_WORLD);

  return (global_sum / global_count);
}

Real get_density(StrideIterator<Real*> w) {
  return w[IDN];
}

Real get_massflux(StrideIterator<Real*> w) {
  return w[IDN] * w[IVX + i_flow - 1];
}

Real get_energy(StrideIterator<Real*> w) {
  return w[IEN];
}

void Nudge(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {

  auto pthermo = Thermodynamics::GetInstance();
  const int iH2O = pthermo->SpeciesIndex("H2O");

  const Real cv = (
    pthermo->GetRd() * pthermo->GetCvRatio(iH2O)
    / (pthermo->GetGammad() - 1.0)
  );

  const Real mean_density = get_domain_average(get_density, pmb, w);
  const Real mean_massflux = get_domain_average(get_massflux, pmb, w);
  const Real mean_energy = get_domain_average(get_energy, pmb, w);

  const static Real prescribed_density = mean_density;
  const static Real prescribed_massflux = mean_massflux;
  const static Real prescribed_energy = mean_energy;

  const Real relaxation_rate = 2e-2;

  const Real src_density = relaxation_rate * (
    prescribed_density - mean_density);
  const Real src_massflux = relaxation_rate * (
    prescribed_massflux - mean_massflux);
  const Real src_energy = relaxation_rate * (
    prescribed_energy - mean_energy);

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const auto w_kji = w.at(k, j, i);
        u(IDN, k, j, i) += src_density;
        u(IVX + i_flow - 1, k, j, i) += src_massflux;
        u(IEN, k, j, i) += src_energy;
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

void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  EnrollUserExplicitSourceFunction(Forcing);
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
        phydro->w(pthermo->SpeciesIndex("H2O"), k, j, i) = 1.;
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
