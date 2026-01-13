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


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(1);
  SetUserOutputVariableName(0, "temp");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        user_out_var(0, k, j, i) = pthermo->GetTemp(w.at(k, j, i));
      }
    }
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
    get_xmax(pmb, axis) + get_dxf(pmb, axis, k, j, i)
  );
}

inline bool is_boundary(MeshBlock *pmb, const int axis,
      const int k, const int j, const int i) {
  return (
    is_left_boundary(pmb, axis, k, j, i)
    || is_right_boundary(pmb, axis, k, j, i)
  );
}

const int i_wall = 2;
const int i_flow = 1;

void WallInteraction(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();

  const Real nu_iso = pmb->phydro->hdif.nu_iso;

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (is_boundary(pmb, i_wall, k, j, i)) {
          const auto w_kji = w.at(k, j, i);
          const Real d = get_dxf(pmb, i_wall, k, j, i);
          const Real r = -dt * nu_iso / square(0.5 * d);

          const int nvs[] = {IVX, IVY, IVZ};
          for (auto n: nvs) {
            u(n, k, j, i) += r * w_kji[n] * w_kji[IDN];
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

  const Real velocity_scale = 400.;
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



void Forcing(MeshBlock *pmb, Real const time, Real const dt,
             AthenaArray<Real> const &w, AthenaArray<Real> const &r,
             AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
             AthenaArray<Real> &s) {
  WallInteraction(pmb, time, dt, w, r, bcc, u, s);
  BottomInjection(pmb, time, dt, w, r, bcc, u, s);
  TopSuction(pmb, time, dt, w, r, bcc, u, s);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  EnrollUserExplicitSourceFunction(Forcing);
}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  auto pthermo = Thermodynamics::GetInstance();
  auto const water_ice_eos = WaterIceEOS();
  const Real frac = 1e-8;
  const Real temperature = water_ice_eos.temp3;
  const Real pressure = frac * water_ice_eos.pres_sat(temperature);
  const Real density = water_ice_eos.gas.density(temperature, pressure);

  // populate to 3D mesh
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->w(IDN, k, j, i) = density;
        phydro->w(pthermo->SpeciesIndex("H2O"), k, j, i) = 1.;
        phydro->w(IPR, k, j, i) = pressure;
      }
    }
  }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                             js, je, ks, ke);
}
