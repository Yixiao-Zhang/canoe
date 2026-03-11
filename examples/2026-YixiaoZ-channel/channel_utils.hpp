#pragma once

#include <athena/mesh/mesh.hpp>
#include <mpi.h>

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

inline Real get_x_range(MeshBlock *pmb, const int axis) {
  return get_xmax(pmb, axis) - get_xmin(pmb, axis);
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

inline int get_mpi_rank(const MPI_Comm mpi_world = MPI_COMM_WORLD) {
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
