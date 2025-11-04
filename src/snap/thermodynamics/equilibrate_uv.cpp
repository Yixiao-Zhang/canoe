// Eigen
#include <Eigen/Core>

// cantera
#include <cantera/kinetics.h>
#include <cantera/kinetics/Condensation.h>
#include <cantera/thermo.h>

// snap
#include "thermodynamics.hpp"

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

template<class Real>
struct EquilibriumCondensationState {
  const Real temp;
  const Real dry_frac;
  const Real vapor_frac;
};


template<class Real>
class EquilibriumCondensation {
  public:

    typedef EquilibriumCondensationState<Real> EQState;

    IdealGas<Real> dry;
    IdealGas<Real> vapor;
    CondensedMatter<Real> cond;

    EquilibriumCondensation(const IdealGas<Real> &dry,
        const IdealGas<Real> &vapor, const CondensedMatter<Real> &cond):
      dry(dry), vapor(vapor), cond(cond) {}

    template<class R1, class R2, class R3>
    inline auto specific_internal_energy(const R1 &temp,
        const R2 & dry_frac, const R3 & vapor_frac) const {
      auto cond_frac = 1. - dry_frac - vapor_frac;
      return (
        dry_frac * dry.specific_internal_energy(temp)
        + vapor_frac * vapor.specific_internal_energy(temp)
        + cond_frac * cond.specific_internal_energy(temp)
      );
    }

    EQState find_equilibrium(
        const Real init_temp, const Real density,
        const Real dry_frac, const Real init_vapor_frac) const {

      const Real init_ie = specific_internal_energy(
        init_temp, dry_frac, init_vapor_frac
      );

      const int max_iter = 32;
      const Real abstol = 1e-5;

      Real temp_min = 1.;
      Real temp_max = 3000.;

      Real temp, vapor_frac;

      for (int iter = 0; iter < max_iter; ++iter) {
        temp = temp_min + 0.5 * (temp_max - temp_min);
        vapor_frac = std::min(
          cond.vapor_density_sat(temp)/ density,
          1. - dry_frac
        );
        if (temp_max - temp_min < abstol) {
          break;
        }
        Real ie = specific_internal_energy(temp, dry_frac, vapor_frac);
        if (ie > init_ie) {
          temp_max = temp;
        } else {
          temp_min = temp;
        }
      }
      return {temp, dry_frac, vapor_frac};
    }
};

void Thermodynamics::EquilibrateUV(Real dt) const {

  const double Avogadro = 6.02214076e23;
  const double Boltzmann = 1.380649e-23;
  const Real atomic_mass_C = 12.011e-3;
  const Real atomic_mass_O = 15.999e-3;
  const Real atomic_mass_Si = 28.085e-3;
  const Real universial_gas_constant = Avogadro * Boltzmann;

  const Real dry_mw = atomic_mass_C + 2 * atomic_mass_O;
  const Real water_mw = atomic_mass_Si + atomic_mass_O;

  const Real dry_gas_constant = universial_gas_constant / dry_mw;
  const Real water_gas_constant = universial_gas_constant / water_mw;

  const Real dry_cp_mol = 37.12;
  const Real water_vapor_cp_mol = 29.9;

  const Real dry_cp = dry_cp_mol / dry_mw;
  const Real water_vapor_cp = water_vapor_cp_mol / water_mw;

  const Real dry_cv = dry_cp - dry_gas_constant;
  const Real water_vapor_cv = water_vapor_cp - water_gas_constant;

  const Real temp3 = 1975.;
  const Real pres3 = 5000.;
  const Real beta = 24.76;
  const Real delta = 0.;

  IdealGas<Real> dry(dry_gas_constant, dry_cv);
  IdealGas<Real> water_vapor(water_gas_constant, water_vapor_cv);
  CondensedMatter<Real> water_ice(water_vapor, temp3, pres3, beta, delta);
  EquilibriumCondensation<Real> eq(dry, water_vapor, water_ice);

  Eigen::VectorXd yfrac(Size);

  auto& thermo = kinetics_->thermo();

  Real ie_0 = thermo.intEnergy_mass();

  thermo.getMassFractions(yfrac.data());
  Real temp = thermo.temperature();
  Real density = thermo.density();

  auto eq_state = eq.find_equilibrium(temp, density, yfrac(0), yfrac(1));

  auto init_temp = temp;
  Eigen::VectorXd init_yfrac = yfrac;

  temp = eq_state.temp;
  yfrac(0) = eq_state.dry_frac;
  yfrac(1) = eq_state.vapor_frac;
  yfrac(2) = 1. - eq_state.dry_frac - eq_state.vapor_frac;

  thermo.setMassFractions(yfrac.data());
  thermo.setTemperature(temp);

  Real ie_1 = thermo.intEnergy_mass();
  // std::cout << ie_1 - ie_0 << std::endl;
  if (std::abs(ie_0 - ie_1) > 10.) {
    std::cout << "intEnergy diff " << ie_1 - ie_0 << std::endl;
    std::cout << "density " << density << std::endl;
    std::cout << "temp " << temp << std::endl;
    std::cout << "yfrac" << yfrac << std::endl;
    std::cout << "init_temp " << init_temp << std::endl;
    std::cout << "init_yfrac" << init_yfrac << std::endl;
  }
}
