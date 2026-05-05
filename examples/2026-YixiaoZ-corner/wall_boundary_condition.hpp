#pragma once

#include "adcpp.hpp"

namespace WallBoundaryCondition {

  template<class Real>
  class VaporSolidInterface {
    public:
      const Real p3;
      const Real temp3;
      const Real gas_constant;
      const Real gamma;
      const Real beta;
      const Real delta;
      VaporSolidInterface(Real p3, Real temp3,
        Real gas_constant, Real gamma, Real beta, Real delta):
          p3(p3), temp3(temp3), gas_constant(gas_constant),
          gamma(gamma), beta(beta), delta(delta) {}

      template<class R>
      inline auto p_sat(const R &temp) const {
        auto t3 = temp / temp3;
        return p3 * exp(beta * (1. - 1./t3) - delta * log(t3));
      }

      template<class R1, class R2>
      inline auto specific_enthalpy_diff(
          const R1 &ice_temp, const R2 &air_temp) const {
        return gas_constant * (
            gamma / (gamma - 1.) * (air_temp - ice_temp)
            + beta * temp3 - delta * ice_temp
        );
      }

      template<class R>
      inline auto one_side_vapor_flux(const R &temp) const {
        return p_sat(temp) / sqrt(2 * M_PI * gas_constant * temp);
      }

      template<class R>
      inline auto one_side_vapor_flux(const R &temp, const R &pres) const {
        return pres / sqrt(2 * M_PI * gas_constant * temp);
      }

      template<class R1, class R2, class R3>
      inline auto net_vapor_flux(
          const R1 &ice_temp, const R2 &air_temp, const R3 &vapor_p) const {
        return (
          one_side_vapor_flux(ice_temp)
          - one_side_vapor_flux(air_temp, vapor_p)
        );
      }

      template<class R1, class R2, class R3>
      inline auto energy_flux(
          const R1 &ice_temp, const R2 &air_temp, const R3 &vapor_p) const {
        return  -(
           net_vapor_flux(ice_temp, air_temp, vapor_p)
           * specific_enthalpy_diff(ice_temp, air_temp)
        );
      }
  };

  template<class Real>
  class OuterSurfaceRadiation {
    public:
      const Real effective_temp;
      const Real stefan_boltzmann_const;

      OuterSurfaceRadiation(Real effective_temp, Real stefan_boltzmann_const):
        effective_temp(effective_temp),
        stefan_boltzmann_const(stefan_boltzmann_const) {}

      template<class R>
      inline auto pow4(const R &x) const {
        auto x2 = x * x;
        return x2 * x2;
      }

      template<class R>
      inline auto energy_flux(const R &ice_temp) const {
        return stefan_boltzmann_const * (pow4(ice_temp) - pow4(effective_temp));
      }

      template<class R>
      inline auto outer_surface_temp(const R &total_energy_flux) const {
        const auto t4 = (
          pow4(effective_temp)
          + total_energy_flux / stefan_boltzmann_const
        );
        return pow(t4, 0.25);
      }
  };

  template<class Real>
  class NewtonRaphsonSolver {
    public:
      const int max_iter;
      const Real f_abstol;

      NewtonRaphsonSolver(const int max_iter, const Real f_abstol):
        max_iter(max_iter), f_abstol(f_abstol) {}

      template<class F>
      Real solve(const F f, Real x, const Real x_min, const Real x_max) const {
        typedef adcpp::fwd::Number<Real> Dual;
        for (int i = 0; i < max_iter; ++i) {
          Dual x_ad(x, 1.);
          Dual f_ad = f(x_ad);
          if (std::abs(f_ad.value()) < f_abstol) {
            break;
          }
          x -= f_ad.value() / f_ad.derivative();
          x = std::clamp(x, x_min, x_max);
        }
        return x;
      }
  };

  template<class Real>
  class BisectSolver {
    public:
      const int max_iter;
      const Real f_abstol;

      BisectSolver(const int max_iter, const Real f_abstol):
        max_iter(max_iter), f_abstol(f_abstol) {}

      template<class F>
      Real solve(const F f, const Real x_min, const Real x_max) const {
        Real xl = x_min;
        Real xr = x_max;
        Real x = xl + 0.5 * (xr - xl);

        const Real fxl = f(xl);
        const Real fxr = f(xr);

        if (fxl <= 0 && fxr >= 0) {
          ;
        } else if (fxl >= 0 && fxr <= 0) {
          const auto tmp = xl;
          xl = xr;
          xr = tmp;
        } else {
          throw std::domain_error("f(xr) and f(xl) have the same sign.");
        }

        for (int i = 0; i < max_iter; ++i) {
          x = xl + 0.5 * (xr - xl);
          const Real fx = f(x);
          if (std::abs(fx) < f_abstol) {
            break;
          }
          if (fx < 0) {
            xl = x;
          } else {
            xr = x;
          }
        }
        return x;
      }
  };


  template<class Real>
  class Solver {
    public:
      VaporSolidInterface<Real> wall;
      OuterSurfaceRadiation<Real> radiation;

      Solver(VaporSolidInterface<Real> wall,
            OuterSurfaceRadiation<Real> radiation):
        wall(wall),
        radiation(radiation) {}

      auto solve(const Real air_temp, const Real vapor_p) const {
        const int max_iter = 64;
        const Real abstol = 1e-12;

        const Real t_min = 50.;
        const Real t_max = 400.;

        // const auto solver = BisectSolver(max_iter, abstol);
        const auto solver = NewtonRaphsonSolver(max_iter, abstol);

        auto f = [this, air_temp, vapor_p](auto ice_temp) {
          const auto rad_energy_flux = radiation.energy_flux(ice_temp);
          const auto wall_energy_flux = wall.energy_flux(
            ice_temp, air_temp, vapor_p);
          return wall_energy_flux - rad_energy_flux;
        };

        const Real ice_temp = solver.solve(f, air_temp, t_min, t_max);
        const Real e = wall.net_vapor_flux(ice_temp, air_temp, vapor_p);

        struct Solution {
          Real ice_temp;
          Real evaporation;
        } solution = {ice_temp, e};
        return solution;
      }
  };

  template<class Real>
  auto build_solver() {
    const Real Avogadro = 6.02214076e23;
    const Real Boltzmann = 1.380649e-23;
    const Real atomic_mass_H = 1.008e-3;
    const Real atomic_mass_O = 15.999e-3;
    const Real water_vapor_cp_mol = 37.4;
    const Real temp3 = 273.16;
    const Real pres3 = 611.7;
    const Real beta = 22.46;
    const Real delta = 0.;

    const Real outer_surface_temp_eff = 67.;
    const Real stefan_boltzmann_const = 5.67e-8;

    const Real universial_gas_constant = Avogadro * Boltzmann;
    const Real water_mw = 2 * atomic_mass_H  + atomic_mass_O;
    const Real water_gas_constant = universial_gas_constant / water_mw;
    const Real water_vapor_cp = water_vapor_cp_mol / water_mw;
    const Real water_vapor_cv = water_vapor_cp - water_gas_constant;
    const Real gamma = water_vapor_cp / water_vapor_cv;

    VaporSolidInterface<Real> wall(
        pres3, temp3, water_gas_constant,
        gamma, beta, delta
    );

    OuterSurfaceRadiation<Real> radiation(
      outer_surface_temp_eff, stefan_boltzmann_const
    );

    Solver<Real> solver (wall, radiation);
    return solver;
  }
}
