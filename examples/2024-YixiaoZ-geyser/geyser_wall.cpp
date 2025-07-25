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

Real H2Oratio, CO2ratio, grav;
int iH2O, iH2Oc, iCO2, iCO2c;

Real x1min, x1max, x2min, x2max;

Real massflux_H2ratio, massflux_CO2ratio;
Real Tm, Ts;

template<class Real>
class VaporCondensation {
  public:
    Real p3;
    Real temp3;
    Real gas_constant;
    Real gamma;
    Real beta;
    Real delta;
    VaporCondensation(Real p3, Real temp3,
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
};

bool fclose(Real x, Real x0) { return std::abs(x - x0) < 1.e-6; }

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(1);
  SetUserOutputVariableName(0, "temp");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        user_out_var(0, k, j, i) = pthermo->GetTemp(w.at(k, j, i));
      }
}

Real surface_temperature(Real x) {
  return 250 + 50 * std::exp(-0.5 * (x / 100.)*(x / 100.));
}

void BottomInjection(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();

  const Real startup_time = 5.;

  // Hard coding the saturated vapor pressure function
  const Real vapor_p3 = 611.7; // Pascal
  const Real vapor_t3 = 273.16; // Kelvin
  const Real vapor_gas_const = 461.5; // J kg-1 K-1
  const Real vapor_adiabatic_index = 1.4;
  const Real vapor_beta = 24.845;
  const Real vapor_delta = 4.986;

  auto vapor_cond = VaporCondensation<Real>(vapor_p3, vapor_t3,
      vapor_gas_const, vapor_adiabatic_index,
      vapor_beta, vapor_delta);

  int i = pmb->is;

  if (pmb->pcoord->x1v(i) <
      pmb->pmy_mesh->mesh_size.x1min + pmb->pcoord->dx1f(i)) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {

        Real t_surface = surface_temperature(pmb->pcoord->x2v(j));
        Real t_air = pthermo->GetTemp(w.at(k, j, i));
        Real p_vapor =  (w(IDN, k, j, i) * w(iH2O, k, j, i)
          * pthermo->GetTemp(w.at(k, j, i))
          * pthermo->GetRd() * pthermo->GetInvMuRatio(iH2O)
        );

        Real drho_dt = (
            vapor_cond.net_vapor_flux(t_surface, t_air, p_vapor)
            * std::min(1., time / startup_time)
            / pmb->pcoord->dx1f(i)
        );
        Real drhoH2O = dt * drho_dt;
        Real drhoH2 = drhoH2O * massflux_H2ratio;
        Real drho = drhoH2O + drhoH2;
        Real t_exchange = (drho > 0) ? t_surface : t_air;

        u(iH2O, k, j, i) += drhoH2O;
        u(IEN, k, j, i) += drhoH2O * (
          (pthermo->GetRd() / (pthermo->GetGammad() - 1.)) * t_exchange
          * pthermo->GetCvRatio(iH2O)
        );

        u(IDN, k, j, i) += drhoH2;
        u(IEN, k, j, i) += drhoH2 * (
          (pthermo->GetRd() / (pthermo->GetGammad() - 1.)) * t_exchange
        );

        if (drho < 0) {
          Real u1 = pmb->phydro->w(IVX, k, j, i);
          Real u2 = pmb->phydro->w(IVY, k, j, i);
          Real u3 = pmb->phydro->w(IVZ, k, j, i);
          Real ke = 0.5 * (u1*u1 + u2*u2 + u3*u3);

          u(IEN, k, j, i) += drho * ke;
          u(IVX, k, j, i) += drho * u1;
          u(IVY, k, j, i) += drho * u2;
          u(IVZ, k, j, i) += drho * u3;
        }
      }
    }
  }
}

void Forcing(MeshBlock *pmb, Real const time, Real const dt,
             AthenaArray<Real> const &w, AthenaArray<Real> const &r,
             AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
             AthenaArray<Real> &s) {
  BottomInjection(pmb, time, dt, w, r, bcc, u, s);
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  H2Oratio = pin->GetReal("initialcondition", "H2Oratio");
  CO2ratio = pin->GetReal("initialcondition", "CO2ratio");

  grav = -pin->GetReal("hydro", "grav_acc1");

  // index
  iH2O = pthermo->SpeciesIndex("H2O");
  iH2Oc = pthermo->SpeciesIndex("H2O(s)");
  // iCO2 = pthermo->SpeciesIndex("CO2");
  // iCO2c = pthermo->SpeciesIndex("CO2(s)");

  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");
  x2min = pin->GetReal("mesh", "x2min");
  x2max = pin->GetReal("mesh", "x2max");

  massflux_H2ratio = pin->GetReal("problem", "massflux_H2ratio");
  massflux_CO2ratio = pin->GetReal("problem", "massflux_CO2ratio");

  Tm = pin->GetReal("problem", "Tm");
  Ts = pin->GetReal("problem", "Ts");

  EnrollUserExplicitSourceFunction(Forcing);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  // construct 1d atmosphere from bottom up
  std::vector<Real> yfrac(IVX, 0.);
  yfrac[iH2O] = H2Oratio;
  // yfrac[iCO2] = CO2ratio;
  yfrac[0] = 1. - H2Oratio;

  int nx1 = pmy_mesh->mesh_size.nx1;
  Real dz = (x1max - x1min) / (nx1 - 1);
  std::cout << "nx1 = " << nx1 << std::endl;

  AthenaArray<Real> w1, z1;
  w1.NewAthenaArray(NHYDRO, nx1);

  z1.NewAthenaArray(nx1);
  z1(0) = x1min + dz / 2.;
  for (int i = 1; i < nx1; ++i) z1(i) = z1(i - 1) + dz;

  pthermo->SetMassFractions<Real>(yfrac.data());
  pthermo->EquilibrateTP(273., 600.);

  // half a grid to cell center
  pthermo->Extrapolate_inplace(dz / 2., "isothermal", grav);

  for (int i = 0; i < nx1; ++i) {
    pthermo->GetPrimitive(w1.at(i));

    // set all clouds to zero
    for (int n = 1 + NVAPOR; n < IVX; ++n) w1(n, i) = 0.;

    // move to the next cell
    pthermo->Extrapolate_inplace(dz, "isothermal", grav);
  }

  // populate to 3D mesh
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        for (int n = 0; n < NHYDRO; ++n) {
          phydro->w(n, k, j, i) =
              interp1(pcoord->x1v(i), w1.data() + n * nx1, z1.data(), nx1);
        }
      }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                             js, je, ks, ke);
}
