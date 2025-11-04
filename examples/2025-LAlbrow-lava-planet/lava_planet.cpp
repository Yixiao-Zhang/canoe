// ==========================================================
// Includes
// ==========================================================
#include <athena/athena.hpp>
#include <athena/athena_arrays.hpp>
#include <athena/bvals/bvals.hpp>
#include <athena/coordinates/coordinates.hpp>
#include <athena/eos/eos.hpp>
#include <athena/field/field.hpp>
#include <athena/hydro/hydro.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/parameter_input.hpp>
#include <climath/interpolation.h>
#include <snap/thermodynamics/atm_thermodynamics.hpp>


// ========================
// CONFIGURATION PARAMETERS
// ========================

// --- Physical constants for vapor ---
const Real VAPOR_P3              = 5000.0;       // Pascal
const Real VAPOR_T3              = 1975.0;       // Kelvin
const Real VAPOR_GAS_CONST       = 188.605;      // J/kg/K
const Real VAPOR_ADIABATIC_INDEX = 1.4;
const Real VAPOR_BETA            = 24.76;
const Real VAPOR_DELTA           = 0.;

// --- Vapor pressure relation constants ---
// Saturation pressure
const Real ASAT = std::pow(10.0, 13.1);
const Real BSAT = 49520.0;
// Chemical equilibrium pressure
const Real AEQ  = std::pow(10.0, 14.086);
const Real BEQ  = 70300.0;

// --- Surface temperature parameters ---
const Real SURF_TEMP_COEFF = 1500.0;  // scaling constant (Atemp)
const Real SURF_TEMP_MIN   = 250.0;   // minimum temperature (Btemp)

// --- Miscellaneous ---
const Real STARTUP_TIME = 5.0; // Time for gradual injection ramp-up

// ========================
// END CONFIGURATION SECTION
// ========================


// NOTES:
// Add SiO chemistry, currently not implemented.
// I don't know whether it is only using values from here or whether it is using the chemistry yaml.
// Where is the forcing used? Yixiao's code seems to have it in this way so i think it's correct. 
// Heating comes from the condensation so weird behaviour without. I've added condensates back.



// ==========================================================
// Globals
// ==========================================================
Real SiOratio, CO2ratio, grav;
int iSiO, iSiOc, iCO2, iCO2c;

Real x1min, x1max, x2min, x2max;
Real massflux_CO2ratio;
Real radius;


// ==========================================================
// VaporCondensation class
// ==========================================================
template<class Real>
class VaporCondensation {
 public:
  Real p3, temp3, gas_constant, gamma, beta, delta;

  VaporCondensation(Real p3, Real temp3,
                    Real gas_constant, Real gamma,
                    Real beta, Real delta)
      : p3(p3), temp3(temp3), gas_constant(gas_constant),
        gamma(gamma), beta(beta), delta(delta) {}

  VaporCondensation(void)
      : p3(VAPOR_P3), temp3(VAPOR_T3), gas_constant(VAPOR_GAS_CONST),
        gamma(VAPOR_ADIABATIC_INDEX),
        beta(VAPOR_BETA), delta(VAPOR_DELTA) {}

  template<class R>
  inline auto p_sat(const R &temp) const {
    auto t3 = temp / temp3;
    return p3 * exp(beta * (1. - 1./t3) - delta * log(t3));
  }

  template<class R1, class R2>
  inline auto specific_enthalpy_diff(const R1 &ice_temp,
                                     const R2 &air_temp) const {
    return gas_constant *
           (gamma / (gamma - 1.0) * (air_temp - ice_temp)
            + beta * temp3 - delta * ice_temp);
  }

  template<class R>
  inline auto one_side_vapor_flux(const R &temp) const {
    return one_side_vapor_flux(temp, p_sat(temp));
  }

  template<class R>
  inline auto one_side_vapor_flux(const R &temp, const R &pres) const {
    return pres / sqrt(2.0 * M_PI * gas_constant * temp);
  }

  template<class R1, class R2, class R3>
  inline auto net_vapor_flux(const R1 &ice_temp,
                             const R2 &air_temp,
                             const R3 &vapor_p) const {
    return one_side_vapor_flux(ice_temp)
           - one_side_vapor_flux(air_temp, vapor_p);
  }
};


// ==========================================================
// Utility functions
// ==========================================================
bool fclose(Real x, Real x0) { return std::abs(x - x0) < 1.e-6; }


// ==========================================================
// MeshBlock Data Setup
// ==========================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(1);
  SetUserOutputVariableName(0, "temp");
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  auto &w = phydro->w;

  auto vapor_cond = VaporCondensation<Real>();

  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        Real temp = pthermo->GetTemp(w.at(k, j, i));
        user_out_var(0, k, j, i) = temp;
      }
}


// ==========================================================
// Surface temperature function
// ==========================================================
template<class Real>
Real surface_temperature(Real theta, Real tempcoeff, Real mintemp) {
  Real c = std::max(std::cos(theta), 0.);
  Real val = tempcoeff * std::pow(c, 0.25);
  return std::max(val, mintemp);
}


// ==========================================================
// Bottom Injection Function
// ==========================================================
void BottomInjection(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();

  auto vapor_cond = VaporCondensation<Real>();

  int i = pmb->is;

  if (pmb->pcoord->x1v(i) <
      pmb->pmy_mesh->mesh_size.x1min + pmb->pcoord->dx1f(i)) {
    for (int k = pmb->ks; k <= pmb->ke; ++k)
      for (int j = pmb->js; j <= pmb->je; ++j) {

        Real t_surface = surface_temperature(pmb->pcoord->x2v(j),
                                             SURF_TEMP_COEFF, SURF_TEMP_MIN);
        Real t_air = pthermo->GetTemp(w.at(k, j, i));

        Real p_vapor = (w(IDN, k, j, i) * w(iSiO, k, j, i)
                        * pthermo->GetTemp(w.at(k, j, i))
                        * pthermo->GetRd() * pthermo->GetInvMuRatio(iSiO));

        Real drho_dt = (vapor_cond.net_vapor_flux(t_surface, t_air, p_vapor)
                        * std::min(1.0, time / STARTUP_TIME)
                        / pmb->pcoord->dx1f(i));

        Real drhoSiO = dt * drho_dt;
        Real drhoCO2 = std::max(drhoSiO * massflux_CO2ratio, 0.);
        Real drho = drhoSiO + drhoCO2;
        Real t_exchange = (drho > 0) ? t_surface : t_air;

        u(iSiO, k, j, i) += drhoSiO;
        u(IEN, k, j, i) += drhoSiO *
          ((pthermo->GetRd() / (pthermo->GetGammad() - 1.0)) *
           t_exchange * pthermo->GetCvRatio(iSiO));

        u(IDN, k, j, i) += drhoCO2;
        u(IEN, k, j, i) += drhoCO2 *
          ((pthermo->GetRd() / (pthermo->GetGammad() - 1.0)) *
           t_exchange);

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


// ==========================================================
// Forcing Wrapper
// ==========================================================
void Forcing(MeshBlock *pmb, Real const time, Real const dt,
             AthenaArray<Real> const &w, AthenaArray<Real> const &r,
             AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
             AthenaArray<Real> &s) {
  BottomInjection(pmb, time, dt, w, r, bcc, u, s);
}


// ==========================================================
// Mesh Initialization
// ==========================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  SiOratio = pin->GetReal("initialcondition", "SiOratio");
  CO2ratio = pin->GetReal("initialcondition", "CO2ratio");
  grav = -pin->GetReal("hydro", "grav_acc1");

  iSiO = pthermo->SpeciesIndex("SiO");
  iSiOc = pthermo->SpeciesIndex("SiO(s)");

  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");
  x2min = pin->GetReal("mesh", "x2min");
  x2max = pin->GetReal("mesh", "x2max");

  massflux_CO2ratio = pin->GetReal("problem", "massflux_CO2ratio");
  radius = pin->GetReal("problem", "radius");

  EnrollUserExplicitSourceFunction(Forcing);
}


// ==========================================================
// Problem Generator
// ==========================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  std::vector<Real> yfrac(IVX, 0.);
  yfrac[0] = 1.;
  yfrac[iSiOc] = 0.;
  yfrac[iSiO] = 1. - yfrac[0] - yfrac[iSiOc];
  pthermo->SetMassFractions<Real>(yfrac.data());

  auto vapor_cond = VaporCondensation<Real>();

  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {
        Real t_surface = surface_temperature(pcoord->x2v(j),
                                             SURF_TEMP_COEFF, SURF_TEMP_MIN);
        Real z = pcoord->x1v(i) - radius;
        Real p_surface = 0.1 * vapor_cond.p_sat(SURF_TEMP_COEFF);
        Real pres = p_surface * std::exp(
          - (z * grav) / (pthermo->GetRd() * t_surface)
        );
        pthermo->EquilibrateTP(t_surface, pres);

        phydro->w(iSiO, k, j, i) = yfrac[iSiO];
        phydro->w(iSiOc, k, j, i) = yfrac[iSiOc];
        phydro->w(IDN, k, j, i) = pthermo->GetDensity();
        phydro->w(IPR, k, j, i) = pthermo->GetPres();
      }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u,
                             pcoord, is, ie, js, je, ks, ke);
}

// reduce cfl,
