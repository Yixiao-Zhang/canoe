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

template<class Real>
inline auto square(Real x) {
  return x * x;
}


// ========================
// CONFIGURATION PARAMETERS
// ========================

// --- Physical constants for vapor ---
const Real SiO_VAPOR_GAS_CONST       = 188.605;      // J/kg/K
const Real SiO_VAPOR_ADIABATIC_INDEX = 1.4;

// --- Vapor pressure relation constants ---
// Saturation pressure
// const Real SiO_ASAT = std::pow(10.0, 13.1);
// const Real SiO_BSAT = 49520.0;
// Chemical equilibrium pressure
const Real SiO_AEQ  = std::pow(10.0, 14.086);
const Real SiO_BEQ  = 70300.0;

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
Real surface_temperature_min;
Real surface_temperature_max;
Real surface_grav;
Real planet_rotation;
Real mass_attenuation_coefficient;
Real substellar_incoming_radiation;

template<class Real>
class OneStreamRadiativeTransfer {
  public:
    OneStreamRadiativeTransfer(
        const int i_species,
        const Real mass_attenuation_coefficient,
        const Real substellar_incoming_radiation):
          i_species(i_species),
          mass_attenuation_coefficient(mass_attenuation_coefficient),
          substellar_incoming_radiation(substellar_incoming_radiation) {}

    void apply_forcing(const int j, const int k, const Real dt, MeshBlock *pmb,
        AthenaArray<Real> const &w, AthenaArray<Real> &u) const {

      const Real cos_zenith_angle = std::cos(pmb->pcoord->x2v(j));

      if (cos_zenith_angle < 0) {
        return;
      }

      const Real inv_cos_zenith_angle = 1./cos_zenith_angle;
      Real flux_r = substellar_incoming_radiation * cos_zenith_angle;
      Real flux_l;

      for (int i = pmb->ie; i >= pmb->is; --i) {
        const Real dz = pmb->pcoord->dx1f(i);
        const Real density = w(IDN, k, j, i) * w(i_species, k, j, i);
        const Real dtau = (
            mass_attenuation_coefficient * inv_cos_zenith_angle
            * density * dz
        );
        flux_l = flux_r * std::exp(-dtau);
        u(IEN, k, j, i) += dt * (flux_l - flux_r) / dz;
        flux_r = flux_l;
      }
    }

  private:
    const int i_species;
    const Real mass_attenuation_coefficient;
    const Real substellar_incoming_radiation;
};

// ==========================================================
// VaporCondensation class
// ==========================================================
template<class Real>
class VaporCondensation {
 public:
  const Real gas_constant, gamma;
  const Real Aeq, Beq;

  VaporCondensation(Real gas_constant, Real gamma,
                    Real Aeq, Real Beq)
      : gas_constant(gas_constant), gamma(gamma),
      Aeq(Aeq), Beq(Beq) {}

  static auto SiOVaporCondensation(void) {
    return VaporCondensation<Real>(
      SiO_VAPOR_GAS_CONST, SiO_VAPOR_ADIABATIC_INDEX,
      SiO_AEQ, SiO_BEQ);
  }

  template<class R>
  inline auto p_eq(const R &temp) const {
    return Aeq * exp(-Beq / temp);
  }

  template<class R>
  inline auto one_side_vapor_flux(const R &temp) const {
    return one_side_vapor_flux(temp, p_eq(temp));
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
inline Real surface_temperature(Real theta) {
  Real c = std::max(std::cos(theta), 0.);
  Real val = surface_temperature_max * std::pow(c, 0.25);
  return std::max(val, surface_temperature_min);
}


// ==========================================================
// Bottom Injection Function
// ==========================================================
void BottomInjection(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();
  auto vapor_cond = VaporCondensation<Real>::SiOVaporCondensation();
  const auto i_species = pthermo->SpeciesIndex("SiO");

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real radius = pmb->pmy_mesh->mesh_size.x1min;
        const Real z = pmb->pcoord->x1v(i) - radius;
        const Real dz = pmb->pcoord->dx1f(i);
        const Real delta = z < dz ? 1. / dz : 0.;
        if (delta > 0) {
          std::cout << "Bottom: i = " << i << std::endl;
          const Real t_surface = surface_temperature(pmb->pcoord->x2v(j));
          const Real t_air = pthermo->GetTemp(w.at(k, j, i));
          const Real gas_constant = (pthermo->GetRd()
            * pthermo->GetInvMuRatio(i_species));
          const Real cv = (pthermo->GetRd()
            * pthermo->GetCvRatio(i_species) / (pthermo->GetGammad() - 1.0));
          const Real p_vapor = (w(IDN, k, j, i) * w(i_species, k, j, i)
                            * t_air * gas_constant);
          const Real drho= (
            dt * delta
            * vapor_cond.net_vapor_flux(t_surface, t_air, p_vapor)
          );
          const Real t_exchange = drho > 0 ? t_surface : t_air;

          u(i_species, k, j, i) += drho;
          u(IEN, k, j, i) += drho * (cv + gas_constant) * t_exchange;
          if (drho < 0) {
            u(IVX, k, j, i) += drho * w(IVX, k, j, i);
            u(IVY, k, j, i) += drho * w(IVY, k, j, i);
            u(IVZ, k, j, i) += drho * w(IVZ, k, j, i);
            u(IEN, k, j, i) += drho * 0.5 * (
              square(w(IVX, k, j, i))
              + square(w(IVY, k, j, i))
              + square(w(IVZ, k, j, i))
            );
          }
        }
      }
    }
  }
}

void Gravity(MeshBlock *pmb, Real const time, Real const dt,
                     AthenaArray<Real> const &w, AthenaArray<Real> const &r,
                     AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
                     AthenaArray<Real> &s) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real radius = pmb->pmy_mesh->mesh_size.x1min;
        // const Real grav = (
        //   surface_grav * square(radius / pmb->pcoord->x1v(i))
        //   - 3 * square(planet_rotation) * pmb->pcoord->x1v(i)
        // );
        const Real grav = surface_grav;
        const Real src = -grav * dt * w(IDN, k, j, i);
        u(IVX, k, j, i) += src;
        u(IEN, k, j, i) += src * w(IVX, k, j, i);
      }
    }
  }
}

void RadiativeTransfer(MeshBlock *pmb, Real const time, Real const dt,
             AthenaArray<Real> const &w, AthenaArray<Real> const &r,
             AthenaArray<Real> const &bcc, AthenaArray<Real> &u,
             AthenaArray<Real> &s) {
  auto pthermo = Thermodynamics::GetInstance();
  const int i_species = pthermo->SpeciesIndex("SiO");
  const OneStreamRadiativeTransfer<Real> rt(i_species,
    mass_attenuation_coefficient,
    substellar_incoming_radiation);
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      rt.apply_forcing(j, k, dt, pmb, w, u);
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
  Gravity(pmb, time, dt, w, r, bcc, u, s);
  // RadiativeTransfer(pmb, time, dt, w, r, bcc, u, s);
}


// ==========================================================
// Mesh Initialization
// ==========================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();

  surface_temperature_min = pin->GetReal("problem", "surface_temperature_min");
  surface_temperature_max = pin->GetReal("problem", "surface_temperature_max");
  surface_grav = pin->GetReal("problem", "surface_grav");
  planet_rotation = pin->GetReal("problem", "planet_rotation");
  mass_attenuation_coefficient = pin->GetReal("problem", "mass_attenuation_coefficient");
  substellar_incoming_radiation = pin->GetReal("problem", "substellar_incoming_radiation");

  EnrollUserExplicitSourceFunction(Forcing);
}


// ==========================================================
// Problem Generator
// ==========================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  auto pthermo = Thermodynamics::GetInstance();
  const int iSiO = pthermo->SpeciesIndex("SiO");
  const int iSiOc = pthermo->SpeciesIndex("SiO(s)");

  const Real radius = pmy_mesh->mesh_size.x1min;
  const Real p_surface = pin->GetReal("initialcondition", "surface_pres");
  const Real temperature = pin->GetReal("initialcondition", "temperature");

  Real yfrac[IVX];
  yfrac[iSiO] = pin->GetReal("initialcondition", "SiO_massfrac");
  yfrac[iSiOc] = pin->GetReal("initialcondition", "SiOc_massfrac");
  yfrac[0] = 1. - yfrac[iSiO] - yfrac[iSiOc];

  pthermo->SetMassFractions<Real>(yfrac);

  const Real gas_constant = pthermo->GetRd() * (
    yfrac[0]
    + yfrac[iSiO] * pthermo->GetInvMuRatio(iSiO)
  );

  // const Real inv_scale_height = surface_grav / (gas_constant * temperature);
  const Real inv_scale_height = 0.;

  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is; i <= ie; ++i) {

        const Real z = pcoord->x1v(i) - radius;
        const Real pres = p_surface * std::exp(
          -inv_scale_height * radius * z / (radius + z)
        );
        const Real rho = pres / (gas_constant * temperature);

        phydro->w(IDN, k, j, i) = rho;
        phydro->w(iSiO, k, j, i) = yfrac[iSiO];
        phydro->w(iSiOc, k, j, i) = yfrac[iSiOc];
        phydro->w(IPR, k, j, i) = pres;
      }

  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u,
                             pcoord, is, ie, js, je, ks, ke);
}
