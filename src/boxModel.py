"""
Created: November 2026 by Camden Opfer

A simple box model with one box each for the atmosphere, surface, and land. Parameters are intended to be observable by telescopes, so they include stellar insolation at the TOA of the planet, radius of the exoplanet, and atmospheric composition. While ocean fraction and atmospheric thickness may be hard to measure, they are extremely important to this kind of energy budget model. If applying this model in a exoplanet hunt, it may be worth varying those two variables to see if there's any configuration in which the observable parameters could correspond to a habitable planet.
"""
import os
import functools
import numpy as np
rand = np.random.rand
from numpy.typing import ArrayLike
import hapi

## Model Parameters
# Earth-like parameters
earth_s0 = 1361/4 #W m^-2
earth_sT = 5777 #K Temperature of the Sun
earth_r = 6.371E6 #m Authalic radius of Earth (gives the right surface area), taken from https://link.springer.com/content/pdf/10.1007/s001900050278.pdf
earth_M = 5.972E24 #kg Mass of Earth, taken from https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
earth_fracOcn = 0.708 #Dimenstionless Fraction of Earth covered in ocean
earth_co2 = 3.9E-4 #kgkg^-1 Atmospheric CO2 concentration (ish)
earth_pSurf = 1 #atm Mean sea level pressure

# Parameter ranges
range_s0 = (1E2, 1E4) #W m^-2
range_sT = (1E3, 1E5) #K
range_r = (1E5, 1E8) #m
range_M = (1E18, 1E30) #kg
range_fracOcn = (0, 1) #Dimensionless
range_co2 = (0, 1E-3) #ppm atmospheric CO2
range_pSurf = (0.7, 1.3) #atm Surface pressure

# Initial conditions (Earth-like)
# TODO Determine what actually happens for Earth
init_Tatm = 200
init_Tsurf = 250
init_Tocn = 250

init_Qatm = 1E-3 #kg kg^-1 Concentration of atmospheric water vapour
init_Qsurf = 0 # Land moisture TODO Figure out what variety of moisture to use for surface

## Universal Constants
c = 2.99E8 #m s^-1 Speed of light in a vacuum
k_b = 1.381E-23 #J K^-1 Boltzmann constant
h = 6.626E-34 #J Hz^-1 Planck constant
sigma = 5.6704E-8 #Wm^-2 K^-4 Stefan-Boltzman constant
G = 6.674E-11 #m^3 kg^-1 s^-2 Newtonian constant of gravitation
R_d = 287 #J kg^-1 K^-1 Ideal gas constant for dry air NOTE This assumes a vaguely Earth-like atmosphere
paToAtm = 101325 #Pa atm^-1 Conversion factor

## Case setup
def monteCarloParams(n:int = 1) -> dict|ArrayLike:
    """
    Makes n random sets of parameters describing planets, with each variable constrained by the range set above. Some are randomly sampled following a uniform distribution, while others are distributed from a logarithmic distribution to add more meaningful variety.
    """
    def linearSample(range_var):
        diff_var = range_var[1] - range_var[0]
        return rand(n) * diff_var + range_var[0]

    def logSample(range_var):
        logMin_var = np.log(range_var[0])
        logDiff_var = np.log(range_var[1]) - logMin_var
        return np.exp(rand(n) * logDiff_var + logMin_var)

    s0 = logSample(range_s0)
    r = logSample(range_r)
    M = logSample(range_M)
    A = np.pi * r**2
    fracOcn = linearSample(range_fracOcn)
    A_ocn, A_lnd = A * fracOcn, A * (1 - fracOcn)
    co2 = logSample(range_co2)
    pSurf = linearSample(range_pSurf)

    if n > 1:
        def hstackPrep(arrIn):
            return np.transpose(np.atleast_2d(arrIn))

        bigOutArr = np.hstack((hstackPrep(s0), hstackPrep(A_ocn), hstackPrep(A_lnd), hstackPrep(co2), hstackPrep(pSurf)))
        return bigOutArr

    return {'s0':s0, 'A_ocn':A_ocn, 'A_lnd':A_lnd, 'r':r, 'M':M, 'co2':co2, 'pSurf':pSurf}

def makeLayers(pSurf:float, nLayers:int=4, pFactor:float=0.75) -> ArrayLike[float]:
    """
    Makes nLayers layers spaced in pressure by a factor of pFactor, which must be in the range (0,1). Returns the thickness and mean pressure of each layer.
    """
    ps = pSurf * np.logspace(0, nLayers, nLayers+1, base=pFactor) #atm Pressure at layer bounds. Sequence is [pSurf, 0.75 * pSurf, 0.75**2 * pSurf, ...]. Goes for nLayers+1 elements

    return ps

## Radiation
def surfGravity(M:float|ArrayLike[float], r:float|ArrayLike[float]) -> float|ArrayLike[float]:
    """
    Get's surface gravity based on planetary mass and the gravitational shell theorem (g = GM/r**2). Used in the hydrostatic equation when determining atmospheric thickness.
    """
    return G * M/r**2

def layerProperties(pBot:float|ArrayLike[float], pTop:float|ArrayLike[float], T:float|ArrayLike[float], g:float) -> tuple[float|ArrayLike[float], float|ArrayLike[float]]:
    """
    Uses the hydrostatic equation (dP/dz = -rho g) to determine the vertical thickness of atmosphere between two pressure levels. Input should satisfy p2 < p1 so that p2 is higher in the column, otherwise a negative thickness will be provided. Also returns
    """
    pBot_pa, pTop_pa = pBot * paToAtm, pTop * paToAtm #Pa from atm
    pMid = (pBot_pa + pTop_pa) / 2 #Pa
    rhoMid = pMid / (R_d * T) #kg m^-3
    Ls = -rhoMid * g / (pTop_pa - pBot_pa) #m
    return Ls, pMid/paToAtm

def plancksLaw(T:float|ArrayLike[float], wavelength:float|ArrayLike[float]) -> float|ArrayLike[float]:
    """
    Computes Planck's law to give the amount of radiation at a given wavelength. See https://topex.ucsd.edu/rs/radiation.pdf
    """
    B = 2 * h * c**2 / (wavelength**5 * np.exp(h * c/(wavelength * k_b * T) - 1)) #W m^-3 sr^-1 Spectral radiance
    return B

# Set global objects that have more meaningful values assigned by hapiInit()
hapiStarted = False
def getAbsCoeffs(): return None

def hapiInit(nu_min:int|float=100, nu_max:int|float=1E5, nu_step:int=10):
    """
    Fetches relevant data from the HITRAN API (HAPI) and prepares 
    """
    hapi.db_begin(os.path.join('~', 'hitran-api_data'))
    hapi.fetch('H2O',1,1,nu_min,nu_max)
    hapi.fetch('CO2',1,1,nu_min,nu_max)

    global getAbsCoeffs
    getAbsCoeffs = functools.partial(hapi.absorptionCoefficient_Lorentz, WavenumberRange=(nu_min, nu_max), WavenumberStep=nu_step, HITRAN_units=False) # HITRAN_units=False means coefficients will be in cm^-1, as required by hapi.absorptionspectrum

    global hapiStarted
    hapiStarted = True

def netAbs(T_emission:float, T_atm:float, P_atm:float, L:float, co2:float, Q:float) -> tuple[float, float]:
    """
    Calculates the absorption coefficient for a layer of the atmosphere.
    """
    if not hapiStarted:
        hapiInit()

    environment_h2o = {"T": T_atm, "p": P_atm * Q}
    environment_co2 = {"T": T_atm, "p": P_atm * co2}

    # Get absorption coefficients in cm^-1
    nu, absCoeff_h2o = getAbsCoeffs(SourceTables="H2O", Environment=environment_h2o)
    _, absCoeff_co2 = getAbsCoeffs(SourceTables="CO2", Environment=environment_co2)

    # Apply physical absorption coefficients to layer of thickness L
    _, abs_h2o = hapi.absorptionSpectrum(nu, absCoeff_h2o, Environment={'l':L})
    _, abs_co2 = hapi.absorptionSpectrum(nu, absCoeff_co2, Environment={'l':L})

    B_incoming = plancksLaw(T_emission, 1/nu) #W m^-3 sr^-1 Spectral radiance

    A = B_incoming * (abs_h2o + abs_co2) / np.sum(B_incoming) #Dimensionless Normalized absorption amount, will fall in range [0.1]

    return A

def absFromComposition(ps:ArrayLike[float], Ts:ArrayLike[float], T_surf:float, T_stellar:float, co2:float, Q:float, g:float):
    """
    Gets the absorption coefficient of both short- and longwave radiation for each layer of the atmosphere.
    """
    Ls, pMids = layerProperties(ps[:-1], ps[1:], Ts, g)

    absorption = np.zeros((2, len(Ts))) # Rows are long- and shortwave radiation respectively. Collumns are layer.
    for i, T_emission in enumerate([T_surf, T_stellar]):
        for j, (T_atm, P_atm, L) in enumerate(zip(Ts, pMids, Ls)):
            absorption[i,j] = netAbs(T_emission, T_atm, P_atm, L, co2, Q) #Dimensionless Normalized absorption amount, will fall in range [0.1]

    return absorption

def boxModelIter(params:dict, init:dict, outFile:str|None = None):
    """
    Iterates forward following the system of differential equations, optionally saving the model state to the provided file path.
    """
    # TODO This whole function. So, arguably, the actual model
    # A very nice website that walks through a simpler version: https://www.atmos.albany.edu/facstaff/brose/classes/ATM623_Spring2015/Notes/Lectures/Lecture06%20--%20Elementary%20greenhouse%20models.html
