"""
Created: November 2026 by Camden Opfer

A simple box model with one box each for the atmosphere, surface, and land. Water and carbon dioxide are the two allowed greenhouse gasses, and water moves between boxes. Assuming a lack of existing life, it is unlikely that carbon could cycle between systems, so that is neglected.
"""
import os
import functools
import numpy as np
rand = np.random.rand
from numpy.typing import ArrayLike
import hapi

## Model Parameters
# Earth-like parameters
earth_s0 = 1361 #W m^-2
earth_r = 6371007.2 #m Authalic radius of Earth, taken from https://link.springer.com/content/pdf/10.1007/s001900050278.pdf
earth_fracOcn = 0.708
earth_co2 = 500 #ppm atmospheric CO2
earth_zAtm = 8500 #m atmospheric thickness

# Parameter ranges
range_s0 = (1E2, 1E4) #W m^-2
range_r = (1E5, 1E8) #m
range_fracOcn = (0, 1)
range_co2 = (0, 2000) #ppm atmospheric CO2
range_zAtm = (1E3, 1E4) #m atmospheric thickness

# Initial conditions (Earth-like)
# TODO Determine what actually happens for Earth
init_Tatm = 200
init_Tsurf = 250
init_Tocn = 250

init_Qatm = 0 #kg m^-3 Concentration of atmospheric water vapour
init_Qsurf = 0 # TODO Figure out what variety of moisture is used for surface

## Universal Constants
sigma = 5.6704E-8 #Wm^-2 K^-4 Stefan-Boltzman constant
N_A = 6.023E23 #molec mol^-1 Avogadro's Number
M_co2 = 0.044 #kg mol^-1 Molar mass of CO2
M_air = 0.02896 #kg mol^-1 Molecular mass of dry air NOTE This value is for Earth!!
rho_0 = 1.29 #kg m^-3 Density of atmosphere at the surface NOTE This value is for Earth!!

tau_sw_h20 = 0.98 #m^-1 Optical depth of water vapour for shortwave


def monteCarloParams(n:int = 1) -> ArrayLike:
    """
    Makes n random sets of parameters describing planets, with each variable constrained by the range set above.
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
    A = np.pi * r**2
    fracOcn = linearSample(range_fracOcn)
    A_ocn, A_lnd = A * fracOcn, A * (1 - fracOcn)
    co2 = logSample(range_co2)
    zAtm = logSample(range_zAtm)

    if n > 1:
        def hstackPrep(arrIn):
            return np.transpose(np.atleast_2d(arrIn))

        bigOutArr = np.hstack((hstackPrep(s0), hstackPrep(A_ocn), hstackPrep(A_lnd), hstackPrep(co2), hstackPrep(zAtm)))
        return bigOutArr

    return {'s0':s0, 'A_ocn':A_ocn, 'A_lnd':A_lnd, 'co2':co2, 'zAtm':zAtm}

hapiStarted = False
def getAbsCoeffs(): return None

def hapiInit(nu_min:int|float=100, nu_max:int|float=3000, nu_step:int=10):
    hapi.db_begin(os.path.join('~', 'hitran-api_data'))
    hapi.fetch('H2O',1,1,nu_min,nu_max)
    hapi.fetch('CO2',1,1,nu_min,nu_max)
    global getAbsCoeffs
    getAbsCoeffs = functools.partial(hapi.absorptionCoefficient_Lorentz, WavenumberRange=(nu_min, nu_max), WavenumberStep=nu_step, HITRAN_units=True)
    global hapiStarted
    hapiStarted = True

def longwaveAtm(T_surf:float, T_atm:float, L:int|float, co2) -> tuple[float, float]:
    """
    Calculates the absorbed and transmitted radiation from the surface. Returns fluxes in Wm^-2.
    """
    if not hapiStarted:
        hapiInit()

    P_surf = np.exp(L - earth_zAtm) #atm When L = earth_zAtm, P_atm = 1atm
    P_atm = P_surf * 0.7

    environment = {"T": T_atm, "p": P_atm}

    nu, absCoeff_h2o = getAbsCoeffs(SourceTables="H2O", Environment=environment)
    _, absCoeff_co2 = getAbsCoeffs(SourceTables="CO2", Environment=environment)

    k_spectral_co2 = absCoeff_co2/10000 * N_A/M_co2 #m^2 kg^-1 Conversion from cm^2 molec^-1 to more useful units for Beer's Law 
    w_co2 = co2 * 1E-6 * (M_co2/M_air) #kg kg^-1 Mixing ratio of CO2 in the atmosphere

    tau_star = w_co2 * k_spectral_co2 * (w_h2o * k_spectral_h2o) * rho_0 * L
    tau = tau_star # NOTE Assumes constant pressure through entire column

    transmitByWavenumber = np.exp(-tau)

def boxModelIter(params:dict, init:dict, outFile:str|None = None):
    """
    Iterates forward following the system of differential equations, optionally saving the model state to the provided file path.
    """
    A_tot = params['A_ocn'] + params['A_lnd'] # TODO Determine if this should be in the params dict

    sAtmSurf = params['s0'] * np.exp(-params['zAtm'] * tau_sw_h20 * init['Qatm']) #W m^-2 Following the Beer-Lambert Law
    sToaAtm = params['s0'] - sAtmSurf #Wm^-2 By conservation of energy

    SToaLnd = sAtmSurf * params['A_lnd'] #W
    SToaOcn = sAtmSurf * params['A_ocn'] #W

    SOcnAtm = sigma * init['T_ocn']**4 * params['A_ocn'] #W
    SLndAtm = sigma * init['T_lnd']**4 * params['A_lnd'] #W
