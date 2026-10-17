import sys, os
from pathlib import Path
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PACKAGE_ROOT / "vendor/Triangle-Simulator"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  
from py_code.model import *
from py_code.parameters import *
import numpy as np
import matplotlib.pyplot as plt
import logging
from py_code.model import *
from Triangle.Constants import *
from Triangle.FFTTools import *
from Triangle.Noise import *
from Triangle.Orbit import *
from Triangle.Data import *
from Triangle.Interferometer import *
from Triangle.TDI import *
from Triangle.Glitch import *
from Triangle.GW import *
import h5py
import healpy as hlp
from tqdm import tqdm

# Triangle functions accept a pool argument, but creating a multiprocessing.Pool
# at import time can leak workers or hang notebook/cluster jobs. Keep the default
# serial path explicit; launch scripts can pass their own pool if needed.
pool = None
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO) 

def unit_factor(f):
    return (L/c * 2*np.pi*f)**2

def generate_TDI_data(noise, signal_type, day_number, fsample,orbit_type="unequal", orbit_dir=None, parameters=None):
    include_instrumental_noise = bool(noise)
    if orbit_dir is None:
        if orbit_type == "unequal":
            orbit_dir = str(_PACKAGE_ROOT / "data/OrbitData/MicroSateOrbitEclipticTCB")
            orbit = Orbit(OrbitDir=orbit_dir, pn_order = 2) 
        elif orbit_type == "equal":
            orbit_dir = str(_PACKAGE_ROOT / "data/OrbitData/TaijiEqualArmOrbit")
            orbit = Orbit(OrbitDir=orbit_dir, pn_order = 0) 
    else:
        if orbit_type == "unequal":
            orbit = Orbit(OrbitDir=orbit_dir, pn_order = 2)
        elif orbit_type == "equal":
            orbit = Orbit(OrbitDir=orbit_dir, pn_order = 0)
    data_time = day_number * DAY
    size = int(data_time * fsample) 
    factor = data_time * fsample**2/2
    interp_order = 31
    default_noise = True
    TDI_method = "fast_michelson" 
    offset = OffsetFunctions(laser_flag=False, clock_flag=False)
    instrumental_noise = InstrumentalPSDs(L=L_nominal) 
    def PL(f, log_Omega, alpha):
        return 10**(log_Omega) * (f / (1e-3)) ** alpha
    def dwd(f,log_A1,alpha1,log_A2,alpha2, f0=0.016):
        return 10**(log_A1)*(f/f0)**alpha1 /(1 + 10**(log_A2)*(f/f0)**alpha2)
    def sw(f,log_Omega_0,log_fp):
        return 10**log_Omega_0 * (f/(10**log_fp))**3 * (7 / (4 + 3 * (f/10**(log_fp))**2))**(7/2)
    def S_SGWB(f):
        if signal_type == "response":
            return f*0+1.0 
        if signal_type == "noise+ast":
            log_Omega_ast = parameters[2]
            alpha_ast = parameters[3]
            Omega = PL(f, log_Omega_ast, alpha_ast)
            return 3.0 * H0**2 / 4.0 / np.pi**2 / f**3 * Omega * 1e6 / MPC**2 
        if signal_type == "noise+dwd+ast":
            log_A1 = parameters[2]
            alpha1 = parameters[3]
            log_A2 = parameters[4]
            alpha2 = parameters[5]
            log_Omega_ast = parameters[6]
            alpha_ast = parameters[7]
            Omega_dwd = dwd(f, log_A1, alpha1, log_A2, alpha2)
            Omega_ast = PL(f, log_Omega_ast, alpha_ast)
            Omega = Omega_dwd + Omega_ast
            return 3.0 * H0**2 / 4.0 / np.pi**2 / f**3 * Omega * 1e6 / MPC**2
        if signal_type == "noise+dwd+ast+sw":
            log_A1 = parameters[2]
            alpha1 = parameters[3]
            log_A2 = parameters[4]
            alpha2 = parameters[5]
            log_Omega_ast = parameters[6]
            alpha_ast = parameters[7]
            log_Omega_0 = parameters[8]
            log_fp = parameters[9]
            Omega_dwd = dwd(f, log_A1, alpha1, log_A2, alpha2)
            Omega_ast = PL(f, log_Omega_ast, alpha_ast)
            Omega_sw = sw(f, log_Omega_0, log_fp)
            Omega = Omega_dwd + Omega_ast + Omega_sw
            return 3.0 * H0**2 / 4.0 / np.pi**2 / f**3 * Omega * 1e6 / MPC**2
    NSIDE = 5 
    NPIX = hlp.nside2npix(NSIDE) 
    colatitude, longitude = hlp.pix2ang(nside=NSIDE, ipix=np.arange(NPIX))
    latitude = np.pi / 2.0 - colatitude
    tdata = np.arange(size) / fsample
    sgwb_generator = GeneralNoise(S_SGWB) 
    if signal_type is not None :
        gw = [] 
        for i in range(NPIX):
            hp_k = sgwb_generator(fsample=fsample, size=size) / np.sqrt(NPIX)
            hc_k = sgwb_generator(fsample=fsample, size=size) / np.sqrt(NPIX)
            Waveform_k = GeneralWaveform(tdata=tdata, hpdata=hp_k, hcdata=hc_k)
            GW_k = GW(orbit=orbit, ext_params=[longitude[i], latitude[i], 0.0], GWwaveform=Waveform_k)  
            gw.append(GW_k) 
    elif signal_type == None:
        gw = None
    ifo = Interferometers(
        size=size,
        fsample=fsample,
        time_frame="ProperTime",
        acc_noise=include_instrumental_noise,
        ro_noise=include_instrumental_noise, 
        laser_noise=False,
        clock_noise=False,
        ob_noise=False,
        bl_noise=False,
        op_noise=False, 
        ranging_noise=False, 
        modulation_noise=False, 
        orbit_class=orbit,
        offset_class=offset,
        noise_class=instrumental_noise,
        gw_class=gw, 
        fplan=None,
        order=interp_order,
        pool=pool,
        clean_memory=True,  
    ) 
    ifo.SimulateInterferometers()
    m = ifo.OutputMeasurements(mode="decomposed")
    m1 = TPStoTCB(m, orbit_class=orbit, order=interp_order, pool=pool)
    ifo.clean() 
    tdi_interp_order = 31 
    tdi = TDI(measurements=m1, delays=m1["ltt"], fsample=m1["fsample"], order=tdi_interp_order)
    tdi.CalculateXi(doppler=True, pool=pool)
    tdi.CalculateEta(doppler=True)
    tdi.FastMichelson(doppler=True, channel="AET")
    return tdi

def get_PSD_data_from_TDI(tdi,tdi_generation):
    if tdi_generation == "1st":
        T = tdi.measurements['time']['1'][-1]
        fs = tdi.measurements['fsample']
        factor = T * fs**2 / 2
        ff, xf = PSD_window(tdi.measurements["A1"][100:-100] / F_LASER, fsample=fs, nbin=1, window_type="hann")
        psd_A_data = xf/unit_factor(ff)
        ff, xf = PSD_window(tdi.measurements["E1"][100:-100] / F_LASER, fsample=fs, nbin=1, window_type="hann")
        psd_E_data = xf/unit_factor(ff)
        ff, xf = PSD_window(tdi.measurements["T1"][100:-100] / F_LASER, fsample=fs, nbin=1, window_type="hann")
        psd_T_data = xf/unit_factor(ff)
        psd_data = np.array([ff, psd_A_data, psd_E_data, psd_T_data])
        d_2 = np.array([ff, factor*psd_A_data, factor*psd_E_data, factor*psd_T_data])
    elif tdi_generation == "2nd":
        T = tdi.measurements['time']['1'][-1]
        fs = tdi.measurements['fsample']
        factor = T * fs**2 / 2
        ff, xf = PSD_window(tdi.measurements["A2"][100:-100] / F_LASER, fsample=fs, nbin=1, window_type="hann")
        psd_A_data = xf/unit_factor(ff)
        ff, xf = PSD_window(tdi.measurements["E2"][100:-100] / F_LASER, fsample=fs, nbin=1, window_type="hann")
        psd_E_data = xf/unit_factor(ff)
        ff, xf = PSD_window(tdi.measurements["T2"][100:-100] / F_LASER, fsample=fs, nbin=1, window_type="hann")
        psd_T_data = xf/unit_factor(ff)
        psd_data = np.array([ff, psd_A_data, psd_E_data, psd_T_data])
        d_2 = np.array([ff, factor*psd_A_data, factor*psd_E_data, factor*psd_T_data])
    return psd_data, d_2

def CalculatePSD_SingleSegment(AET_data_TimeDomain, fsample, day_seg=10,window_type="hann"):
    A_data_TimeDomain = AET_data_TimeDomain[0]
    E_data_TimeDomain = AET_data_TimeDomain[1]
    T_data_TimeDomain = AET_data_TimeDomain[2]
    T = day_seg * DAY 
    fs = fsample
    factor = T * fs**2 / 2
    ff, xf = PSD_window(A_data_TimeDomain[100:-100] / F_LASER, fsample=fs, nbin=1, window_type=window_type)
    psd_A_data = xf/unit_factor(ff)
    ff, xf = PSD_window(E_data_TimeDomain[100:-100] / F_LASER, fsample=fs, nbin=1, window_type=window_type)
    psd_E_data = xf/unit_factor(ff)
    ff, xf = PSD_window(T_data_TimeDomain[100:-100] / F_LASER, fsample=fs, nbin=1, window_type=window_type)
    psd_T_data = xf/unit_factor(ff)
    psd_data = np.array([ff, psd_A_data, psd_E_data, psd_T_data])
    d_2 = np.array([ff, factor*psd_A_data, factor*psd_E_data, factor*psd_T_data])
    return psd_data, d_2

def calculate_response_function(frequency, seg_index, orbit_type = 'unequal'):
    orbit_time = (10*(seg_index-1) + 5) * DAY 
    if orbit_type == 'unequal':
        orbit = Orbit(OrbitDir='./data/OrbitData/MicroSateOrbitEclipticTCB/', pn_order=2) 
    else:
        orbit = Orbit(OrbitDir='./data/OrbitData/TaijiEqualArmOrbit/', pn_order=0)
    Ri_test = dict() 
    nij_test = dict()
    dij_test = dict()
    for key in SC_labels:
        Ri_test[key] = orbit.Positionfunctions()[key](orbit_time)
    for key in MOSA_labels:
        nij_test[key] = orbit.ArmVectorfunctions()[key](orbit_time)
        dij_test[key] = orbit.LTTfunctions()[key](orbit_time)
    sens = TDISensitivity(Ri=Ri_test, nij=nij_test, dij=dij_test)
    A2_string, E2_string, T2_string = TDIStringManipulation.AETStringsfromXString(TDI.X2_string)
    A2_operator = sens.TDI_P_ij(P_ij_strings=A2_string, f=frequency)
    E2_operator = sens.TDI_P_ij(P_ij_strings=E2_string, f=frequency)
    T2_operator = sens.TDI_P_ij(P_ij_strings=T2_string, f=frequency)
    RA = []
    RE = []
    RT = []
    for _ in range(2**10):
        longitude = np.random.uniform(0, TWOPI)
        latitude = np.arcsin(np.random.uniform(-1, 1))
        RA.append(sens.TDI_response_function(lam=longitude, beta=latitude, f=frequency, P_ij=A2_operator))
        RE.append(sens.TDI_response_function(lam=longitude, beta=latitude, f=frequency, P_ij=E2_operator))
        RT.append(sens.TDI_response_function(lam=longitude, beta=latitude, f=frequency, P_ij=T2_operator))
    RA = np.mean(np.array(RA), axis=0) / unit_factor(frequency) * 2
    RE = np.mean(np.array(RE), axis=0) / unit_factor(frequency) * 2
    RT = np.mean(np.array(RT), axis=0) / unit_factor(frequency) * 2
    return RA, RE, RT
