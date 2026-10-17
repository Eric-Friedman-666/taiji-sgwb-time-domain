import hashlib

import numpy as np
from py_code.triangle_code import *
from py_code.parameters import *
from tqdm import tqdm

_RESPONSE_FUNCTION_CACHE = {}
_TRANSFER_FUNCTION_CACHE = {}


def _frequency_cache_key(frequency):
    frequency = np.ascontiguousarray(np.asarray(frequency, dtype=float))
    digest = hashlib.sha1(frequency.view(np.uint8)).hexdigest()
    return frequency.shape, frequency.dtype.str, digest


def calculate_response_function(frequency, seg_index, orbit_type='unequal', generation_index=2, show_progress=False, use_cache=True):
    frequency = np.asarray(frequency, dtype=float)
    cache_key = (seg_index, orbit_type, generation_index, _frequency_cache_key(frequency))
    if use_cache and cache_key in _RESPONSE_FUNCTION_CACHE:
        return _RESPONSE_FUNCTION_CACHE[cache_key]

    orbit_time = (10 * (seg_index - 1) + 5) * DAY
    if orbit_type == 'unequal':
        orbit = Orbit(OrbitDir='./data/OrbitData/MicroSateOrbitEclipticTCB/', pn_order=2)
    else:
        orbit = Orbit(OrbitDir='./data/OrbitData/TaijiEqualArmOrbit/', pn_order=0)

    Ri_test = {}
    nij_test = {}
    dij_test = {}
    for key in SC_labels:
        Ri_test[key] = orbit.Positionfunctions()[key](orbit_time)
    for key in MOSA_labels:
        nij_test[key] = orbit.ArmVectorfunctions()[key](orbit_time)
        dij_test[key] = orbit.LTTfunctions()[key](orbit_time)

    sens = TDISensitivity(Ri=Ri_test, nij=nij_test, dij=dij_test)
    if generation_index == 2:
        A2_string, E2_string, T2_string = TDIStringManipulation.AETStringsfromXString(TDI.X2_string)
        A_operator = sens.TDI_P_ij(P_ij_strings=A2_string, f=frequency)
        E_operator = sens.TDI_P_ij(P_ij_strings=E2_string, f=frequency)
        T_operator = sens.TDI_P_ij(P_ij_strings=T2_string, f=frequency)
    elif generation_index == 1:
        A1_string, E1_string, T1_string = TDIStringManipulation.AETStringsfromXString(TDI.X1_string)
        A_operator = sens.TDI_P_ij(P_ij_strings=A1_string, f=frequency)
        E_operator = sens.TDI_P_ij(P_ij_strings=E1_string, f=frequency)
        T_operator = sens.TDI_P_ij(P_ij_strings=T1_string, f=frequency)
    else:
        raise ValueError(f'Unsupported generation_index: {generation_index}')

    RA = []
    RE = []
    RT = []
    for _ in tqdm(range(2**10), disable=not show_progress, leave=False):
        longitude = np.random.uniform(0, TWOPI)
        latitude = np.arcsin(np.random.uniform(-1, 1))
        RA.append(sens.TDI_response_function(lam=longitude, beta=latitude, f=frequency, P_ij=A_operator))
        RE.append(sens.TDI_response_function(lam=longitude, beta=latitude, f=frequency, P_ij=E_operator))
        RT.append(sens.TDI_response_function(lam=longitude, beta=latitude, f=frequency, P_ij=T_operator))

    response = (
        np.mean(np.array(RA), axis=0) / unit_factor(frequency) * 2,
        np.mean(np.array(RE), axis=0) / unit_factor(frequency) * 2,
        np.mean(np.array(RT), axis=0) / unit_factor(frequency) * 2,
    )
    if use_cache:
        _RESPONSE_FUNCTION_CACHE[cache_key] = response
    return response


def calculate_transfer_function(frequency, seg_index, orbit_type='unequal', use_cache=True):
    frequency = np.asarray(frequency, dtype=float)
    cache_key = (seg_index, orbit_type, _frequency_cache_key(frequency))
    if use_cache and cache_key in _TRANSFER_FUNCTION_CACHE:
        return _TRANSFER_FUNCTION_CACHE[cache_key]

    orbit_time = (10 * (seg_index - 1) + 5) * DAY
    if orbit_type == 'equal':
        orbit = Orbit(OrbitDir='./data/OrbitData/TaijiEqualArmOrbit/', pn_order=0)
    elif orbit_type == 'unequal':
        orbit = Orbit(OrbitDir='./data/OrbitData/MicroSateOrbitEclipticTCB/', pn_order=2)
    else:
        raise ValueError(f'Unsupported orbit_type: {orbit_type}')

    Ri_test = {}
    nij_test = {}
    dij_test = {}
    for key in SC_labels:
        Ri_test[key] = orbit.Positionfunctions()[key](orbit_time)
    for key in MOSA_labels:
        nij_test[key] = orbit.ArmVectorfunctions()[key](orbit_time)
        dij_test[key] = orbit.LTTfunctions()[key](orbit_time)

    noise_psd = TDIPSDs(sacc=1, sro=0, L=L_nominal)
    acc_transfer_A = noise_psd.PSD_A2_unequal(frequency, dij_test) / unit_factor(frequency)
    acc_transfer_E = noise_psd.PSD_E2_unequal(frequency, dij_test) / unit_factor(frequency)
    acc_transfer_T = noise_psd.PSD_T2_unequal(frequency, dij_test) / unit_factor(frequency)

    noise_psd = TDIPSDs(sacc=0, sro=1, L=L_nominal)
    oms_transfer_A = noise_psd.PSD_A2_unequal(frequency, dij_test) / unit_factor(frequency)
    oms_transfer_E = noise_psd.PSD_E2_unequal(frequency, dij_test) / unit_factor(frequency)
    oms_transfer_T = noise_psd.PSD_T2_unequal(frequency, dij_test) / unit_factor(frequency)

    transfer = (
        acc_transfer_A,
        oms_transfer_A,
        acc_transfer_E,
        oms_transfer_E,
        acc_transfer_T,
        oms_transfer_T,
    )
    if use_cache:
        _TRANSFER_FUNCTION_CACHE[cache_key] = transfer
    return transfer

def NA2(parameters, seg_index, acc_transfer_A_table, oms_transfer_A_table):
    seg_index = seg_index % 36
    N_acc_amp = parameters[0]
    delta_x_amp = parameters[1]
    N_acc = N_acc_amp * 1e-15
    delta_x = delta_x_amp * 1e-12
    acc_transfer_A = acc_transfer_A_table[seg_index - 1]
    oms_transfer_A = oms_transfer_A_table[seg_index - 1]
    return (N_acc**2) * acc_transfer_A + (delta_x**2) * oms_transfer_A
def NE2(parameters, seg_index, acc_transfer_E_table, oms_transfer_E_table):
    seg_index = seg_index % 36
    N_acc_amp = parameters[0]
    delta_x_amp = parameters[1]
    N_acc = N_acc_amp * 1e-15
    delta_x = delta_x_amp * 1e-12
    acc_transfer_E = acc_transfer_E_table[seg_index - 1]
    oms_transfer_E = oms_transfer_E_table[seg_index - 1]
    return (N_acc**2) * acc_transfer_E + (delta_x**2) * oms_transfer_E
def NT2(parameters, seg_index, acc_transfer_T_table, oms_transfer_T_table):
    seg_index = seg_index % 36
    N_acc_amp = parameters[0]
    delta_x_amp = parameters[1]
    N_acc = N_acc_amp * 1e-15
    delta_x = delta_x_amp * 1e-12
    acc_transfer_T = acc_transfer_T_table[seg_index - 1]
    oms_transfer_T = oms_transfer_T_table[seg_index - 1]
    return (N_acc**2) * acc_transfer_T + (delta_x**2) * oms_transfer_T



def W(f):
    return 1 - np.exp(-2j * f / fstar)
def WW(f):
    return 1 - np.exp(-4j * f / fstar)
def R_AE(f,generation_index,fit=False):
    R1 = 4 * 9/20 * np.abs(W(f))**2 * (1 + (3 * f / (4 * fstar))**2) ** (-1) 
    if generation_index == 1:
        return R1
    elif generation_index == 2:
        return R1 * np.abs(WW(f))**2
    elif generation_index == 0 :
        return R1 / np.abs(W(f))**2
def R_T(f, generation_index, fit=False):
    R1 = 4 * 1/4032 * np.abs(W(f))**2 * (f/fstar)**6 * (1 + 5/16128 * (f/fstar)**8) ** (-1) 
    if generation_index == 1:
        return R1
    elif generation_index == 2:
        return R1 * np.abs(WW(f))**2

def read_seg_ltt(LTT_dict,ith_seg,segments=36):
    mosa_label = ["12", "13", "23", "21", "31", "32"] 
    seg_ltt_dict = {}
    for label in mosa_label:
        seg_ltt_dict[label] = np.mean(np.array(np.array_split(LTT_dict[label],segments))[ith_seg])
    return seg_ltt_dict 

def S_acc(f, paramters):
    return (10**paramters[0] * np.sqrt(1 + (0.0004 / f)**2) * np.sqrt(1 + (f / 0.008)**4))**2
def S_oms(f, parameters):
    return (10**parameters[1] * np.sqrt(1 + (0.002 / f)**4))**2
def N_1(f, parameters, generation_index=1):
    return (1 / L**2) * (4 * S_oms(f, parameters) + 8 * (1 + np.cos(f / fstar)**2) * S_acc(f, parameters) / (2 * np.pi * f)**4) * np.abs(W(f))**2 * np.abs(WW(f))**(2*generation_index-2)
def N_2(f, parameters, generation_index=1):
    return (-1 / L**2) * (2 * S_oms(f, parameters) + 8 * S_acc(f, parameters) / (2 * np.pi * f)**4) * np.cos(f / fstar) * np.abs(W(f))**2 * np.abs(WW(f))**(2*generation_index-2)
def N_A(f, parameters, generation_index=1):
    return N_1(f, parameters, generation_index) - N_2(f, parameters, generation_index)
N_E = N_A  
def N_T(f, parameters, generation_index=1):
    return N_1(f, parameters, generation_index) + 2 * N_2(f, parameters, generation_index) 

def PL(f, log_Omega, alpha): 
    return 10**(log_Omega) * (f / (1e-3)) ** alpha
def dwd(f,log_A1,alpha1,log_A2,alpha2, f0=0.016):
    return 10**(log_A1)*(f/f0)**alpha1 /(1 + 10**(log_A2)*(f/f0)**alpha2)
def sw(f,log_Omega_0,log_fp):
    return 10**log_Omega_0 * (f/(10**log_fp))**3 * (7 / (4 + 3 * (f/10**(log_fp))**2))**(7/2)
def S_h_ast(f, log_Omega_ast, alpha_ast):
    Omega = PL(f, log_Omega_ast, alpha_ast) 
    return 3 * H0**2 / (4 * np.pi**2) * Omega / f**3 
def S_h_dwd_ast(f, parameters): 
    log_A1 = parameters[2]
    alpha1 = parameters[3]
    log_A2 = parameters[4]
    alpha2 = parameters[5]
    log_Omega_ast = parameters[6]
    alpha_ast = parameters[7]
    Omega = dwd(f,log_A1,alpha1,log_A2,alpha2) + PL(f, log_Omega_ast, alpha_ast)
    return 3 * H0**2 / (4 * np.pi**2) * Omega / f**3
def S_h_dwd_ast_sw(f, parameters):
    log_A1 = parameters[2]
    alpha1 = parameters[3]
    log_A2 = parameters[4]
    alpha2 = parameters[5]
    log_Omega_ast = parameters[6]
    alpha_ast = parameters[7]
    log_Omega_0 = parameters[8]
    log_fp = parameters[9]
    Omega = dwd(f,log_A1,alpha1,log_A2,alpha2) + PL(f, log_Omega_ast, alpha_ast)+ sw(f,log_Omega_0,log_fp)
    return 3 * H0**2 / (4 * np.pi**2) * Omega / f**3
def S_h_dwd(f, parameters): 
    log_A1 = parameters[2]
    alpha1 = parameters[3]
    log_A2 = parameters[4]
    alpha2 = parameters[5]
    Omega = dwd(f,log_A1,alpha1,log_A2,alpha2)
    return 3 * H0**2 / (4 * np.pi**2) * Omega / f**3








