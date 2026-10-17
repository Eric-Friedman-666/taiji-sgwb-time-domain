import numpy as np
L = 3e9
H0 = 2.18429e-18  
c = 299792458
fref = 1e-3
fstar = c/(2*np.pi*L)
h = 0.674
vw = 0.63
cs = 1 / np.sqrt(3)  
gstar = 100

def unit_factor(f):
    return (L/c * 2*np.pi*f)**2

N_acc_amp = 3
delta_x_amp = 8

log_A1_true = np.log10(3.98e-16)
alpha1_true = -5.7
log_A2_true = np.log10(4.79e-7)
alpha2_true = -6.2 

log_Omega_ast_true = -11.5
alpha_ast_true = 2/3
log_Omega_0_true =  -11.5
log_fp_true = -2.25

para_range = dict({
    "N_acc_amp": [0, 20, 3],
    "delta_x_amp": [0, 20, 8],
    "log_A1": [-18, -12, -13],
    "alpha1": [-8, -2, -5],
    "log_A2": [-10, -2, -5],
    "alpha2": [-8, -2, -5],
    "log_Omega_ast": [-14, -9, -11],
    "alpha_ast": [-1, 3, 0.5],
    "log_Omega_0": [-13, -10, -11],
    "log_fp": [-3, -0.5, -2],
})

def input_parameters(model="noise"):
    if model == "noise":
        parameters_true = np.array([N_acc_amp, delta_x_amp])
    elif model == "noise+ast":
        parameters_true = np.array([N_acc_amp,
                                    delta_x_amp,
                                    log_Omega_ast_true,
                                    alpha_ast_true])
    elif model == "noise+dwd":
        parameters_true = np.array([N_acc_amp,
                                    delta_x_amp,
                                    log_A1_true,
                                    alpha1_true,
                                    log_A2_true,
                                    alpha2_true])
    elif model == "noise+dwd+ast":
        parameters_true = np.array([N_acc_amp,
                                    delta_x_amp,
                                    log_A1_true,
                                    alpha1_true,
                                    log_A2_true,
                                    alpha2_true,
                                    log_Omega_ast_true,
                                    alpha_ast_true])
    elif model == "noise+dwd+ast+sw":
        parameters_true = np.array([N_acc_amp,
                                    delta_x_amp,
                                    log_A1_true,
                                    alpha1_true,
                                    log_A2_true,
                                    alpha2_true,
                                    log_Omega_ast_true,
                                    alpha_ast_true,
                                    log_Omega_0_true,
                                    log_fp_true])
    return parameters_true
