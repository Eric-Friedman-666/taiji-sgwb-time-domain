import bilby
import jax
import jax.numpy as jnp
from py_code.parameters import *


def get_priors(model):
    priors = bilby.prior.PriorDict()
    if model == "noise":
        priors["N_acc_amp"] = bilby.prior.Uniform(para_range["N_acc_amp"][0], para_range["N_acc_amp"][1], name="N_acc_amp")
        priors["delta_x_amp"] = bilby.prior.Uniform(para_range["delta_x_amp"][0], para_range["delta_x_amp"][1], name="delta_x_amp")
    elif model == "noise+dwd":
        priors["N_acc_amp"] = bilby.prior.Uniform(para_range["N_acc_amp"][0], para_range["N_acc_amp"][1], name="N_acc_amp")
        priors["delta_x_amp"] = bilby.prior.Uniform(para_range["delta_x_amp"][0], para_range["delta_x_amp"][1], name="delta_x_amp")
        priors["log_A1"] = bilby.prior.Uniform(para_range["log_A1"][0], para_range["log_A1"][1], name="log_A1")
        priors["alpha1"] = bilby.prior.Uniform(para_range["alpha1"][0], para_range["alpha1"][1], name="alpha1")
        priors["log_A2"] = bilby.prior.Uniform(para_range["log_A2"][0], para_range["log_A2"][1], name="log_A2")
        priors["alpha2"] = bilby.prior.Uniform(para_range["alpha2"][0], para_range["alpha2"][1], name="alpha2")
    elif model == "noise+dwd+ast":
        priors["N_acc_amp"] = bilby.prior.Uniform(para_range["N_acc_amp"][0], para_range["N_acc_amp"][1], name="N_acc_amp")
        priors["delta_x_amp"] = bilby.prior.Uniform(para_range["delta_x_amp"][0], para_range["delta_x_amp"][1], name="delta_x_amp")
        priors["log_A1"] = bilby.prior.Uniform(para_range["log_A1"][0], para_range["log_A1"][1], name="log_A1")
        priors["alpha1"] = bilby.prior.Uniform(para_range["alpha1"][0], para_range["alpha1"][1], name="alpha1")
        priors["log_A2"] = bilby.prior.Uniform(para_range["log_A2"][0], para_range["log_A2"][1], name="log_A2")
        priors["alpha2"] = bilby.prior.Uniform(para_range["alpha2"][0], para_range["alpha2"][1], name="alpha2")
        priors["log_Omega_ast"] = bilby.prior.Uniform(para_range["log_Omega_ast"][0], para_range["log_Omega_ast"][1], name="log_Omega_ast")
        priors["alpha_ast"] = bilby.prior.Uniform(para_range["alpha_ast"][0], para_range["alpha_ast"][1], name="alpha_ast")
    elif model == "noise+dwd+ast+sw":
        priors["N_acc_amp"] = bilby.prior.Uniform(para_range["N_acc_amp"][0], para_range["N_acc_amp"][1], name="N_acc_amp")
        priors["delta_x_amp"] = bilby.prior.Uniform(para_range["delta_x_amp"][0], para_range["delta_x_amp"][1], name="delta_x_amp")
        priors["log_A1"] = bilby.prior.Uniform(para_range["log_A1"][0], para_range["log_A1"][1], name="log_A1")
        priors["alpha1"] = bilby.prior.Uniform(para_range["alpha1"][0], para_range["alpha1"][1], name="alpha1")
        priors["log_A2"] = bilby.prior.Uniform(para_range["log_A2"][0], para_range["log_A2"][1], name="log_A2")
        priors["alpha2"] = bilby.prior.Uniform(para_range["alpha2"][0], para_range["alpha2"][1], name="alpha2")
        priors["log_Omega_ast"] = bilby.prior.Uniform(para_range["log_Omega_ast"][0], para_range["log_Omega_ast"][1], name="log_Omega_ast")
        priors["alpha_ast"] = bilby.prior.Uniform(para_range["alpha_ast"][0], para_range["alpha_ast"][1], name="alpha_ast")
        priors["log_Omega_0"] = bilby.prior.Uniform(para_range["log_Omega_0"][0], para_range["log_Omega_0"][1], name="log_Omega_0")
        priors["log_fp"] = bilby.prior.Uniform(para_range["log_fp"][0], para_range["log_fp"][1], name="log_fp")
    elif model == "noise+ast":
        priors["N_acc_amp"] = bilby.prior.Uniform(para_range["N_acc_amp"][0], para_range["N_acc_amp"][1], name="N_acc_amp")
        priors["delta_x_amp"] = bilby.prior.Uniform(para_range["delta_x_amp"][0], para_range["delta_x_amp"][1], name="delta_x_amp")
        priors["log_Omega_ast"] = bilby.prior.Uniform(para_range["log_Omega_ast"][0], para_range["log_Omega_ast"][1], name="log_Omega_ast")
        priors["alpha_ast"] = bilby.prior.Uniform(para_range["alpha_ast"][0], para_range["alpha_ast"][1], name="alpha_ast")
    else:
        raise ValueError(f"Unsupported model: {model}")
    return priors


class Likelihood(bilby.Likelihood):
    def __init__(self, params, model, loglike):
        self.params = params
        self.model = model
        self.loglike = loglike
        if model == "noise+dwd+ast+sw":
            super().__init__(parameters=dict(
                N_acc_amp=self.params['N_acc_amp'], delta_x_amp=self.params['delta_x_amp'], log_Omega_ast=self.params['log_Omega_ast'][-1], alpha_ast=self.params['alpha_ast'][-1], log_A1=self.params['log_A1'][-1], alpha1=self.params['alpha1'][-1], log_A2=self.params['log_A2'][-1], alpha2=self.params['alpha2'][-1], log_Omega_0=self.params['log_Omega_0'][-1], log_fp=self.params['log_fp'][-1]
            ))
        elif model == "noise+dwd+ast":
            super().__init__(parameters=dict(
                N_acc_amp=self.params['N_acc_amp'], delta_x_amp=self.params['delta_x_amp'], log_A1=self.params['log_A1'][-1], alpha1=self.params['alpha1'][-1], log_A2=self.params['log_A2'][-1], alpha2=self.params['alpha2'][-1], log_Omega_ast=self.params['log_Omega_ast'][-1], alpha_ast=self.params['alpha_ast'][-1]
            ))
        elif model == "noise+dwd":
            super().__init__(parameters=dict(
                N_acc_amp=self.params['N_acc_amp'], delta_x_amp=self.params['delta_x_amp'], log_A1=self.params['log_A1'][-1], alpha1=self.params['alpha1'][-1], log_A2=self.params['log_A2'][-1], alpha2=self.params['alpha2'][-1]
            ))
        elif model == "noise":
            super().__init__(parameters=dict(
                N_acc_amp=self.params['N_acc_amp'], delta_x_amp=self.params['delta_x_amp']
            )) 
        elif model == "noise+ast":
            super().__init__(parameters=dict(
                N_acc_amp=self.params['N_acc_amp'], delta_x_amp=self.params['delta_x_amp'], log_Omega_ast=self.params['log_Omega_ast'][-1], alpha_ast=self.params['alpha_ast'][-1]
            ))
        else:
            raise ValueError(f"Unsupported model: {model}")
    def log_likelihood(self):
        if self.model == "noise+dwd+ast+sw":
            theta = jnp.array([
                self.parameters["N_acc_amp"],
                self.parameters["delta_x_amp"],
                self.parameters["log_A1"],    
                self.parameters["alpha1"],     
                self.parameters["log_A2"],     
                self.parameters["alpha2"],    
                self.parameters["log_Omega_ast"],
                self.parameters["alpha_ast"],
                self.parameters["log_Omega_0"],
                self.parameters["log_fp"],
            ], dtype=jnp.float64)
        elif self.model == "noise+dwd+ast":
            theta = jnp.array([
                self.parameters["N_acc_amp"],
                self.parameters["delta_x_amp"],
                self.parameters["log_A1"],    
                self.parameters["alpha1"],     
                self.parameters["log_A2"],     
                self.parameters["alpha2"],    
                self.parameters["log_Omega_ast"],
                self.parameters["alpha_ast"],
                -100,
                0
            ], dtype=jnp.float64)
        elif self.model == "noise+dwd":
            theta = jnp.array([
                self.parameters["N_acc_amp"],
                self.parameters["delta_x_amp"],
                self.parameters["log_A1"],    
                self.parameters["alpha1"],     
                self.parameters["log_A2"],     
                self.parameters["alpha2"],    
                -100,
                0,
                -100,
                0
            ], dtype=jnp.float64)
        elif self.model == "noise+ast":
            theta = jnp.array([
                self.parameters["N_acc_amp"],
                self.parameters["delta_x_amp"],
                self.parameters["log_Omega_ast"],
                self.parameters["alpha_ast"],
                -100,
                0,
                -100,
                0,
                -100,
                0
            ], dtype=jnp.float64)
        elif self.model == "noise":
            theta = jnp.array([
                self.parameters["N_acc_amp"],
                self.parameters["delta_x_amp"],
                -100,
                0,
                -100,
                0,
                -100,
                0,
                -100,
                0
            ], dtype=jnp.float64)
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        return float(self.loglike(theta))





