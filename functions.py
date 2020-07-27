import pandas as pd
import pymc3 as pm
import numpy as np

import os
import arviz as az
from _cheshift import write_teo_cs


def load_data(protein=None):

    """Load CS data for a protein in the reference data set. Run with protein=None for loading
    the entire data set."""
    dataframe = pd.read_csv(
        "data/results_cheshift_theo_exp.csv",
        names=["protein", "res", "ca_teo", "ca_exp", "cb_teo", "cb_exp"],
    )

    if protein:
        dataframe = dataframe[dataframe["protein"] == protein]

    dataframe = dataframe[dataframe["res"] != "UNK"]
    dataframe = dataframe[dataframe["ca_exp"] < 100]
    dataframe = dataframe[dataframe["ca_teo"] < 100]
    dataframe.dropna(inplace=True)
    ca_exp = dataframe.ca_exp
    ca_teo = dataframe.ca_teo

    return ca_exp, ca_teo, dataframe


def get_biomolecular_data(protein_name, bmrb_code):

    """Combine CS data from a PDB file and a BMRB file into a single dataframe"""
    write_teo_cs(protein_name, bmrb_code)
    dataframe = pd.read_csv(
        f"{protein_name.lower()}_cs_theo_exp.csv",
        names=["pdb_code", "res", "bmrb_code", "ca_exp", "ca_teo"],
    )

    dataframe = dataframe[dataframe["res"] != "UNK"]
    dataframe = dataframe[dataframe["ca_exp"] < 500]
    dataframe = dataframe[dataframe["ca_teo"] < 500]
    dataframe = dataframe[dataframe.res != "CYS"]
    dataframe.dropna(inplace=True)

    return dataframe


def hierarchical_reg_reference(samples=2000):

    """Runs a hierarchical model over the reference data set."""
    _, _, dataframe = load_data()

    mean_teo = dataframe["ca_teo"].mean()
    mean_exp = dataframe["ca_exp"].mean()
    std_teo = dataframe["ca_teo"].std()
    std_exp = dataframe["ca_exp"].std()

    ca_exp = (dataframe.ca_exp - mean_exp) / std_exp
    ca_teo = (dataframe.ca_teo - mean_teo) / std_teo

    categories = pd.Categorical(dataframe["res"])
    index = categories.codes
    N = len(np.unique(index))

    with pm.Model() as model:
        # hyper-priors
        alpha_sd = pm.HalfNormal("alpha_sd", 1.0)
        beta_sd = pm.HalfNormal("beta_sd", 1.0)
        # priors
        α = pm.Normal("α", 0, alpha_sd, shape=N)
        β = pm.HalfNormal("β", beta_sd, shape=N)
        σ = pm.HalfNormal("σ", 1.0)
        # linear model
        μ = pm.Deterministic("μ", α[index] + β[index] * ca_teo)
        # likelihood
        cheshift = pm.Normal("cheshift", mu=μ, sigma=σ, observed=ca_exp)
        trace = pm.sample(samples, tune=2000, random_seed=18759)

    y_pred = (
        pm.sample_posterior_predictive(
            trace, model=model, samples=samples * trace.nchains
        )["cheshift"]
        * std_exp
        + mean_exp
    )

    az.to_netcdf(trace, "all_trace_reparam.nc")
    dataframe["y_pred"] = y_pred.mean(0)

    return dataframe, trace


def hierarchical_reg_target(dataframe, samples=2000):
    """
    Runs a hierarchical model over the target structure CS data set.
    
    Parameters:
    ----------
    dataframe : contains experimental and theoretical CS data
    
    """
    _, _, reference_dataframe = load_data()
    mean_teo = reference_dataframe["ca_teo"].mean()
    mean_exp = reference_dataframe["ca_exp"].mean()
    std_teo = reference_dataframe["ca_teo"].std()
    std_exp = reference_dataframe["ca_exp"].std()

    ca_exp = (dataframe.ca_exp - mean_exp) / std_exp
    ca_teo = (dataframe.ca_teo - mean_exp) / std_exp

    categories = pd.Categorical(dataframe["res"])
    index = categories.codes
    N = len(np.unique(index))

    if os.path.isfile("all_trace_reparam.nc"):
        trace_all_proteins = az.from_netcdf("all_trace_reparam.nc")
    else:
        dataframe_all_proteins, trace_all_proteins = hierarchical_reg_reference()
        trace_all_proteins = az.from_pymc3(trace_all_proteins)
        az.to_netcdf(trace_all_proteins, "all_trace_reparam.nc")
        dataframe_all_proteins.to_csv("dataframe_all_proteins.csv")

    learnt_alpha_sd_mean = trace_all_proteins.posterior.alpha_sd.mean(
        dim=["chain", "draw"]
    ).values
    learnt_beta_sd_mean = trace_all_proteins.posterior.beta_sd.mean(
        dim=["chain", "draw"]
    ).values

    with pm.Model() as model:
        # hyper-priors
        alpha_sd = pm.HalfNormal("alpha_sd", learnt_alpha_sd_mean)
        beta_sd = pm.HalfNormal("beta_sd", learnt_beta_sd_mean)
        # priors
        α = pm.Normal("α", 0, alpha_sd, shape=N)
        β = pm.HalfNormal("β", beta_sd, shape=N)
        σ = pm.HalfNormal("σ", 1.0)
        # linear model
        μ = pm.Deterministic("μ", α[index] + β[index] * ca_teo)
        # likelihood
        cheshift = pm.Normal("cheshift", mu=μ, sigma=σ, observed=ca_exp)
        trace = pm.sample(samples, tune=2000, random_seed=18759)

    y_pred = (
        pm.sample_posterior_predictive(
            trace, model=model, samples=samples * trace.nchains
        )["cheshift"]
        * std_exp
        + mean_exp
    )

    dataframe["y_pred"] = y_pred.mean(0)
    return dataframe, trace, y_pred
