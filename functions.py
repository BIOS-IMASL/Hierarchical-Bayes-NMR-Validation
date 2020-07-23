import pandas as pd
import pymc3 as pm
import numpy as np

import os
import arviz as az
from _cheshift import write_teo_cs


def load_data(protein=None):
    dataframe = pd.read_csv(
        "data/results_cheshift_theo_exp.csv",
        names=["protein", "res", "ca_teo", "ca_exp", "cb_teo", "cb_exp"],
    )

    if protein:
        dataframe = dataframe[dataframe["protein"] == protein]
        df = dataframe[dataframe["protein"] == protein]
    else:
        df = dataframe
    dataframe = dataframe[dataframe["res"] != "UNK"]
    dataframe = dataframe[dataframe["ca_exp"] < 100]
    dataframe = dataframe[dataframe["ca_teo"] < 100]
    dataframe.dropna(inplace=True)
    ca_exp = dataframe.ca_exp
    ca_teo = dataframe.ca_teo

    return ca_exp, ca_teo, dataframe, df


def hierarchical_reg(dataframe, samples=2000):
    """
    Runs a hierarchical model over a protein's residue.
    
    Parameters:
    ----------
    dataframe : contains cs experimental and theoretical data
    
    """

    dataframe = dataframe[dataframe["res"] != "UNK"]
    dataframe = dataframe[dataframe["ca_exp"] < 100]
    dataframe = dataframe[dataframe["ca_teo"] < 100]
    dataframe.dropna(inplace=True)

    mean_teo = dataframe["ca_teo"].mean()
    mean_exp = dataframe["ca_exp"].mean()
    std_teo = dataframe["ca_teo"].std()
    std_exp = dataframe["ca_exp"].std()

    ca_exp = (dataframe.ca_exp - mean_exp) / std_exp
    ca_teo = (dataframe.ca_teo - mean_teo) / std_teo

    categories = pd.Categorical(dataframe["res"])
    index = categories.codes
    N = len(np.unique(index))

    with pm.Model() as model_che:
        # hyper-priors
        alpha_sd = pm.HalfNormal("alpha_sd", 1.0)
        beta_sd = pm.HalfNormal("beta_sd", 1.0)
        # priors
        α = pm.Normal("α", 0, alpha_sd, shape=N)
        β = pm.HalfNormal("β", beta_sd, shape=N)
        σ = pm.HalfNormal("σ", 1.0)

        μ = pm.Deterministic("μ", α[index] + β[index] * ca_teo)

        cheshift = pm.Normal("cheshift", mu=μ, sigma=σ, observed=ca_exp)
        trace_che = pm.sample(samples, tune=2000, random_seed=18759)

    y_pred = (
        pm.sample_posterior_predictive(
            trace_che, model=model_che, samples=samples * trace_che.nchains
        )["cheshift"]
        * std_exp
        + mean_exp
    )

    az.to_netcdf(trace_che, "all_trace_reparam.nc")
    dataframe["y_pred"] = y_pred.mean(0)

    return y_pred, index, categories, dataframe, trace_che, model_che


def hierarchical_reg_one_protein(dataframe, samples=2000):
    """
    Runs a hierarchical model over a protein's residue.
    
    Parameters:
    ----------
    dataframe : contains cs experimental and theoretical data
    
    """

    dataframe = dataframe[dataframe["res"] != "UNK"]
    dataframe = dataframe[dataframe["ca_exp"] < 500]
    dataframe = dataframe[dataframe["ca_teo"] < 500]
    dataframe.dropna(inplace=True)
    mean_teo = 56.69
    mean_exp = 56.70
    std_teo = 4.65
    std_exp = 4.90

    # XXX No alteramos el dataframe
    ca_exp = (dataframe.ca_exp - mean_exp) / std_exp
    ca_teo = (dataframe.ca_teo - mean_exp) / std_exp

    categories = pd.Categorical(dataframe["res"])
    index = categories.codes
    N = len(np.unique(index))

    (
        ca_exp_all_proteins,
        ca_teo_all_proteins,
        dataframe_all_proteins,
        df_all_proteins,
    ) = load_data()
    categories_all_proteins = pd.Categorical(dataframe_all_proteins["res"])
    index_all_proteins = categories_all_proteins.codes
    N_all_proteins = len(np.unique(index_all_proteins))

    aa_to_delete = list(
        set(categories_all_proteins.categories) - set(categories.categories)
    )

    index_to_delete = []
    for aa in aa_to_delete:
        indice = np.argwhere(categories_all_proteins.categories == aa)[0][0]
        index_to_delete.append(indice)

    # XXX Modifiqué esto por que fallaba
    if os.path.isfile("all_trace_reparam.nc"):
        trace_all_proteins = az.from_netcdf("all_trace_reparam.nc")
    else:
        _, _, _, _, trace_all_proteins, _ = hierarchical_reg(dataframe_all_proteins)
        trace_all_proteins = az.from_pymc3(trace_all_proteins)
        az.to_netcdf(trace_all_proteins, "all_trace_reparam.nc")

    learnt_alpha_sd_mean = trace_all_proteins.posterior.alpha_sd.mean(
        dim=["chain", "draw"]
    ).values
    learnt_beta_sd_mean = trace_all_proteins.posterior.beta_sd.mean(
        dim=["chain", "draw"]
    ).values

    with pm.Model() as model_che:
        # hyper-priors
        alpha_sd = pm.HalfNormal("alpha_sd", learnt_alpha_sd_mean)
        beta_sd = pm.HalfNormal("beta_sd", learnt_beta_sd_mean)
        # priors
        α = pm.Normal("α", 0, alpha_sd, shape=N)
        β = pm.HalfNormal("β", beta_sd, shape=N)
        σ = pm.HalfNormal("σ", 1.0)

        μ = pm.Deterministic("μ", α[index] + β[index] * ca_teo)

        cheshift = pm.Normal("cheshift", mu=μ, sigma=σ, observed=ca_exp)
        trace_che = pm.sample(samples, tune=2000, random_seed=18759)

    y_pred = (
        pm.sample_posterior_predictive(
            trace_che, model=model_che, samples=samples * trace_che.nchains
        )["cheshift"]
        * std_exp
        + mean_exp
    )

    dataframe["y_pred"] = y_pred.mean(0)
    return y_pred, index, categories, dataframe, trace_che, model_che


def get_biomolecular_data(protein_name, bmrb_code):

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

    (
        pps,
        index,
        categories,
        dataframe_protein,
        trace_che,
        model,
    ) = hierarchical_reg_one_protein(dataframe, samples=2000)

    return dataframe


def one_letter_code(aa):
    one_letter_code_d = {
        "CYS": "C",
        "ASP": "D",
        "SER": "S",
        "GLN": "Q",
        "LYS": "K",
        "ILE": "I",
        "PRO": "P",
        "THR": "T",
        "PHE": "F",
        "ASN": "N",
        "HIS": "H",
        "LEU": "L",
        "ARG": "R",
        "TRP": "W",
        "ALA": "A",
        "VAL": "V",
        "GLU": "E",
        "TYR": "Y",
        "MET": "M",
        "GLY": "G",
    }

    three_letter_code_d = {v: k for k, v in one_letter_code_d.items()}
    if len(aa) > 1:
        return one_letter_code_d[aa]
    else:
        return three_letter_code_d[aa]
