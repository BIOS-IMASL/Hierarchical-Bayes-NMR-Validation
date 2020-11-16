import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from cheshift._cheshift import write_teo_cs


def plot_reference_densities(residue_list, text_size=12, figsize=None, save=False):

    """Plot the reference densities of CS differences for high quality protein structures."""
    l = len(residue_list) % 3
    if l == 0:
        plot_lenght = len(residue_list) // 3
    else:
        plot_lenght = len(residue_list) // 3 + 1

    if not figsize:
        figsize = (13, plot_lenght * 2)

    _, ax = plt.subplots(
        plot_lenght,
        3,
        figsize=figsize,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )

    ax = ax.ravel()

    if os.path.isfile("dataframe_all_proteins.csv"):
        dataframe_all = pd.read_csv("dataframe_all_proteins.csv")
    else:
        dataframe_all, trace_all = hierarchical_reg_reference()
        trace_all = az.from_pymc3(trace_all_proteins)
        az.to_netcdf(trace_all_proteins, "all_trace_reparam.nc")
        dataframe.to_csv("dataframe_all_proteins.csv")

    categories_all = pd.Categorical(dataframe_all["res"])

    index_all = categories_all.codes

    perct_dict = {}

    if "CYS" in residue_list:
        dataframe_all = dataframe_all[dataframe_all.res != "CYS"]
    for i, residue in enumerate(residue_list):
        ca_teo = dataframe_all[dataframe_all.res == residue].y_pred.values
        ca_exp = dataframe_all[dataframe_all.res == residue].ca_exp.values

        difference_dist = ca_teo - ca_exp
        _, density = az.stats.density_utils.kde(difference_dist)
        x0, x1 = np.min(difference_dist), np.max(difference_dist)
        x_range = np.linspace(x0, x1, len(density))

        perct = np.percentile(difference_dist, [0, 5, 20, 80, 95, 100])
        perct_dict[residue] = perct

        perct_colors = ["C8", "C3", "C8", "C2", "C8", "C3"]
        idx0 = 0
        for index, p in enumerate(perct):
            ax[i].tick_params(labelsize=16)
            idx1 = np.argsort(np.abs(x_range - p))[0]

            ax[i].fill_between(
                x_range[idx0:idx1],
                density[idx0:idx1],
                color="C0",
                zorder=0,
                alpha=0.3,
            )
            idx0 = idx1

        ax[i].set_title(residue, fontsize=text_size)

    [
        ax[idy].spines[position].set_visible(False)
        for position in ["left", "top", "right"]
        for idy in range(len(ax))
    ]
    [ax_.set_yticks([]) for ax_ in ax]
    [ax_.set_xlim(-5, 5) for ax_ in ax]

    for i in range(1, len(ax) - len(residue_list) + 1):
        ax[-i].axis("off")

    if save:
        plt.savefig(f"reference.png", dpi=300, transparent=True)
    return _, ax, perct_dict


def plot_cs_differences(
    protein_code,
    save=False,
    bmrb_code=None,
    residues=None,
    pymol_session=False,
    ax=None,
    marker="o",
    perct_dict=None,
):

    """Plot the reference densities of CS differences for target protein structures."""
    dataframe_full = get_biomolecular_data(protein_code, bmrb_code=bmrb_code)
    dataframe_full, trace, y_pred = hierarchical_reg_target(dataframe_full)

    if residues is None:
        residues = np.unique(dataframe_full.res.values)

    if ax is None:
        _, ax, perct_dict = plot_reference_densities(residues)

    param_list = []

    differences = dataframe_full.y_pred - dataframe_full.ca_exp

    len_residues = len(differences)
    red_residues = 0
    yellow_residues = 0
    green_residues = 0
    for a, res in enumerate(residues):

        idx = np.array(dataframe_full.res.values == res).ravel()
        residue_indexes = np.array([dataframe_full.index + 1]).ravel()[idx]

        difference = differences[dataframe_full.res == res]
        n = len(difference)
        jitter = np.linspace(-0.15, 0.0015, n)

        for z, diff in enumerate(difference):

            if diff > 5:
                diff = 5
            if diff < -5:
                diff = -5

            perct = perct_dict[res]

            if diff < perct[1] or diff > perct[-2]:
                color = ["C3", "red"]
                red_residues += 1
            elif diff < perct[2] or diff > perct[-3]:
                color = ["C8", "yellow"]
                yellow_residues += 1
            else:
                color = ["C2", "green"]
                green_residues += 1

            if res in dataframe_full.res.values:
                ax[a].scatter(
                    diff,
                    jitter[z],
                    marker=marker,
                    c=color[0],
                    linewidth=5,
                    s=10,
                    alpha=1,
                )

                param_list.append((residue_indexes[z], color[1], res))
            else:
                print(f"Residue {res} not in protein {protein_code}")

        annot = ax[a].annotate(
            "",
            xy=(0, 0),
            xytext=(7, 7),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="k", alpha=0.1),
        )

    print(
        np.round(
            np.array([red_residues, yellow_residues, green_residues])
            / len_residues
            * 100
        )
    )

    if save:
        plt.savefig(f".\\images\\{protein_code}_differences.png", dpi=300)

    if pymol_session:
        import color

        color.create_pymol_session(
            protein_code,
            param_list,
        )

        print(f"Search working directory for a PyMol session of protein {protein_code}")
        """
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"{pos[0]:.2f}"
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    # _.canvas.mpl_connect("motion_notify_event", hover)
        """
    # if residue_list is None:
    #    dataframe_full["colors"] = color_list

    return ax, dataframe_full, perct_dict, trace, y_pred

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



def create_pymol_session(protein_name, param_list):
    from pymol import cmd, stored

    """Create a pymol session for the protein structure. Colored as in difference_plot."""
    cmd.load("./pdbs/" + protein_name + ".pdb")
    cmd.color("white", "all")

    for index, color, res in param_list:
        if color == "red":
            color = "0xd62728"
        elif color == "yellow":
            color = "0xbcbd22"
        else:
            color = "0x2ca02c"
        cmd.select("sele", "resn {} and resi {}".format(res, index))
        cmd.color(color, "sele")
        cmd.delete("sele")

    cmd.save(os.path.join("pymol_sessions", "protein_name" + ".pse"))
    cmd.delete("all")
