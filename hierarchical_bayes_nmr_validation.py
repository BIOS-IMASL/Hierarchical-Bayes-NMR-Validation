import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from cheshift._cheshift import write_teo_cs

np.random.seed(18759)

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

    if os.path.isfile(os.path.join("data", "dataframe_reference_structures.csv")):
        dataframe_all = pd.read_csv(os.path.join("data", "dataframe_reference_structures.csv"))
    else:
        dataframe_all, trace_all = hierarchical_reg_reference()
        trace_all = az.from_pymc3(trace_all_proteins)
        az.to_netcdf(trace_all_proteins, os.path.join("data", "trace_reference_structures.nc"))
        dataframe.to_csv(os.path.join("data", "dataframe_reference_structures.csv"))

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
    [ax_.set_xlim(-6, 6) for ax_ in ax]

    for i in range(1, len(ax) - len(residue_list) + 1):
        ax[-i].axis("off")

    if save:
        plt.savefig(f"reference.png", dpi=300, transparent=True)
    return _, ax, perct_dict


def plot_cs_differences(
    protein_code,
    target_accept=0.9,
    save=False,
    bmrb_code=None,
    residues=None,
    pymol_session=False,
    ax=None,
    marker="o",
    perct_dict=None,
    plot_kwargs=None,
):

    """Plot the reference densities of CS differences for target protein structures."""

    _, _, reference_df = load_data()
    mean_exp = reference_df["ca_exp"].mean()
    std_exp = reference_df["ca_exp"].std()

    if not plot_kwargs:
        plot_kwargs = {}
    plot_kwargs.setdefault("s", 10)
    plot_kwargs.setdefault("alpha", 1)

    dataframe_full = get_biomolecular_data(protein_code, bmrb_code=bmrb_code)
    if f'idata_{protein_code}.nc' in os.listdir('./data/'):
        idata_target = az.from_netcdf(f'data/idata_{protein_code}.nc')
    else:
        dataframe_reference, idata = hierarchical_reg_reference(target_df=dataframe_full)
        idata_target = idata.sel(
        cheshift_dim_0=slice(dataframe_reference.shape[0]-dataframe_full.shape[0], 
            dataframe_reference.shape[0]))

    idata_target.posterior_predictive = idata_target.posterior_predictive * std_exp + mean_exp

    if residues is None:
        residues = np.unique(dataframe_full.res.values)

    if ax is None:
        _, ax, perct_dict = plot_reference_densities(residues)

    param_list = []

    differences = idata_target.posterior_predictive['cheshift'].values.mean(axis=(0, 1)) - dataframe_full.ca_exp

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
                color = ["C1", "yellow"]
                red_residues += 1
            elif diff < perct[2] or diff > perct[-3]:
                color = ["C6", "orange"]
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
                    
                    **plot_kwargs,
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
        plt.savefig(os.path.join("images", f"{protein_code}_differences.png"), dpi=600)

    if pymol_session:

        create_pymol_session(
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

    return ax, dataframe_full, perct_dict, idata_target

def load_data(protein=None):

    """Load CS data for a protein in the reference data set. Run with protein=None for loading
    the entire data set."""
    dataframe = pd.read_csv(os.path.join("data", "results_cheshift_theo_exp.csv"),
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
    dataframe = pd.read_csv(os.path.join("data",
        f"{protein_name.lower()}_cs_theo_exp.csv"),
        names=["protein", "res", "bmrb_code", "ca_exp", "ca_teo"],
    )

    dataframe = dataframe[dataframe["res"] != "UNK"]
    dataframe = dataframe[dataframe["ca_exp"] < 500]
    dataframe = dataframe[dataframe["ca_teo"] < 500]
    dataframe = dataframe[dataframe.res != "CYS"]
    dataframe.dropna(inplace=True)

    return dataframe


def hierarchical_reg_reference(samples=2000, target_df=None):

    """Runs a hierarchical model over the reference data set."""
    _, _, dataframe = load_data()

    if target_df is None:
        target_df = pd.DataFrame({})

    else:
        del target_df['bmrb_code']
        dataframe = dataframe[dataframe.protein != '1UBQ']
        dataframe = pd.concat([dataframe, target_df], ignore_index=True)


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
        sigma_sd = pm.HalfNormal("sigma_sd", 1.0)
        # priors
        α = pm.Normal("α", 0, alpha_sd, shape=N)
        β = pm.HalfNormal("β", beta_sd, shape=N)
        σ = pm.HalfNormal("σ", sigma_sd, shape=N)
        # linear model
        μ = pm.Deterministic("μ", α[index] + β[index] * ca_teo)
        # likelihood
        cheshift = pm.Normal("cheshift", mu=μ, sigma=σ[index], observed=ca_exp)
        idata = pm.sample(samples, tune=2000, random_seed=18759, target_accept=0.9, return_inferencedata=True)
        pps = pm.sample_posterior_predictive(idata, samples=samples * idata.posterior.dims["chain"], random_seed=18759)
        idata.add_groups({"posterior_predictive":{"cheshift":pps["cheshift"][None,:,:]}})

    if target_df is None:
        az.to_netcdf(idata, os.path.join("data", "trace_reference_structures.nc"))

    return dataframe, idata


def hierarchical_reg_target(dataframe, target_accept=0.9, samples=2000):
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

    if os.path.isfile(os.path.join("data", "trace_reference_structures.nc")):
        trace_all_proteins = az.from_netcdf(os.path.join("data", "trace_reference_structures.nc"))
        print(f"Loaded reference trace from {os.path.join('data', 'trace_reference_structures.nc')}")
    else:
        print(f"could not find reference trace from {os.path.join('data', 'trace_reference_structures.nc')}")
        print("Running model for reference structures")
        dataframe_all_proteins, trace_all_proteins = hierarchical_reg_reference()
        #trace_all_proteins = az.from_pymc3(trace_all_proteins)
        #az.to_netcdf(trace_all_proteins, os.path.join("data", "trace_reference_structures.nc"))
        #dataframe_all_proteins.to_csv(os.path.join("data", "dataframe_reference_structures.csv"))

    learnt_alpha_sd_mean = trace_all_proteins.posterior.alpha_sd.mean(
        dim=["chain", "draw"]
    ).values
    learnt_beta_sd_mean = trace_all_proteins.posterior.beta_sd.mean(
        dim=["chain", "draw"]
    ).values
    learnt_sigma_sd_mean = trace_all_proteins.posterior.sigma_sd.mean(
        dim=["chain", "draw"]
    ).values


    with pm.Model() as model:
        # hyper-priors
        alpha_sd = pm.HalfNormal("alpha_sd", learnt_alpha_sd_mean)
        beta_sd = pm.HalfNormal("beta_sd", learnt_beta_sd_mean)
        sigma_sd = pm.HalfNormal("sigma_sd", learnt_beta_sd_mean)
        # priors
        α = pm.Normal("α", 0, alpha_sd, shape=N)
        β = pm.HalfNormal("β", beta_sd, shape=N)
        σ = pm.HalfNormal("σ", sigma_sd, shape=N)
        # linear model
        μ = pm.Deterministic("μ", α[index] + β[index] * ca_teo)
        # likelihood
        cheshift = pm.Normal("cheshift", mu=μ, sigma=σ[index], observed=ca_exp)
        idata = pm.sample(samples, tune=2000, random_seed=18759, target_accept=0.9, return_inferencedata=True)
        pps = pm.sample_posterior_predictive(idata, samples=samples * idata.posterior.dims["chain"], random_seed=18759)
        idata.add_groups({"posterior_predictive":{"cheshift":pps["cheshift"][None,:,:]}})

    return dataframe, idata



def create_pymol_session(protein_name, param_list):
    from pymol import cmd, stored

    """Create a pymol session for the protein structure. Colored as in difference_plot."""
    cmd.load( os.path.join("data", f"{protein_name}" + ".pdb"))
    cmd.color("white", "all")

    cmd.set_color('yellow', [0.9803921568627451, 0.48627450980392156, 0.09019607843137255] )
    cmd.set_color('orange', [0.9019607843137255, 0.8823529411764706, 0.20784313725490197])
    cmd.set_color('green', [0.19607843137254902, 0.5490196078431373, 0.023529411764705882])

    for index, color, res in param_list:            
        cmd.select("sele", "resn {} and resi {}".format(res, index))
        cmd.color(color, "sele")
        cmd.delete("sele")

    cmd.save(os.path.join("pymol_sessions", f"{protein_name}" + ".pse"))
    cmd.delete("all")
