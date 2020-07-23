import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functions import get_biomolecular_data

def plot_reference_curves(residue_list, text_size=12, figsize=None, save=False):

    l = len(residue_list) % 3
    if l == 0:
        plot_lenght = len(residue_list) // 3
        # uneven = False
    else:
        # uneven = True
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

    dataframe_all = pd.read_csv("dataframe_all_proteins.csv")
    categories_all = pd.Categorical(dataframe_all["res"])

    index_all = categories_all.codes

    model_cheshift_hierarchical = return_models()

    perct_dict = {}

    if "CYS" in residue_list:
        dataframe_all = dataframe_all[dataframe_all.res != "CYS"]
    for i, residue in enumerate(residue_list):
        ca_teo = dataframe_all[dataframe_all.res == residue].y_pred.values
        ca_exp = dataframe_all[dataframe_all.res == residue].ca_exp.values

        difference_dist = ca_teo - ca_exp
        density, x0, x1 = az.numeric_utils._fast_kde(difference_dist)
        x_range = np.linspace(x0, x1, len(density))

        perct = np.percentile(difference_dist, [0, 5, 20, 80, 95, 100])
        perct_dict[residue] = perct

        perct_colors = ["C8", "C3", "C8", "C2", "C8", "C3"]
        idx0 = 0
        for index, p in enumerate(perct):
            ax[i].tick_params(labelsize=16)
            idx1 = np.argsort(np.abs(x_range - p))[0]

            ax[i].fill_between(
                x_range[idx0:idx1], density[idx0:idx1], color="C0", zorder=0, alpha=0.3,
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
    residue_list=None,
    pymol_session=False,
    ax=None,
    marker="o",
    perct_dict=None,
):

    _, _, _, dataframe_full, _, _ = get_biomolecular_data(
        protein_code, bmrb_code=bmrb_code
    )

    if residue_list is None:
        residues = np.unique(dataframe_full.res.values)

    else:
        residues = residue_list

    if ax is None:
        _, ax, perct_dict = plot_reference_curves(residues)

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
    ### XXX cambiar print a una leyenda o algo por el estilo
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
            protein_code, param_list,
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

    return ax, dataframe_full, perct_dict
