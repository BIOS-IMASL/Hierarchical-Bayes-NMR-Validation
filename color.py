from pymol import cmd, stored
import numpy as np


def create_pymol_session(protein_name, param_list):

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

    cmd.save("./pymol_sessions/" + protein_name + ".pse")
    cmd.delete("all")
