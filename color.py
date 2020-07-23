from pymol import cmd, stored

"""
import __main__

__main__.pymol_argv = ["pymol", "-qc"]  # Pymol: quiet and no GUI
import sys, pymol

stdout = sys.stdout
pymol.finish_launching()
"""
import numpy as np


def create_pymol_session(protein_name, param_list):

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
