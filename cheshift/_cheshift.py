import numpy as np
from scipy.interpolate import griddata
import os
import sys
import re
import glob


def split_pdb(pdb_filename):
    pdb_list = []
    for line in open(os.path.join("data","%s.pdb" % pdb_filename.lower())).readlines():
        if line[0:5] == "ATOM " and (line[16:17] == " " or line[16:17] == "A"):
            pdb_list.append(line)
    # extracts only one chain if more than one
    PDB_chain_list = []
    chain = pdb_list[0][21:23]
    for atomline in range(len(pdb_list)):
        if chain == pdb_list[atomline][21:23]:
            PDB_chain_list.append(pdb_list[atomline])
    # the use of primer allows the script to detect when to split the file
    primer = -1e6
    count = 0
    fd = open(os.path.join("data", "%s_%03d.pdb" % (pdb_filename, count)), "w")
    for atomline in range(len(PDB_chain_list)):
        atomnumber = int(PDB_chain_list[atomline][6:12])
        if atomnumber > primer:
            fd.write("%s" % (PDB_chain_list[atomline]))
            primer = int(PDB_chain_list[atomline][6:12])
        else:
            primer = -1e6
            fd.close()
            count += 1
            fd = open(os.path.join("data","%s_%03d.pdb" % (pdb_filename, count)), "w")
            fd.write("%s" % (PDB_chain_list[atomline]))
    fd.close()
    conformations_num = count + 1
    return conformations_num


def bmrb2cheshift(bmrb):
    """Parse the Star-bmrb format into a 4 column file"""
    reference = None
    for line in open(os.path.join("data", bmrb)).readlines():
        if 'DSS' in line:
            reference = 1.7
        elif 'TSP' in line:
            reference = 1.82
        elif 'TMS' in line:
            reference = 0.00
        if reference is not None:
            break
    if reference is None:
        reference = 0.0
    try:
        bmrb_ca = []
        bmrb_cb = []
        a = re.compile('[0-9]{1,5}\s{1,5}[A-Z]{3}\s{1,4}CA\s{0,6}C.{0,10}[0-9]*\.[0-9]{0,2}')
        b = re.compile('[0-9]{1,5}\s{1,5}[A-Z]{3}\s{1,4}CB\s{0,6}C.{0,10}[0-9]*\.[0-9]{0,2}')
        for line in open(os.path.join("data","%s" % bmrb)).readlines():
            if a.search(line):
                data = a.search(line).group().split()
                bmrb_ca.append(data)
            if b.search(line):
                data = b.search(line).group().split()
                bmrb_cb.append(data)
        len_a = len(bmrb_ca)
        len_b = len(bmrb_cb)
        if len_a > len_b:
            dif = len_a - len_b
            for i in range(0, dif):
                bmrb_cb.append(['99999'])
        elif len_a < len_b:
            dif = len_b - len_a
            for i in range(0, dif):
                bmrb_ca.append(['99999'])
        count_ca = 0
        count_cb = 0
        ocs_list = []
        while True:
            try:
                resn_ca = int(bmrb_ca[count_ca][0])
                resn_cb = int(bmrb_cb[count_cb][0])
                if resn_ca == resn_cb:
                    line = '%4s %3s  %6.2f  %6.2f\n' % (bmrb_ca[count_ca][0], bmrb_ca[count_ca][1], float(bmrb_ca[count_ca][-1]), float(bmrb_cb[count_cb][-1]))
                    ocs_list.append(line)
                    count_ca += 1
                    count_cb += 1
                if resn_ca > resn_cb:
                    line = '%4s %3s  %6.2f  %6.2f\n' % (bmrb_cb[count_cb][0], bmrb_cb[count_cb][1], 999.00, float(bmrb_cb[count_cb][-1]))
                    ocs_list.append(line)
                    count_cb += 1
                if resn_ca < resn_cb:
                    line = '%4s %3s  %6.2f  %6.2f\n' % (bmrb_ca[count_ca][0], bmrb_ca[count_ca][1], float(bmrb_ca[count_ca][-1]), 999.00)
                    ocs_list.append(line)
                    count_ca += 1
            except:
                break
        res_old = int(ocs_list[0].split()[0])
        count0 = 0
        count1 = 0
        safe = 0
        fd = open(os.path.join("data", f"{bmrb}.ocs"), "w")
        while count0 < len(ocs_list):
            safe += 1
            if safe > len(bmrb_ca)*5:
                break
            res_new = int(ocs_list[count0].split()[0])
            if res_old + count1 == res_new:
                fd.write(ocs_list[count0])
                count0 += 1
                count1 += 1
            else:
                fd.write('%4s UNK  999.00  999.00\n' % (res_old + count1))
                count1 += 1
        fd.close()
    except:
        fd = open(os.path.join("data", f"{bmrb}.ocs"), "w")
        cs_file = open(os.path.join("data", f"{bmrb}")).readlines()
        reference = cs_file[0]
        for line in cs_file[1:]:
            fd.write(line)
        fd.close()
    return reference


def check_seq(pdb_filename, cs_exp_name):
    # read a pdb and extract the sequence using three leter code and save it to a list
    ok = True
    pdb_list = []
    seqlist = []
    seqlist2 = []
    for line in open(os.path.join("data","%s_000.pdb" % (pdb_filename))).readlines():
        if "ATOM " in line:
            pdb_list.append(line)
    res_num = pdb_list[0][22:26]
    seqlist.append(pdb_list[0][17:20])
    for atoms in range(len(pdb_list)):
        if res_num != pdb_list[atoms][22:26]:
            seqlist.append(pdb_list[atoms][17:20])
            res_num = pdb_list[atoms][22:26]
    # check missing residues in the pdb file
    res_num = int(pdb_list[0][22:26])
    seqlist2.append(pdb_list[0][17:20])
    for atoms in range(len(pdb_list)):
        if res_num + 1 == int(pdb_list[atoms][22:26]):
            seqlist2.append(pdb_list[atoms][17:20])
            res_num = int(pdb_list[atoms][22:26])
    if len(seqlist) != len(seqlist2):
        ok = False
        pdb_res = len(seqlist2) + int(pdb_list[0][22:26])
        pdb_res = len(seqlist2) + int(pdb_list[0][22:26])
    else:
        # read the ocs file and extract the sequence using three letter code and save it to a list
        ocslist = []  # contains the sequence from the ocs file
        ocslist_full = []  # contains the whole ocs file
        ocslist_full_new = []
        for line in open(os.path.join("data", f"{cs_exp_name}.ocs")).readlines():
            ocslist_full.append(line)
            ocslist.append(line.split()[1])
        indelfirst, indellast = align(ocslist, seqlist)
        if indelfirst == 0 and indellast == 0:
            ocslist_full_new = list(ocslist_full)
        else:
            firstocs = int(ocslist_full[0].split()[0])
            lastocs = int(ocslist_full[-1].split()[0])
            newfirst = firstocs - indelfirst
            start = 0
            stop = len(ocslist_full)
            if indelfirst < 0:
                start = abs(indelfirst)
            if indellast < 0:
                stop = len(ocslist_full) + indellast
            line = "%s" % ocslist_full[0]
            for i in range(
                newfirst, firstocs
            ):  # works only if indelfirst is greater than 0
                line = "%4s %3s  %8.2f  %8.2f\n" % (i, "UNK", 999.00, 999.00)
                ocslist_full_new.append(line)
            for i in range(start, stop):
                line = "%s" % ocslist_full[i]
                ocslist_full_new.append(line)
            for i in range(
                lastocs, lastocs + indellast
            ):  # works only if indellast is positive
                line = "%4s %3s  %8.2f  %8.2f\n" % (i, "UNK", 999.00, 999.00)
                ocslist_full_new.append(line)
        # check if both sequences match
        fd = open(os.path.join("data", f"{cs_exp_name}.ocs"), "w")
        for i in range(len(seqlist)):
            if (
                seqlist[i] == ocslist_full_new[i].split()[1]
                or ocslist_full_new[i].split()[1] == "UNK"
            ):
                fd.write("%s" % ocslist_full_new[i])
            else:
                pdb_res = seqlist[i]
                ocs_res = ocslist[i - indelfirst]
                pdb_num = i + 1
                ocs_num = i + 1 - indelfirst
                pdb_seq = " ".join(seqlist)
                ocs_seq = " ".join(ocslist)
                ok = False
                for ocs in ocslist:
                    if ocs_seq.endswith("UNK"):
                        ocs_seq = ocs_seq[:-4]
                    else:
                        break
                break
        fd.close()
    return ok, pdb_list, ocslist_full_new


def align(three0_list, three1_list):
    """This function aligns two sequences using a brute force algorithm.
    Returns how many positions the first sequence is shifted at the beginning
    and how many at the end. The sequences must have not indels and the first
    sequence must be shorter than the second"""

    three2one = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "UNK": "U",
    }
    # convert both lists from 3 letter code to one letter code
    one0_list = []
    one1_list = []
    for res in three0_list:
        one0_list.append(three2one[res])
    for res in three1_list:
        one1_list.append(three2one[res])
    # convert one letter code list to strings
    a_seq = "".join(one0_list)
    b_seq = "".join(one1_list)
    # get the length of the sequences
    len_a_seq = len(a_seq)
    len_b_seq = len(b_seq)
    # create two new sequences of the same length
    seq_1 = len(b_seq) * "." + a_seq
    seq_2 = b_seq + len(a_seq) * "."
    # compare both sequences, the trick is to delete (iteratively) the first chacter
    # of sequence 1 and the last of sequence 2.
    dif = []
    for shift in range(len(seq_1) - 1):
        seq_1 = seq_1[1:]
        seq_2 = seq_2[0 : len(seq_2) - 1]
        matching = 0
        for i in range(len(seq_1)):
            if seq_1[i] == seq_2[i]:
                matching += 1
        dif.append(matching)
    maximun = max(dif)
    for values in range(len(dif)):
        if maximun == dif[values]:
            beginning = len_b_seq - (values + 1)
            endding = len_b_seq - len_a_seq - beginning
    return beginning, endding




def load(path):
    """
    Load cheshift's look-up table as a dictionary of arrays.

    The keys are the 3-letter code for amino acids.
    """
    aminoacids = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TYR",
        "TRP",
        "VAL",
    ]
    Db = {}
    for aminoacid in aminoacids:
        Db[aminoacid] = np.fromfile("%s/CS_db_%s" % (path, aminoacid), sep=" ").reshape(
            -1, 6
        )
    return Db


def pdb_parser(filename):
    residues = []
    PDB_list = []
    for line in open( filename).readlines():
        if line[0:5] == "ATOM " and (line[16:17] == " " or line[16:17] == "A"):
            PDB_list.append(line)
    res_num_first = int(PDB_list[0][22:26])
    res_num_last = int(PDB_list[-1][22:26])
    total_res = res_num_last - res_num_first + 1
    coord = np.zeros((total_res, 6, 3))
    count = 0
    for res_num in range(res_num_first, res_num_last + 1):
        for line in PDB_list:
            if res_num == int(line[22:26]):
                atom_name = line[12:16]
                if "H" not in atom_name[0:3]:
                    if "N  " in atom_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        a = np.array([x, y, z])
                        coord[count][0] = a
                    if "CA " in atom_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        a = np.array([x, y, z])
                        coord[count][1] = a
                        residues.append(line[17:20])
                    if "C  " in atom_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        a = np.array([x, y, z])
                        coord[count][2] = a
                    if "CB " in atom_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        a = np.array([x, y, z])
                        coord[count][3] = a
                    if "G1" in atom_name or "G " in atom_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        a = np.array([x, y, z])
                        coord[count][4] = a
                    if "D1" in atom_name or "D " in atom_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        a = np.array([x, y, z])
                        coord[count][5] = a
        count += 1
    return residues, coord


def get_torsional(seg_AB, seg_BC, seg_CD):
    seg_Aseg_BC_plane = np.cross(seg_AB, seg_BC)
    seg_Bseg_CD_plane = np.cross(seg_BC, seg_CD)

    dot_plane = sum(seg_Aseg_BC_plane * seg_Bseg_CD_plane)
    seg_Aseg_BC_mod = np.sqrt(sum((seg_Aseg_BC_plane) ** 2))
    seg_Bseg_CD_mod = np.sqrt(sum((seg_Bseg_CD_plane) ** 2))
    cos = dot_plane / (seg_Aseg_BC_mod * seg_Bseg_CD_mod)
    if cos < -1.0:  # a trick to avoid problems when dihe_rad is 180.0
        cos = -1.0
    dihe_rad = np.arccos(cos) * 180 / 3.1415926
    sign = sum(seg_Aseg_BC_plane * seg_CD)
    if sign > 0:
        dihe_rad = -dihe_rad
        return dihe_rad
    else:
        return dihe_rad


def get_omega(res):
    # CA-C-N-CA
    try:
        if res == len(residues) - 1:
            return float("nan")
        else:
            seg_AB = residuescoord[res][1] - residuescoord[res][2]
            seg_BC = residuescoord[res][2] - residuescoord[res + 1][0]
            seg_CD = residuescoord[res + 1][0] - residuescoord[res + 1][1]
            dihe_rad = get_torsional(seg_AB, seg_BC, seg_CD)
            return dihe_rad
    except:
        return float("nan")


def get_psi(res):
    # N-CA-C-N
    try:
        if res == len(residues) - 1:
            return float("nan")
        else:
            seg_AB = residuescoord[res][0] - residuescoord[res][1]
            seg_BC = residuescoord[res][1] - residuescoord[res][2]
            seg_CD = residuescoord[res][2] - residuescoord[res + 1][0]
            dihe_rad = get_torsional(seg_AB, seg_BC, seg_CD)
            return dihe_rad
    except:
        return float("nan")


def get_phi(res):
    # C-N-Ca-C
    try:
        if res == 0:
            return float("nan")
        else:
            seg_AB = residuescoord[res - 1][2] - residuescoord[res][0]
            seg_BC = residuescoord[res][0] - residuescoord[res][1]
            seg_CD = residuescoord[res][1] - residuescoord[res][2]
            dihe_rad = get_torsional(seg_AB, seg_BC, seg_CD)
            return dihe_rad
    except:
        return float("nan")


def get_chi1(res):
    # N-Ca-Cb-Cg
    try:
        if residues[res] in ["ALA", "GLY", "PRO"]:
            return float("nan")
        else:
            seg_AB = residuescoord[res][0] - residuescoord[res][1]
            seg_BC = residuescoord[res][1] - residuescoord[res][3]
            seg_CD = residuescoord[res][3] - residuescoord[res][4]
            dihe_rad = get_torsional(seg_AB, seg_BC, seg_CD)
            return dihe_rad
    except:
        return float("nan")


def get_chi2(res):
    # Ca-Cb-Cg-Cd
    try:
        if residues[res] in ["ALA", "GLY", "PRO", "SER", "THR", "VAL", "CYS"]:
            return float("nan")
        else:
            seg_AB = residuescoord[res][1] - residuescoord[res][3]
            seg_BC = residuescoord[res][3] - residuescoord[res][4]
            seg_CD = residuescoord[res][4] - residuescoord[res][5]
            dihe_rad = get_torsional(seg_AB, seg_BC, seg_CD)
            return dihe_rad
    except:
        return float("nan")


# a solitare dictionary with the chi2 angles in the CheShift look-up table
chi2_rot = {
    "ARG": np.array((-60, 60, 180)),
    "ASN": np.array((-75, 20, 30)),
    "ASP": np.array((-15, 0)),
    "GLN": np.array((-65, 65, 180)),
    "GLU": np.array((-60, 60, 180)),
    "HIS": np.array((-75, 60, 75)),
    "ILE": np.array((-60, 60, 180)),
    "LEU": np.array((65, 175)),
    "LYS": np.array((-60, 60, 180)),
    "MET": np.array((-60, 60, 180)),
    "PHE": np.array((-90, 90)),
    "TRP": np.array((-105, -90, 90)),
    "TYR": np.array((-85, 80)),
}


def nearest_chi2(chi2, res_name):
    """Compute the nearest angle in chi2_rot given a chi2 angle
    and the res_name of a residue"""
    angles = chi2_rot[res_name]
    index = []
    for angle in angles:
        index.append(min(abs(angle - chi2), abs(angle + 360 - chi2)))
    return angles[np.array(index).argsort()][0]


def nearest_omega(omega):
    """Compute the nearest angle (0 or 180) to the given omega angle"""
    if omega <= -90:
        omega = 180
    omega_db = np.array([0, 180])
    return omega_db[(np.abs(omega_db - omega)).argmin()]


def round_down_up(x, degree):
    """Round a number down and up to multiples of degree"""
    down = np.floor(x / degree) * degree
    up = down + degree
    if up > 180:
        up = down
        down = 180 - degree
    return down, up


def compute_cs(phi, psi, chi1, chi2, omega, res_name, Db):
    """Computes the 13Ca and 13Cb theoretical chemical shifts by linear interpolation from the
    CheShift look-up table, Db. The computed Chemical Shifs are assumed to be TMS-referenced"""

    # phi and psi angles in Db are compute in a 10 degree grid.
    if np.isnan(psi):
        return np.nan, np.nan
    else:
        if res_name == "PRO":
            # for proline only the omega and psi angles are taken into account
            # this is the only residue that use the omega angle, for the others
            # omega is assumed to be 180 (trans)
            data = Db[res_name][(Db[res_name][:, 0] == nearest_omega(omega))]
            return griddata(data[:, 1], data[:, 4:], (psi))
    if np.isnan(phi):
        return np.nan, np.nan
    else:
        phi_range = round_down_up(phi, 10)
        psi_range = round_down_up(psi, 10)

        if res_name in ["ALA", "GLY"]:
            s_Db = Db[res_name][
                (Db[res_name][:, 0] >= phi_range[0])
                & (Db[res_name][:, 0] <= phi_range[1])
            ]
            data = s_Db[(s_Db[:, 1] >= psi_range[0]) & (s_Db[:, 1] <= psi_range[1])]
            return griddata(data[:, :2], data[:, 4:], (phi, psi))
        elif res_name in ["SER", "THR", "VAL", "CYS"]:
            # chi1 angles in Db are compute in a 30 degree grid.
            chi1_range = round_down_up(chi1, 30)
            s_Db = Db[res_name][
                (Db[res_name][:, 0] >= phi_range[0])
                & (Db[res_name][:, 0] <= phi_range[1])
            ]
            ss_Db = s_Db[(s_Db[:, 1] >= psi_range[0]) & (s_Db[:, 1] <= psi_range[1])]
            data = ss_Db[
                (ss_Db[:, 2] >= chi1_range[0]) & (ss_Db[:, 2] <= chi1_range[1])
            ]
            return griddata(data[:, :3], data[:, 4:], (phi, psi, chi1))
        else:
            # chi2 is not part of the linear interpolation, instead the nearest
            # value in Db is used. This has historical roots, and could be changed
            # in the future.
            chi1_range = round_down_up(chi1, 30)
            s_Db = Db[res_name][
                (Db[res_name][:, 0] >= phi_range[0])
                & (Db[res_name][:, 0] <= phi_range[1])
            ]
            ss_Db = s_Db[(s_Db[:, 1] >= psi_range[0]) & (s_Db[:, 1] <= psi_range[1])]
            sss_Db = ss_Db[
                (ss_Db[:, 2] >= chi1_range[0]) & (ss_Db[:, 2] <= chi1_range[1])
            ]
            data = sss_Db[(sss_Db[:, 3] == nearest_chi2(chi2, res_name))]
            return griddata(data[:, :3], data[:, 4:], (phi, psi, chi1))


def CheShift(filename, ocs_file, Db, reference):
    global residues, residuescoord
    residues, residuescoord = pdb_parser(filename)
    chemical_shifts = []
    residueNumber = 0
    for line in open(os.path.join("data", ocs_file)
    ).readlines():  # compute chemical shift only for residues with observed
        if line.split()[1] == "UNK":  # chemical shifts
            a = np.nan
            chemical_shifts.append(a)
            phi = get_phi(residueNumber)
            psi = get_psi(residueNumber)
            chi1 = get_chi1(residueNumber)
            chi2 = get_chi2(residueNumber)
            omega = get_omega(residueNumber)
            residueName = residues[residueNumber]
            residueNumber += 1
        else:
            residueName = residues[residueNumber]
            try:
                residueNameNext = residues[residueNumber + 1]
            except:
                residueNameNext = "GLY"
            if residueName != "CYS":
                try:
                    phi = get_phi(residueNumber)
                    psi = get_psi(residueNumber)
                    chi1 = get_chi1(residueNumber)
                    chi2 = get_chi2(residueNumber)
                    omega = get_omega(residueNumber)
                    values_Ca_New, values_Cb_New = compute_cs(
                        phi, psi, chi1, chi2, omega, residueName, Db
                    )
                except ValueError:
                    values_Ca_New = np.nan
                    values_Cb_New = np.nan
            elif residueName == "CYS":
                values_Ca_New = np.nan
                values_Cb_New = np.nan
            if residueNameNext == "PRO":
                a = round((values_Ca_New - 1.95 + reference), 2)
                chemical_shifts.append(a)
            else:
                a = round((values_Ca_New + reference), 2)
                chemical_shifts.append(a)
            residueNumber += 1
    return np.array(chemical_shifts), residues


def rmsd(
    residues,
    conformations_num,
    pdb_filename,
    cs_exp_name,
    Db,
    reference,
    results,
    cs_exp,
):
    cs_theo_array = np.empty((conformations_num, residues))
    for conf in range(conformations_num):
        conf_list, res_names = CheShift(
            os.path.join("data","%s_%03d.pdb" % (pdb_filename, conf)),
            "%s.ocs" % (cs_exp_name),
            Db,
            reference,
        )
        cs_theo_array[conf] = conf_list
    cs_theo_ave = np.nanmean(cs_theo_array, axis=0)
    exp_list = []
    for line in open(os.path.join("data", f"{cs_exp_name}.ocs")).readlines():
        exp_list.append(float(line.split()[2]))

    for i in range(residues):
        residueName = res_names[i]
        exp_ca = exp_list[i]
        theo_ca = cs_theo_ave[i]
        if np.isnan(theo_ca):
            theo_ca = 999.00
        results.write(f"{pdb_filename},{residueName},{cs_exp},{exp_ca},{theo_ca}\n")


def clean():
    # remove the intermediate files
    try:
        for i in range(10000):
            os.remove(os.path.join("data","%s_%03d.pdb" % (pdb_filename, i)))
    except:
        pass
    try:
        os.remove(os.path.join("data","%s.ocs" % cs_exp))
    except:
        pass


def write_teo_cs(pdb_filename, bmrb_code):
    path = os.path.join("cheshift", "CS_DB")
    nuclei = 2
    Db = load(path)

    fd = [(pdb_filename, bmrb_code)]
    results = open(os.path.join("data", f"{fd[0][0].lower()}_cs_theo_exp.csv"), "w")
    for pair in fd:
        pdb_filename, cs_exp = pair
        conformations_num = split_pdb(pdb_filename)
        reference = bmrb2cheshift(cs_exp)
        ok, pdb_list, ocslist_full_new = check_seq(pdb_filename, bmrb_code)
        if ok:
            rmsd(
                len(ocslist_full_new),
                conformations_num,
                pdb_filename,
                bmrb_code,
                Db,
                reference,
                results,
                cs_exp,
            )
        clean()
