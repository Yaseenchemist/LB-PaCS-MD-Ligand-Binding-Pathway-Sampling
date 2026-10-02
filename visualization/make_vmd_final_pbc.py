import json
import numpy as np
import MDAnalysis as mda

from pathlib import Path
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.coordinates.XTC import XTCWriter


ROOT = Path("/path/to/your_project/DYNAMIC/LB_PaCS_MD_final")

TPR = ROOT / "work/04_pacs/cycle_001/rep_04/md.tpr"

IN_XTC = ROOT / (
    "visualization/pacs_binding_movie/"
    "processed_final/pacs_ancestral_whole.xtc"
)

OUTDIR = ROOT / (
    "visualization/pacs_binding_movie/vmd_final"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_XTC = OUTDIR / "pacs_binding_final_pbc.xtc"
OUT_GRO = OUTDIR / "pacs_binding_final_pbc.gro"


# ============================================================
# Pocket definition
# ============================================================

with open(ROOT / "work/03_reference/reference.json") as f:
    info = json.load(f)

pidx = np.asarray(
    info["pocket_atom_indices"],
    dtype=int
)


# ============================================================
# Mobile + independent reference
# ============================================================

u = mda.Universe(
    str(TPR),
    str(IN_XTC)
)

ref = mda.Universe(
    str(TPR),
    str(IN_XTC)
)

ref.trajectory[0]


# ============================================================
# Selections
# ============================================================

lig = u.select_atoms("resname LIG")

pocket = u.atoms[pidx]

pocket_resindices = np.unique(
    pocket.resindices
)

ca_indices = np.asarray([
    atom.index
    for atom in u.atoms
    if atom.name == "CA"
    and atom.resindex in pocket_resindices
], dtype=int)

if len(ca_indices) < 3:
    raise RuntimeError(
        f"Too few pocket CA atoms: {len(ca_indices)}"
    )

pocket_ca = u.atoms[ca_indices]

sel = "index " + " ".join(
    str(i) for i in ca_indices
)


# ============================================================
# IMPORTANT:
# Reference ligand must also be imaged nearest reference pocket
# before using reference for alignment/display.
# ============================================================

ref_lig = ref.select_atoms("resname LIG")
ref_pca = ref.atoms[ca_indices]

raw_vec = (
    ref_lig.center_of_mass()
    - ref_pca.center_of_mass()
)

target_vec = minimize_vectors(
    raw_vec.reshape(1,3),
    ref.dimensions
)[0]

ref_shift = target_vec - raw_vec

ref_lig.translate(ref_shift)


print()
print("========== FINAL PBC-SAFE VMD BUILD ==========")
print(f"Input frames          : {len(u.trajectory)}")
print(f"Total atoms           : {u.atoms.n_atoms}")
print(f"Ligand atoms          : {lig.n_atoms}")
print(f"Pocket atoms          : {len(pidx)}")
print(f"Pocket CA atoms       : {len(ca_indices)}")
print()
print("Input                 : GROMACS -pbc mol trajectory")
print("Ligand imaging        : nearest periodic pocket image")
print("Alignment             : pocket C-alpha")
print("Transform applied to  : COMPLETE SYSTEM")
print("Ligand fitting        : NONE")
print("Interpolation         : NONE")
print()


# ============================================================
# Remove only outputs from this script
# ============================================================

for f in (OUT_XTC, OUT_GRO):
    if f.exists():
        f.unlink()


records = []
shifted_frames = []


# ============================================================
# Build trajectory
# ============================================================

with XTCWriter(
    str(OUT_XTC),
    n_atoms=u.atoms.n_atoms
) as writer:

    for ts in u.trajectory:

        # ----------------------------------------------------
        # 1. Put complete ligand in periodic image nearest
        #    physical binding pocket.
        # ----------------------------------------------------

        lc = lig.center_of_mass()
        pc = pocket_ca.center_of_mass()

        raw_vec = lc - pc

        target_vec = minimize_vectors(
            raw_vec.reshape(1,3),
            ts.dimensions
        )[0]

        shift = target_vec - raw_vec

        shift_norm = np.linalg.norm(shift)

        if shift_norm > 0.1:
            shifted_frames.append(
                (ts.frame, shift_norm)
            )

        # Translate COMPLETE ligand only.
        # This is an integer-box PBC imaging operation.
        lig.translate(shift)


        # ----------------------------------------------------
        # Distance BEFORE visual alignment
        # ----------------------------------------------------

        d_before = np.linalg.norm(
            lig.center_of_mass()
            - pocket_ca.center_of_mass()
        )


        # ----------------------------------------------------
        # 2. Visual stabilization:
        #    calculate fit using pocket CA,
        #    apply rigid transformation to COMPLETE system.
        # ----------------------------------------------------

        align.alignto(
            u,
            ref,
            select={
                "mobile": sel,
                "reference": sel
            },
            weights="mass",
            subselection=None
        )


        # ----------------------------------------------------
        # Distance AFTER alignment.
        # A rigid transform MUST preserve this distance.
        # ----------------------------------------------------

        d_after = np.linalg.norm(
            lig.center_of_mass()
            - pocket_ca.center_of_mass()
        )

        delta = abs(
            d_after - d_before
        )

        if delta > 1.0e-3:
            raise RuntimeError(
                f"Rigid transform changed ligand-pocket "
                f"distance at frame {ts.frame}: "
                f"{d_before:.6f} -> {d_after:.6f} A"
            )


        # ----------------------------------------------------
        # Save first aligned/imaged structure
        # ----------------------------------------------------

        if ts.frame == 0:
            u.atoms.write(
                str(OUT_GRO)
            )


        writer.write(
            u.atoms
        )

        records.append(
            (
                ts.frame,
                ts.time,
                d_after,
                shift_norm
            )
        )


# ============================================================
# Report
# ============================================================

print("BUILD COMPLETE")
print(f"Frames written         : {len(records)}")
print(f"PBC-shifted frames     : {len(shifted_frames)}")

if shifted_frames:
    max_shift = max(
        shifted_frames,
        key=lambda x: x[1]
    )

    print(
        f"Maximum ligand shift   : "
        f"{max_shift[1]:.3f} A "
        f"(frame {max_shift[0]})"
    )

print()
print("Selected checkpoints:")
print(
    "frame   time_ps   ligand-pocket   ligand_PBC_shift"
)
print(
    "---------------------------------------------------"
)

checkpoints = [
    0,25,50,
    74,75,76,77,78,79,80,81,82,83,84,85,
    97,98,
    100,101,102,103,104,105,106,
    120,150,175,196
]

for i in checkpoints:

    fr, t, d, sh = records[i]

    print(
        f"{fr:5d} "
        f"{t:9.1f} "
        f"{d:14.3f} "
        f"{sh:17.3f}"
    )

print()
print(f"GRO : {OUT_GRO}")
print(f"XTC : {OUT_XTC}")
