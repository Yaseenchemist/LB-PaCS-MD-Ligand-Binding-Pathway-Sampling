import csv
import json
import shutil
from pathlib import Path

import MDAnalysis as mda


ROOT = Path("/path/to/your_project/DYNAMIC/LB_PaCS_MD_final")
PACS = ROOT / "work" / "04_pacs"

OUT = ROOT / "visualization" / "pacs_binding_movie"
RAW = OUT / "raw"
META = OUT / "metadata"

RAW.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

TPR = PACS / "cycle_001" / "rep_04" / "md.tpr"
XTC = PACS / "cycle_001" / "rep_04" / "md.xtc"
START_GRO = PACS / "start_em.gro"

OUT_XTC = RAW / "pacs_ancestral_raw.xtc"
OUT_GRO = RAW / "pacs_ancestral_structure.gro"
OUT_CSV = META / "lineage_metadata.csv"
OUT_SEL = META / "selected_lineage_states.csv"


# ============================================================
# 1. Reconstruct exact ancestry backward from Cycle 100
# ============================================================

with open(PACS / "cycle_100" / "winner.json") as f:
    final_winner = json.load(f)

child_replica = int(final_winner["replica"])

lineage = {}

for cyc in range(99, 0, -1):

    cd = PACS / f"cycle_{cyc:03d}"

    with open(cd / "selected_seeds.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    matches = [
        r for r in rows
        if int(r["next_replica"]) == child_replica
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cycle {cyc}: expected exactly one ancestry parent "
            f"for next_replica={child_replica}; found {len(matches)}"
        )

    r = matches[0]

    rank = int(r["rank"])

    lineage[cyc] = {
        "cycle": cyc,
        "rank": rank,
        "source_replica": int(r["source_replica"]),
        "source_frame": int(r["source_frame"]),
        "time_ps": float(r["time_ps"]),
        "com_A": float(r["com_A"]),
        "rmsd_A": float(r["rmsd_A"]),
        "trr": cd / f"seed_{rank:02d}_full.trr",
        "gro": cd / f"seed_{rank:02d}.gro",
    }

    child_replica = int(r["source_replica"])


lineage[100] = {
    "cycle": 100,
    "rank": None,
    "source_replica": int(final_winner["replica"]),
    "source_frame": int(final_winner["frame"]),
    "time_ps": float(final_winner["time_ps"]),
    "com_A": float(final_winner["com_A"]),
    "rmsd_A": float(final_winner["rmsd_A"]),
    "trr": PACS / "cycle_100" / "winner_full.trr",
    "gro": PACS / "cycle_100" / "winner.gro",
}


# ============================================================
# 2. Critical ancestry sanity check
# ============================================================

if lineage[1]["source_replica"] != 4:
    raise RuntimeError(
        f"Expected Cycle 1 ancestral replica 4; "
        f"found {lineage[1]['source_replica']}"
    )

if lineage[1]["source_frame"] != 97:
    raise RuntimeError(
        f"Expected Cycle 1 ancestral frame 97; "
        f"found {lineage[1]['source_frame']}"
    )

print("Exact final ancestry reconstructed.")
print(
    "Cycle 1 ancestor : "
    f"replica {lineage[1]['source_replica']} "
    f"frame {lineage[1]['source_frame']}"
)
print()


# ============================================================
# 3. Verify all selected full-system snapshots exist
# ============================================================

for cyc in range(1, 101):

    if not lineage[cyc]["trr"].exists():
        raise FileNotFoundError(lineage[cyc]["trr"])

print("All 100 selected full-system states are present.")


# ============================================================
# 4. Topology / atom-count reference
# ============================================================

u_cont = mda.Universe(str(TPR), str(XTC))

n_atoms = u_cont.atoms.n_atoms

if n_atoms != 76262:
    raise RuntimeError(
        f"Unexpected atom count in Cycle1 trajectory: {n_atoms}"
    )

if len(u_cont.trajectory) < 98:
    raise RuntimeError(
        f"Cycle1 trajectory has only {len(u_cont.trajectory)} frames"
    )

print(f"Reference atoms       : {n_atoms}")
print(f"Cycle1 XTC frames     : {len(u_cont.trajectory)}")


# ============================================================
# 5. Save structure used for the visualization trajectory
# ============================================================

shutil.copy2(START_GRO, OUT_GRO)


# ============================================================
# 6. Build raw trajectory
#
# Frames 0-97:
#   actual continuous Cycle1/Replica4 MD
#
# Remaining frames:
#   actual selected ancestral PaCS states, one per cycle
#
# XTC time is set to ancestral cumulative MD time.
# ============================================================

metadata = []
selected_metadata = []

display_frame = 0

with mda.Writer(
    str(OUT_XTC),
    n_atoms=n_atoms,
    precision=3
) as writer:

    # --------------------------------------------------------
    # Continuous Cycle 1 ancestral segment: frames 0..97
    # --------------------------------------------------------

    for iframe in range(0, 98):

        u_cont.trajectory[iframe]

        source_time = float(u_cont.trajectory.ts.time)

        # Along Cycle 1 the cumulative ancestry time is simply
        # the MD trajectory time.
        cumulative_ps = source_time

        # Use cumulative ancestry time in output XTC.
        u_cont.trajectory.ts.time = cumulative_ps

        writer.write(u_cont.atoms)

        is_selected = (iframe == lineage[1]["source_frame"])

        com_A = (
            lineage[1]["com_A"]
            if is_selected
            else ""
        )

        rmsd_A = (
            lineage[1]["rmsd_A"]
            if is_selected
            else ""
        )

        row = {
            "display_frame": display_frame,
            "frame_class": (
                "cycle1_selected_state"
                if is_selected
                else "continuous_md"
            ),
            "cycle": 1,
            "source_replica": 4,
            "source_frame": iframe,
            "source_time_ps": source_time,
            "ancestral_cumulative_ps": cumulative_ps,
            "selected_rank": (
                lineage[1]["rank"]
                if is_selected
                else ""
            ),
            "com_A": com_A,
            "rmsd_A": rmsd_A,
            "source_file": str(XTC),
        }

        metadata.append(row)

        if is_selected:
            selected_metadata.append(row.copy())

        display_frame += 1


    # --------------------------------------------------------
    # Selected ancestral states from Cycles 2..100
    # --------------------------------------------------------

    cumulative_ps = float(lineage[1]["time_ps"])

    for cyc in range(2, 101):

        d = lineage[cyc]

        cumulative_ps += float(d["time_ps"])

        # Same atom ordering/topology throughout the PaCS run.
        u = mda.Universe(
            str(TPR),
            str(d["trr"])
        )

        if u.atoms.n_atoms != n_atoms:
            raise RuntimeError(
                f"Cycle {cyc}: atom count mismatch "
                f"{u.atoms.n_atoms} != {n_atoms}"
            )

        if len(u.trajectory) != 1:
            raise RuntimeError(
                f"Cycle {cyc}: expected one-frame TRR; "
                f"found {len(u.trajectory)}"
            )

        u.trajectory[0]

        original_source_time = float(u.trajectory.ts.time)

        # Verify the TRR time agrees with selection metadata.
        if abs(original_source_time - float(d["time_ps"])) > 0.01:
            raise RuntimeError(
                f"Cycle {cyc}: TRR time {original_source_time} ps "
                f"does not match metadata {d['time_ps']} ps"
            )

        # Write monotonic ancestral cumulative MD time.
        u.trajectory.ts.time = cumulative_ps

        writer.write(u.atoms)

        row = {
            "display_frame": display_frame,
            "frame_class": (
                "final_winner"
                if cyc == 100
                else "selected_state"
            ),
            "cycle": cyc,
            "source_replica": d["source_replica"],
            "source_frame": d["source_frame"],
            "source_time_ps": d["time_ps"],
            "ancestral_cumulative_ps": cumulative_ps,
            "selected_rank": (
                "WIN"
                if cyc == 100
                else d["rank"]
            ),
            "com_A": d["com_A"],
            "rmsd_A": d["rmsd_A"],
            "source_file": str(d["trr"]),
        }

        metadata.append(row)
        selected_metadata.append(row.copy())

        display_frame += 1


# ============================================================
# 7. Write metadata
# ============================================================

fields = [
    "display_frame",
    "frame_class",
    "cycle",
    "source_replica",
    "source_frame",
    "source_time_ps",
    "ancestral_cumulative_ps",
    "selected_rank",
    "com_A",
    "rmsd_A",
    "source_file",
]

with open(OUT_CSV, "w", newline="") as f:

    wr = csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()
    wr.writerows(metadata)


with open(OUT_SEL, "w", newline="") as f:

    wr = csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()
    wr.writerows(selected_metadata)


# ============================================================
# 8. Final checks
# ============================================================

expected_frames = 98 + 99

if len(metadata) != expected_frames:
    raise RuntimeError(
        f"Expected {expected_frames} visualization frames; "
        f"created {len(metadata)}"
    )

if len(selected_metadata) != 100:
    raise RuntimeError(
        f"Expected 100 selected lineage states; "
        f"created {len(selected_metadata)}"
    )


print()
print("=" * 70)
print("RAW PaCS ANCESTRAL VISUALIZATION TRAJECTORY CREATED")
print("=" * 70)

print(f"Total visualization frames : {len(metadata)}")
print("Continuous MD frames       : 98  (Cycle 1, frames 0-97)")
print(f"Selected lineage states    : {len(selected_metadata)}")
print(
    f"Final cumulative MD time   : "
    f"{float(metadata[-1]['ancestral_cumulative_ps']):.3f} ps"
)
print(
    f"                           = "
    f"{float(metadata[-1]['ancestral_cumulative_ps'])/1000:.6f} ns"
)

print()
print("Trajectory:")
print(f"  {OUT_XTC}")

print()
print("Structure:")
print(f"  {OUT_GRO}")

print()
print("Metadata:")
print(f"  {OUT_CSV}")

print()
print("Selected states metadata:")
print(f"  {OUT_SEL}")

print("=" * 70)
