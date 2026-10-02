# LB-PaCS-MD-Ligand-Binding-Pathway-Sampling

**Automated LB-PaCS-MD workflow (GROMACS): ligand unbinding/binding pathway sampling with convergence control, PBC fix and VMD visualisation, 100 ns production.**

**Ligand-Binding Parallel Cut-and-Stitch MD (LB-PaCS-MD)** as a reproducible, configuration-driven pipeline on top of GROMACS:
build the system, generate the unbound start, run replica cycles until the ligand-pocket distance converges, then stitch the
pathway, fix PBC, render with VMD and launch the 100 ns production run.

> بالعربي: سير عمل أوتوماتيكي كامل لـ PaCS-MD لاستكشاف مسار ارتباط الليكاند بالبروتين، مع فحص التقارب وتصحيح PBC وتصوير VMD.

## What's inside

- `run.sh` - launcher (`bash run.sh [config.yaml]`); expects your `complex.pdb` in the project root.
- `config.yaml` - every parameter: force field, SwissParam, box/salt, unbound displacement, PaCS replicas/cycles/convergence.
- `scripts/lb_pacs.py` - the pipeline engine.
- `mdp/` - EM, ions, NVT, NPT, PaCS short-MD and 100 ns production settings.
- `visualization/` - `build_raw_lineage.py`, `make_vmd_final_pbc.py` (VMD scripts with PBC correction).
- `ENVIRONMENT.txt` - software versions used for the reference run (GROMACS 2025.3, CUDA).
- `docs/LB_PaCS_MD_Final_Tutorial_AR_v1.1.docx` - the definitive Arabic run book (v1.1, 2026-09-04).

## Notes

- Not redistributed here: the CHARMM36 force-field folder (`charmm36*.ff`, get it from the MacKerell lab), old `.before_*` script backups, bytecode.
- Cluster paths/users are placeholders.


## Repository map

- `ENVIRONMENT.txt`
- `config.yaml`
- `docs/LB_PaCS_MD_Final_Tutorial_AR_v1.1.docx`
- `mdp/displaced_em.mdp`
- `mdp/em.mdp`
- `mdp/ions.mdp`
- `mdp/npt.mdp`
- `mdp/nvt.mdp`
- `mdp/pacs.mdp`
- `mdp/production_100ns.mdp`
- `run.sh`
- `scripts/lb_pacs.py`
- `visualization/build_raw_lineage.py`
- `visualization/make_vmd_final_pbc.py`

## Author

Yaseen Saleem Hamdoon - Pharmaceutical chemist & computational drug designer, Lecturer in Organic Pharmaceutical Chemistry, Al-Kitab University (Kirkuk, Iraq).

## License

MIT - see `LICENSE`. Third-party data/tools keep their own licenses.
