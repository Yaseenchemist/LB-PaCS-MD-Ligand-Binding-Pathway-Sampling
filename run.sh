#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${1:-config.yaml}"

echo "============================================================"
echo " LB-PaCS-MD launcher"
echo "============================================================"
echo "Project root : $ROOT"
echo "Config       : $CONFIG"
echo

[[ -f "$CONFIG" ]] || {
    echo "ERROR: Missing configuration file: $CONFIG" >&2
    exit 2
}

[[ -s complex.pdb ]] || {
    echo "ERROR: Missing or empty complex.pdb in $ROOT" >&2
    exit 2
}

if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
    export PATH="$ROOT/.venv/bin:$PATH"
    echo "[OK] Using project Python: $PYTHON"
else
    PYTHON="$(command -v python3 || true)"
    [[ -n "$PYTHON" ]] || {
        echo "ERROR: python3 not found." >&2
        exit 2
    }
    echo "[INFO] .venv not found; using: $PYTHON"
fi

PROFILE="${LBPACS_PROFILE:-auto}"

if [[ "$PROFILE" == "auto" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
        PROFILE="slurm_gpu"
    else
        PROFILE="local"
    fi
fi

case "$PROFILE" in
    local|slurm_gpu) ;;
    *)
        echo "ERROR: Invalid profile '$PROFILE'" >&2
        exit 2
        ;;
esac

echo "[OK] Execution profile: $PROFILE"

if [[ "$PROFILE" == "slurm_gpu" ]]; then
    command -v module >/dev/null 2>&1 || {
        echo "ERROR: module command not found" >&2
        exit 2
    }

    GMX_MODULE="${LBPACS_GMX_MODULE:-GROMACS/2025.3-foss-2025a-CUDA-12.9.1}"

    echo "[INFO] Loading GROMACS module:"
    echo "       $GMX_MODULE"

    module purge
    module load "$GMX_MODULE"

    # Prevent Python 3.13/SciPy-bundle injected by EasyBuild
    # from contaminating the project's Python 3.12 virtualenv.
    unset PYTHONPATH || true
    unset PYTHONHOME || true
    export PYTHONNOUSERSITE=1

    # Re-prioritize project tools after EasyBuild module changes PATH.
    export PATH="$ROOT/.venv/bin:$PATH"

    command -v gmx >/dev/null 2>&1 || {
        echo "ERROR: gmx not found after loading module" >&2
        exit 2
    }

    echo "[OK] GROMACS: $(command -v gmx)"
    gmx --version | head -n 20
fi

echo
echo "[CHECK] Python:"
"$PYTHON" --version

echo
echo "[CHECK] Python modules:"
"$PYTHON" - <<'PY'
import sys

required = {
    "yaml": "PyYAML",
    "numpy": "numpy",
    "MDAnalysis": "MDAnalysis",
    "matplotlib": "matplotlib",
    "pexpect": "pexpect",
}

missing = []

for modname, package in required.items():
    try:
        mod = __import__(modname)
        ver = getattr(mod, "__version__", "")
        print(f"  OK   {modname:<14} {ver}")
    except Exception as e:
        print(f"  MISS {modname:<14} {e}")
        missing.append(package)

if missing:
    print("\nERROR: Missing Python packages:")
    for p in missing:
        print("  -", p)
    sys.exit(20)
PY

echo
echo "[CHECK] External programs:"

command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl not found" >&2
    exit 2
}
echo "  OK   curl -> $(command -v curl)"

if command -v obabel >/dev/null 2>&1; then
    echo "  OK   obabel -> $(command -v obabel)"
else
    echo "ERROR: obabel not found" >&2
    exit 2
fi

FF_DIR="$(find "$ROOT" -maxdepth 1 -type d -name 'charmm36*.ff' -print -quit)"

[[ -n "$FF_DIR" ]] || {
    echo "ERROR: No charmm36*.ff directory found" >&2
    exit 2
}

echo
echo "[OK] Force field: $FF_DIR"

export GMXLIB="$ROOT"
echo "[OK] GMXLIB=$GMXLIB"

echo
echo "============================================================"
echo " Starting LB-PaCS-MD controller"
echo "============================================================"
echo

exec "$PYTHON" "$ROOT/scripts/lb_pacs.py" \
    --config "$CONFIG" \
    --profile "$PROFILE"
