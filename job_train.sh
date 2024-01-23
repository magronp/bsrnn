source .venv/bin/activate
SRCMOD="$1"
TARGET="$2"

python3 train.py src_mod=${SRCMOD} src_mod.target=${TARGET}