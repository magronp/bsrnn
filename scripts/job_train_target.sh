source .venv/bin/activate
SRCMOD="$1"
TARGET="$2"
SCHEDULER="$3"

python3 train_target.py src_mod=${SRCMOD} src_mod.target=${TARGET} scheduler=${SCHEDULER}