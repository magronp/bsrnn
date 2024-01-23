source .venv/bin/activate

python3 evaluate.py spec_inv.time_domain_tr=True spec_inv.algo=MISI spec_inv.iter=2
python3 evaluate.py spec_inv.time_domain_tr=True spec_inv.algo=MISI spec_inv.iter=3
python3 evaluate.py spec_inv.time_domain_tr=True spec_inv.algo=MISI spec_inv.iter=4
python3 evaluate.py spec_inv.time_domain_tr=True spec_inv.algo=MISI spec_inv.iter=5