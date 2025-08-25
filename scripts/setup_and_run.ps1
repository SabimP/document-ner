# Create venv, install deps, and run training
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python train_funsd.py --data_root dataset --outdir outputs --layoutlm_epochs 3 --bilstm_epochs 5
