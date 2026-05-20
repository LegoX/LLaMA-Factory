cd /jyx_data
bash Miniconda3-latest-Linux-x86_64.sh
cd /jyx_data/jyx_data
chmod -R +x miniconda3/bin/
miniconda3/bin/conda init bash
source ~/.bashrc

apt-get install tmux

conda create -n lf_v3 python=3.12 -y
conda activate lf_v3
cd /jyx_data/LLaMA-Factory-latest


pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

pip install -e .
pip install -r requirements/metrics.txt
pip install -r requirements/deepspeed.txt
pip install -r requirements/liger-kernel.txt

pip install flash-attn --no-build-isolation

pip install wandb

pip install -U "flash-linear-attention>=0.4.1"

pip install tilelang