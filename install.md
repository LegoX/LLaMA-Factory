cd /jyx_data
bash Miniconda3-latest-Linux-x86_64.sh
cd /jyx_data/jyx_data
chmod -R +x miniconda3/bin/
miniconda3/bin/conda init bash
source ~/.bashrc

apt-get install tmux

conda create -n lf_v2 python=3.12 -y
conda activate lf_v2
cd /jyx_data/LLaMA-Factory-latest


pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r requirements/metrics.txt
pip install -r requirements/deepspeed.txt
pip install -r requirements/liger-kernel.txt

# install flash-attn
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install wandb

pip install -U "flash-linear-attention>=0.4.1"

pip install tilelang