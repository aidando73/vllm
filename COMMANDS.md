```bash
docker exec -it aidan bash
source ~/miniconda3/bin/activate ./env

cd /home/aidan/vllm && source ~/miniconda3/bin/activate && conda activate ./env
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

wget https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.tar.gz
tar -xzf cmake-3.31.7-linux-x86_64.tar.gz
sudo mv cmake-3.31.7-linux-x86_64 /opt/cmake-3.31.7-linux-x86_64
echo "export PATH=/opt/cmake-3.31.7-linux-x86_64/bin:$PATH" >> ~/.bashrc
source ~/.bashrc

pip install uv
uv pip install -r requirements/build.txt
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

source ~/miniconda3/bin/activate /home/aidan/vllm/env
CUDA_VISIBLE_DEVICES=6 vllm serve /shared/text-models/hf/qwen2p5-vl-7b-instruct/ \
    --enable-lora \
    --lora-modules qwen2p5-vl-lora=/home/aidan/fireworks/do_not_commit/train_output-06-19_05-02/ \
    --max-lora-rank 32 \
    --model-impl transformers \
    --port 80 \
    --host 0.0.0.0
```