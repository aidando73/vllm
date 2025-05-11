```bash
docker exec -it aidan bash
git remote set-url origin 

conda create --prefix ./env python=3.10

cd ~/vllm

source ~/miniconda3/bin/activate && conda activate ./env
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

wget https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.tar.gz
tar -xzf cmake-3.31.7-linux-x86_64.tar.gz
mv cmake-3.31.7-linux-x86_64 /opt/cmake-3.31.7-linux-x86_64
echo "export PATH=/opt/cmake-3.31.7-linux-x86_64/bin:$PATH" >> ~/.bashrc
source ~/.bashrc


pip install uv
uv pip install -r requirements/build.txt
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

export CUDA_VISIBLE_DEVICES=6
VLLM_TORCH_PROFILER_DIR=/home/aidan/fireworks/vllm_profile \
    python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-vision-128k-instruct \
    --trust-remote-code


python vllm-req.py
```

```bash
# Old
source ~/miniconda3/bin/activate ./vllm
uv pip install pandas datasets
python ~/vllm/benchmarks/benchmark_serving.py --backend vllm --model microsoft/Phi-3-vision-128k-instruct --num-prompts 1 --dataset-name random --random-input 1024 --random-output 512 --profile
python ~/vllm/benchmarks/benchmark_serving.py --backend vllm --model microsoft/Phi-3-vision-128k-instruct --num-prompts 1 --dataset-name aledade_prod.json --dataset-path 
```