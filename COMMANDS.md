```bash
# Create a docker container specifically for vllm
docker run \
    -d \
    --gpus all \
    --restart always \
    --name aidan2 \
    -v /home/aidan/home:/home/aidan \
    us-docker.pkg.dev/fw-ai-cp-prod/inference/cuda-dev/aidan:latest \
    sleep infinity

# Enter the container
docker exec -it aidan2 bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

conda create --prefix ./env python=3.10

cd ~/vllm

cd /home/aidan/vllm && source ~/miniconda3/bin/activate && conda activate ./env
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

wget https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-x86_64.tar.gz
tar -xzf cmake-3.31.7-linux-x86_64.tar.gz
mv cmake-3.31.7-linux-x86_64 /opt/cmake-3.31.7-linux-x86_64
echo "export PATH=/opt/cmake-3.31.7-linux-x86_64/bin:$PATH" >> ~/.bashrc
source ~/.bashrc

source ~/miniconda3/bin/activate ./env
pip install uv
uv pip install -r requirements/build.txt
VLLM_USE_PRECOMPILED=1 uv pip install --editable .

export CUDA_VISIBLE_DEVICES=6
VLLM_TORCH_PROFILER_DIR=/home/aidan/fireworks/vllm_profile \
    python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-vision-128k-instruct \
    --trust-remote-code \
    --no-enable-prefix-caching


python vllm-req.py

CUDA_VISIBLE_DEVICES=7 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --no-enable-prefix-caching
```

```bash
# Old
source ~/miniconda3/bin/activate ./vllm
uv pip install pandas datasets
python ~/vllm/benchmarks/benchmark_serving.py --backend vllm --model microsoft/Phi-3-vision-128k-instruct --num-prompts 1 --dataset-name random --random-input 1024 --random-output 512 --profile
python ~/vllm/benchmarks/benchmark_serving.py --backend vllm --model microsoft/Phi-3-vision-128k-instruct --num-prompts 1 --dataset-name aledade_prod.json --dataset-path 


firectl-admin create deployment accounts/fireworks/models/qwen2p5-vl-7b-instruct \
  -a etsy \
  --accelerator-type="NVIDIA_H100_80GB" \
  --min-replica-count 1 \
  --display-name "Etsy Qwen 2p5 7b Hackathon 2" \
  --description "Qwen 2p5 7b deployment" \
  --scale-down-window 5m \
  --scale-to-zero-window 10m \
  --engine VLLM \
  --disable-accounting \
  --expire-time 2025-05-16 \
  --region US_IOWA_1
```

pixel_values.shape: torch.Size([1, 17, 3, 336, 336])




```
ues': tensor([[ 0.7041,  1.1712,  1.4194,  ...,  1.9042,  1.8757,  1.8757],
        [ 1.6238,  1.6676,  1.5654,  ...,  1.8757,  1.8899,  1.8899],
        [-1.7193, -1.6463, -1.6755,  ...,  1.6766,  2.0464,  2.0748],
        ...,
        [ 1.4924,  1.5070,  1.5216,  ...,  1.6055,  1.5629,  1.5060],
        [ 1.4778,  1.4924,  1.4924,  ...,  1.3780,  1.6055,  1.5771],
        [ 1.4778,  1.4924,  1.4924,  ...,  1.8331,  1.8046,  1.7904]]),
pixel_values torch.Size([34240, 1176])
```
```
hf [[ 0.7040684  1.171218   1.4193913 ...  1.9041561  1.875716   1.875716 ]
 [ 1.6237692  1.6675645  1.5653756 ...  1.875716   1.8899357  1.8899357]
 [-1.7192703 -1.6462783 -1.6754751 ...  1.6766349  2.0463567  2.074797 ]
 ...
 [ 1.4923834  1.5069818  1.5215802 ...  1.6055346  1.5628744  1.5059941]
 [ 1.477785   1.4923834  1.4923834 ...  1.3780135  1.6055346  1.5770944]
 [ 1.477785   1.4923834  1.4923834 ...  1.8330555  1.8046155  1.7903955]]
hf.pixel_values.shape (34240, 1176)
```

# vLLM Prod
```bash
conda create --prefix ./vllm-prod python=3.10

conda activate ./vllm-prod

pip install uv
uv pip install vllm

CUDA_VISIBLE_DEVICES=3,5,6,7 vllm serve OpenGVLab/InternVL2_5-78B \
    --tensor-parallel-size 4
```