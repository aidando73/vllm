```bash
docker exec -it aidan bash
source ~/miniconda3/bin/activate ./env
uv pip install -e .

source ~/miniconda3/bin/activate /home/aidan/vllm/env
CUDA_VISIBLE_DEVICES=6 vllm serve /shared/text-models/hf/qwen2p5-vl-7b-instruct/ \
    --enable-lora \
    --lora-modules qwen2p5-vl-lora=/home/aidan/fireworks/do_not_commit/train_output-06-19_05-02/ \
    --max-lora-rank 32 \
    --model-impl transformers \
    --port 80 \
    --host 0.0.0.0
```