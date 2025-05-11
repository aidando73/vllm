```bash
docker exec -it aidan bash
git remote set-url origin 

source ~/miniconda3/bin/activate
VLLM_USE_PRECOMPILED=1 pip install --editable .
```