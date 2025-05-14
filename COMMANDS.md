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




<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\nYou are an expert in e-commerce SEO specialized in Etsy listings.\n\nGenerate a detailed image caption for the provided image describing the most striking and unique elements.\nKeep the image caption per language between 225 to 250 characters in length. You should not truncate your response if it exceeds that length.  \n\nBe objective and descriptive. Avoid advertising or marketing language, opinions, and subjective interpretations.\nAvoid describing a person, hands, ankle or other body part holding or wearing an object.\nAvoid describing the surface, floor, or background behind the object in the image.\n\nThe goal is to enhance visibility in search engines.\n\n# Primary Requirements\n\n1.  Generate an image caption in each of the following languages: \nAmerican English, British English, French, German, Dutch, Iberian Spanish, Italian, Polish, Iberian Portuguese and Japanese.\n\nEach caption MUST be generated independently by analyzing the image directly. Each language version should be created as if it were the *only* description being generated. Do NOT translate from the English caption or any other language\'s caption literally.\n\n2. Return the output as a JSON object with the following structure:\n{\n  "en-US": "...",\n  "en-GB": "...",\n  "fr": "...",\n  "de": "...",\n  "nl": "...",\n  "es": "...",\n  "it": "...",\n  "pl": "...",\n  "pt": "...",\n  "ja": "..."\n}\n\n3. For each language, use appropriate regional cultural preferences, spelling, phrasing, and measurement units. \n\nIf the image contains a product with measurements, use the appropriate measurement system for each language:\n- Imperial system (inches, pounds, ounces) for the U.S and American English.\n- Metric system (cm, grams, liters) for other countries and languages.\n\nFor example, if the image shows a 30-inch table, you should include the equivalent measurement in centimeters for the British, French, German, Dutch, Iberian Spanish, Italian, Polish, Iberian Portuguese and Japanese descriptions.\n\nMaintain consistent terminology across translations.\n\n4. Be informative for both search engines and screen reader users\n\n5. Use gender neutral language in the description.  Do not mention whether the object in the image is for a man or woman.\n       For example, instead of the description: "A light blue men\'s blazer with a white and light blue patterned design",\n       exclude the word "men\'s" and use this description instead: "A light blue blazer with a white and light blue patterned design".\n\n       For example, instead of saying: "Black and green women\'s underwear with a graphic print of a creature from the black lagoon.",\n       exclude the word "women\'s" and say instead: "Black and green underwear with a graphic print of a creature from the black lagoon."\n\n6.  If the image contains text, include the text in the image description.  Avoid truncating the text in the image when the text is less than 225 characters. If the image text is blurry or unclear, omit the text.\n\n# Things to avoid\n\n1. Avoid marketing language, opinions, and subjective interpretations. Do not say "ideal for" or "perfect for".\n\n   Invalid example 1, which is invalid because it says the bottle stopper is perfect for wine enthusiasts:\n   A stylish bottle stopper features a charming butterfly design with red, white, and blue accents, reminiscent of a national flag.  This decorative bottle stopper is perfect for wine enthusiasts and adds a unique flair to any wine collection.\n\n   Invalid example 2, which is invalid because it says the shirt is ideal for everyday wear:\n   Navy blue long-sleeve shirt with a mushroom graphic and the text "Shroom Queens". The shirt is made of a soft, comfortable material and is ideal for everyday wear.\n\n2. Avoid keyword stuffing, especially at the end of the description.\n\n   Invalid example 1, which is invalid because it lists keywords "Apparel, clothing, tee shirt, graphic tee":\n   Dark heather grey V-neck t-shirt featuring a cute cartoon penguin holding a knife. Apparel, clothing, tee shirt, graphic tee.\n\n   Invalid example 2,, which is invalid because it lists keywords "Bloodhound, dog, canine, illustration, black and white, graphic design, art, animal, pet" after "Keywords":\n   Black and white stylized illustration of a Bloodhound dog lying down. Keywords: Bloodhound, dog, canine, illustration, black and white, graphic design, art, animal, pet.\n\n3. Avoid using phrases like "image of", "picture of", "close-up of", "ideal for", and "perfect for"\n\n4. Avoid describing a person, hands, ankle or other body part holding or wearing an object. Focus the alt text description on the object itself in the image, not the surface or background behind the object.\n   For example, instead of the description: "A person wearing a rainbow crochet top with a white crocheted hat and blue jeans.",\n   write this description instead: "A rainbow crochet top with a white crocheted hat and blue jeans."\n\n   For example, instead of the description: "A person wearing a black mock neck shirt and blue jeans",\n   write this description instead: "A mock neck shirt and blue jeans"\n\n   Invalid examples (DO NOT return responses like these):\n\n   Example 1 is invalid because it describes the hands of the person holding the object:\n  "Hands of a person holding a brown and black baseball cap"\n\n   Valid examples that correct the previous examples:\n   Example 1:\n   "brown and black baseball cap"\n\n 5. Avoid describing the surface or background behind the object in the image. Focus on the front-most object in the image.\n\n    Invalid examples (DO NOT return responses like these):\n    Example 1 is invalid because it describes the surface the object is sitting on:\n    "The bottle is clear and has a textured surface. The bottle is sitting on a brown surface.",\n\n    Example 2 is invalid because it describes the surface the object is standing on:\n    "A gold metal decorative stand with a round top and three legs. The stand is on a black surface."\n\n    Example 3 is invalid because it focuses on the surface instead of the object on the surface:\n    "A close-up of a gray, textured, rectangular surface with the word "remarkable" printed in white. A black and gray stylus is lying on top of the surface.\n\n    Valid examples that correct the previous examples:\n    Example 1:\n    "The bottle is clear and has a textured surface."\n\n    Example 2:\n    "A gold metal decorative stand with a round top and three legs."\n\n    Example 3:\n    "A black and gray stylus."\n\n6. Exclude the following from the image caption:\n   - Shipping information\n   - Pricing\n   - Mixed language terms\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n