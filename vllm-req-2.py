
# image_url = "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg?cs=srgb&dl=pexels-christian-heitz-285904-842711.jpg&fm=jpg&w=5472&h=3648&_gl=1*1ns6hbe*_ga*MzI5NzEzMjE3LjE3NDcwNjIwNzA.*_ga_8JE65Q40S6*czE3NDcwNjIwNjkkbzEkZzEkdDE3NDcwNjIwNzMkajAkbDAkaDA."
import requests
import json
import os

if __name__ == "__main__":
#   image_url = "https://images.pexels.com/photos/842711/pexels-photo-842711.jpeg?cs=srgb&dl=pexels-christian-heitz-285904-842711.jpg&fm=jpg&w=640&h=427&_gl=1*1iij6m0*_ga*MzI5NzEzMjE3LjE3NDcwNjIwNzA.*_ga_8JE65Q40S6*czE3NDcwNjY1MTQkbzIkZzAkdDE3NDcwNjY1MTQkajAkbDAkaDA."
  image_url = "https://i.etsystatic.com/il/97a3f4/6483108510/il_fullxfull.6483108510_nxr1.jpg"
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--image_url", type=str, default=image_url)
  args = parser.parse_args()

  url = "http://localhost:8000/v1/chat/completions"
  payload = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "max_tokens": 4096,
    # "top_p": 1,
    # "top_k": 40,
    # "presence_penalty": 0,
    # "frequency_penalty": 0,
    # "temperature": 1.0,
    # "ignore_eos": True,
    "messages": [{
        "role": "user",
        "content": [
          {
              "type": "text",
              "text": "Describe the image in detail."
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": args.image_url
              }
          }
        ]
    }]
  }
  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
  }
  response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
  print("FW Response: ", response)
  from pprint import pprint
  if response.status_code == 200:
    pprint(response.json())
  else:
    print(response.text)