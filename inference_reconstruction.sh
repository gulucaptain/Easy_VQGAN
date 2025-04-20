# Inference: original three types vq-gan and dall_e
CUDA_VISIBLE_DEVICES=0 python inference_stage_1.py

# Inference: vq-gan model before and after finetuned on your custom data
CUDA_VISIBLE_DEVICES=0 python inference_stage_2.py
