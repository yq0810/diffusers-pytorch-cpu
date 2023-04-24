#!/usr/bin/env python3
import sys
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)  
pipeline.enable_attention_slicing()
pipeline.safety_checker = lambda images, clip_input: (images, False)

pipeline.to("cpu")


prompt = "cat"
results = pipeline([prompt], num_inference_steps=20, height=320, width=320)
image = results.images[0]
image.save("/root/my_image.png")
