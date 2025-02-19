
import torch
import tensorflow as tf
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")
print(f"Device properties: {torch.cuda.get_device_properties(0)}")