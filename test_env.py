# test_env.py
import torch
import networkx
from gliner import GLiNER

print("Torch:", torch.__version__)
print("NetworkX:", networkx.__version__)

model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
print("GLiNER loaded OK")