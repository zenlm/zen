import mlx.core as mx
from mlx_lm.utils import load

model, tokenizer = load("fused_model")

for name, param in model.parameters().items():
    print(name, param.shape)
