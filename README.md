<img src="./hourglass.png" width="500px"></img>

## Hourglass Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2110.13711">Hourglass Transformer</a>, in Pytorch.


## Install

```bash
$ pip install hourglass-transformer-pytorch
```

## Usage

```python
import torch
from hourglass_transformer_pytorch import HourglassTransformerLM

model = HourglassTransformerLM(
    num_tokens = 256,               # number of tokens
    dim = 512,                      # feature dimension
    max_seq_len = 1024,             # maximum sequence length
    heads = 8,                      # attention heads
    dim_head = 64,                  # dimension per attention head
    shorten_factor = 2,             # shortening factor
    depth = (4, 2, 4),              # tuple of 3, standing for pre-transformer-layers, valley-transformer-layers (after downsample), post-transformer-layers (after upsample) - the valley transformer layers can be yet another nested tuple, in which case it will shorten again recursively
)

x = torch.randint(0, 256, (1, 1024))
logits = model(x) # (1, 1024, 256)
```

For something more sophisticated, two hourglasses, with one nested within the other


```python
import torch
from hourglass_transformer_pytorch import HourglassTransformerLM

model = HourglassTransformerLM(
    num_tokens = 256,
    dim = 512,
    max_seq_len = 1024,
    shorten_factor = (2, 4),     # 2x for first hour glass, 4x for second
    depth = (4, (2, 1, 2), 3),   # 4@1 -> 2@2 -> 1@4 -> 2@2 -> 3@1
)

x = torch.randint(0, 256, (1, 1024))
logits = model(x)
```

For images, instead of average pool and repeat for the down and upsampling functions, they found that linear projections worked a lot better. You can use this by setting `updown_sample_type = 'linear'`

```python
import torch
from hourglass_transformer_pytorch import HourglassTransformer

model = HourglassTransformerLM(
    dim = 512,
    shorten_factor = 2,
    depth = (4, 2, 4),
    updown_sample_type = 'linear'
)

img_tokens = torch.randn(1, 1024, 512)
model(img_tokens) # (1, 1024, 512)
```
## Enwik8 autoregressive example

```bash
$ python train.py
```

## Todo

- [ ] work with non-autoregressive, accounting for masking
- [ ] account for shift padding when naive downsampling

## Citations

```bibtex
@misc{nawrot2021hierarchical,
    title   = {Hierarchical Transformers Are More Efficient Language Models}, 
    author  = {Piotr Nawrot and Szymon Tworkowski and Michał Tyrolski and Łukasz Kaiser and Yuhuai Wu and Christian Szegedy and Henryk Michalewski},
    year    = {2021},
    eprint  = {2110.13711},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
