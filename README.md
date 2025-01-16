# [WIP] RWKV: Receptive Key Weight Value

> WIP: This is meant to be reference block implmentaiton for various RWKV modules.
> It is not considered complete

## Pytorch design decisions

- RWKV states are passed around with native `tuples`, and `list` intentionally, benchmarking show this has a measurable speed bump compared to data state classes when needed.
- init state tune weights should be made avaliable via `init_state.x.wkv`

## Conda specific setup

```bash
conda create -n py-3-12 python=3.12 pip nvidia
conda activate py-3-12

# Install cuda in conda env
conda install cuda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install FLA and other required packages
pip3 install -r requirements.txt

# Optional test requirements
pip3 install -r test/requirements.txt
```