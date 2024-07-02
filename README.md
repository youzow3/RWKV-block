# RWKV: Receptive Key Weight Value

> This is a pure pytorch implementation of key RWKV blocks



## Pytorch design decisions

- RWKV states are passed around with native `tuples`, and `list` intentionally, benchmarking show this has a measurable speed bump compared to data state classes when needed.
- init state tune weights should be made avaliable via `init_state.x.wkv`
- `forward_with_default_compile` is optimized for inference with no state initialization forward passes.