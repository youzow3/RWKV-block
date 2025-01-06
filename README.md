# [WIP] RWKV: Receptive Key Weight Value

> WIP: This is meant to be reference block implmentaiton for various RWKV modules.

## Pytorch design decisions

- RWKV states are passed around with native `tuples`, and `list` intentionally, benchmarking show this has a measurable speed bump compared to data state classes when needed.
- init state tune weights should be made avaliable via `init_state.x.wkv`
