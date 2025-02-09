# MMLU tester, with batch mode

This test RWKV models against MMLU, in batch mode.
With blank right padding, to help test the kernel accurately, internally.

This is not meant to be a replacement for the lm-eval-harness, and its more of an internal test tool.