
## Behavioral cloning results

| Environment Name | Input Size | Output Size | Hidden Sizes | Test Loss | Expert Rewards (mean/std) | Clone Rewards (mean/std) | DAgger Rewards (mean/std) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ant-v2 | 111 | 8 | [128,64,32] | 0.00117 | 4749.274/119.876 | 4659.185/109.162 | - |
| HalfCheetah-v2 | 17 | 6 | [128,64,32] | 0.00645 | 4162.080/62.693 | 3949.776/132.038 | - |
| Hopper-v2 | 11 | 3 | [128,64,32] | 0.00290 | 3777.327/4.046 | 2044.425/626.603 | 3777.615/3.178 |
| Humanoid-v2 | 376 | 17 | [256,128,64,32] | 0.0232 | 10177.319/2242.988 |  1595.178/1543.723 | 10677.324/66.859 |
| Reacher-v2 | 11 | 2 | [128,64,32] | 0.00194 | -3.926/1.588 | -10.542/2.839 | -3.637/1.594 |
| Walker2d-v2 | 17 | 6 | [128,64,32] | 0.0190 | 5520.523/55.601 | 286.927/45.992 | 5544.737/32.589 |