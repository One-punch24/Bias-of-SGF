# Implicit Bias of SGD 

for Diagonal Linear Networks: a Provable Benefit of stochasticity

Reimplementations and some further experiments of https://arxiv.org/pdf/2106.09524.pdf 

### The effect of L0 Norm of beta

**Convergence**

1. uniformly gaussian, $\gamma=0.025$, $||\beta||_0=20$

![Fig_1](SupResults/UniL0Norm/Fig_1.jpg)

![Fig_2](SupResults/UniL0Norm/Fig_2.jpg)

![Fig_3](SupResults/UniL0Norm/Fig_3.jpg)

![Fig_4](SupResults/UniL0Norm/Fig_4.jpg)

**Comparison**

2. Comparison of SGD, SDE, Accurate SDE,  $\gamma=0.025$, $||\beta||_0=20$, uniformly, $scale=1$ (10 times average)

![Fig_12_repeat_10](SupResults/UniL0Norm_Comp/Fig_12_repeat_10.jpg)

![Fig_13_repeat_10](SupResults/UniL0Norm_Comp/Fig_13_repeat_10.jpg)

3. Comparison of SGD, SDE, Accurate SDE,  $\gamma=0.0007$, $||\beta||_0=20$, nonuniformly, $start=0.1$, $end=10$ (10times average)

![Fig_12_repeat_10](SupResults/NonUniL0Norm/Fig_12_repeat_10.jpg)

![Fig_13_repeat_10](SupResults/NonUniL0Norm/Fig_13_repeat_10.jpg)

### Some Commands 

```python
## Change the Scale of data

python Simu_av_compare.py --distri uniformly --gamma 0.1 --uni_scale 1 --multi 10

python Simu_av_compare.py --distri nonuniformly --gamma 0.0025 --non_uni_start 0.1 --non_uni_scale 10 --multi 10
```

```python 
## Change the L0 Norm of beta

python Simu_av_compare.py --distri uniformly --gamma 0.08 --uni_scale 1 --multi 10 --L0norm 20

python Simu_av_compare.py --distri nonuniformly --gamma 0.00005 --non_uni_start 0.1 --non_uni_scale 10 --multi 10 --L0norm 20
```



