# Do-PFN  🔨🔍

Attached is the code supporting "Do-PFN: In-Context Learning for Causal Effect Estimation". Do-PFN is a pre-trained transformer for causal inference, trained to predict conditional interventional distributions (CIDs) and conditional average treatment effects (CATEs) from observational data alone.

$$CID := p(y | do(t), x)$$
$$CATE:= E[y | do(t=1), x] - E[y | do(t=0), x]$$

### Installation

Create a conda environment with python version 3.10, activate it, and install the requirements in requirements.txt

```
   conda create -n dopfn_env python=3.10
   conda activate dopfn_env
   pip install -r requirements.txt
```

### Running Do-PFN

Run ```inference_example.py``` to play around with Do-PFN on any of our real-world and synthetic benchmark datasets by changing the ```dataset``` variable (runs on "Observed Confounder" by default)

### Sampling data from the prior

Run ```prior_data_example.py``` to sample data from our prior.
