
# How to run

Singularity image at Plafrim

`/beegfs/ygusak/singularity_images/hiremate_twremat.sif` containing Gurobi and Twremat Haskell to run with HiRemate based on `hiremate_twremat.def`

Run Singularity shell
`sungularity shell --nv beegfs/ygusak/singularity_images/hiremate_twremat.sif`

Inside the shell
```
cd ~/rockmate-private/rockmate && pip install -e . 
cd ~/rockmate-private/rkgb && pip install -e .

```

# Example of how to use HRockmate

```python
import torch
import models
import hrockmate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, sample = models.get_MLP(device)

rematMod = hrockmate.HRockmate(model,sample,budget=4.5e9)

y = rematMod(*sample)
loss = y.sum()
loss.backward()
```

# Example of how to use H-Partition and rkGB
Note that H-Partition correspond to `Ptools.py` and `Htools.py` files.
The have been included in rkGB.

```python
rkgb_res = hrockmate.rkgb.make_all_graphs(model,sample)
```

Using graphviz, you can render the graphs:

```python
rkgb.print(rkgb_res.P_structure,name="Top level graph of the partitioning")
rkgb.print(rkgb_res.H_cluster,name="Top level forward backward graph")
```