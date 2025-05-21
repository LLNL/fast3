import torch
from models import en_diffusion, EGNN_dynamics_QM9
from utils import config_util
from datasets import pdbbind


configs = config_util.load_configs("configs/egcnn_2020.yaml")
dynamics = EGNN_dynamics_QM9.EGNN_dynamics_QM9(in_node_nf=3, context_node_nf=2, n_dims=2)
model = en_diffusion.EnVariationalDiffusion(
    dynamics=dynamics,
    in_node_nf=3,
    n_dims=3
)

dataset = pdbbind.PDBBind2020(
    mode="pdbbind2020",
    subset="train",
    configs=configs["data"]
)

print(model.forward(x= dataset[0]['data'].x, h=dataset[0]['data'].x, node_mask=dataset[0]['data'].x))

from pdb import set_trace

set_trace()