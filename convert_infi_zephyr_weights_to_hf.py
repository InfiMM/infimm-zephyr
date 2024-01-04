import torch

state_dict = torch.load(
    "cruise_logs/zephyr_freeze_ift/mp_rank_00_model_states.pt", map_location="cpu"
)
state_dict = {k.replace("module.", ""): v for k, v in state_dict["module"].items()}
