import argparse

from open_flamingo.eval.models.mistral_model import EvalModel
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)


def main():
    model_args = {
        "config_yaml": "configs/mlm_multi_source_v1_zephyr_ift_zero2.yaml",
        "checkpoint_path": "cruise_logs/zephyr_freeze_ift/mp_rank_00_model_states.pt",
        "precision": "bf16",
    }
    eval_model = EvalModel(model_args)

    tokenizer = eval_model.tokenizer
    # tokenizer.save_pretrained('hf_weights')


if __name__ == "__main__":
    main()
