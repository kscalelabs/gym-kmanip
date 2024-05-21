from pathlib import Path

from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub


# https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py
# https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/push_dataset_to_hub.py

# TODO: lerobot has no pip install, have to install from repo directly

push_dataset_to_hub(
    data_dir = ,
    dataset_id = ,
    raw_format = "aloha_hdf5",
    community_id = "kscalelabs",
    revision = ,
    dry_run = ,
    save_to_disk = ,
    tests_data_dir = ,
    save_tests_to_disk = ,
    fps = ,
    video = ,
    batch_size = ,
    num_workers = ,
    debug = ,
)