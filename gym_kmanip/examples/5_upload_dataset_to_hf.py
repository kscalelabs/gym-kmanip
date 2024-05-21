from pathlib import Path

from lerobot.scripts.push_dataset_to_hub import push_dataset_to_hub

import gym_kmanip as k

# LeRobot is still early days, no pip install, have to install from repo directly:
# pip install git+ssh://git@github.com/huggingface/lerobot.git@e67da1d7a665622c89d32cd2a58e3b4cc5fd6f4a

# You will need to have a HF_TOKEN in your environment

# https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py
# https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/push_dataset_to_hub.py


# this looked very sketchy, likely this has changed
push_dataset_to_hub(
    Path(k.DATA_DIR), # data_dir
    # Fill in your dataset directory here
    # needs to contain "sim" and the directory should be called "_raw" but don't put that here
    "sim_test_gym_kmanip", # dataset_id
    "aloha_hdf5", # raw_format
    "kscalelabs", # community_id
    k.HF_LEROBOT_VERSION, # revision
    False, # dry_run
    False, # save_to_disk
    Path(k.DATA_DIR), # tests_data_dir
    False, # save_tests_to_disk
    k.FPS, # fps
    True, # video
    k.HF_LEROBOT_BATCH_SIZE, # batch_size
    k.HF_LEROBOT_NUM_WORKERS, # num_workers
    False, # debug
)