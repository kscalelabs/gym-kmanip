import os
from typing import Any, Dict
from numpy.typing import NDArray
import gym_kmanip as k

class LogBase:
    def __init__(self, log_dir: str, log_type: str):
        assert os.path.exists(log_dir), f"Directory {log_dir} does not exist"
        self.log_dir = log_dir
        self.log_type = log_type

    def reset(self, info: Dict[str, Any]):
        raise NotImplementedError
    
    def reset_cam(self, cam: k.Cam):
        raise NotImplementedError

    def step(self,
            action: Dict[str, NDArray],
            observation: Dict[str, NDArray],
            info: Dict[str, Any],
            ):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError