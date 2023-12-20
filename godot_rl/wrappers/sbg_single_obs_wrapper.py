from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


# A variant of the Stable Baselines Godot Env that only supports a single obs space from the dictionary - obs["obs"].
# This provides some basic support for using envs that have a single obs space with policies other than MultiInputPolicy.

class SBGSingleObsEnv(StableBaselinesGodotEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs, rewards, term, info = super().step(action)
        return obs["obs"], rewards, term, info

    def reset(self) -> np.ndarray:
        obs = super().reset()
        return obs["obs"]

    @property
    def observation_space(self) -> gym.Space:
        return self.envs[0].observation_space["obs"]
