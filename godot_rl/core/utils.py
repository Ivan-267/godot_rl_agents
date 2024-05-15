import importlib
import re

import gymnasium as gym


def lod_to_dol(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}


def dol_to_lod(dol):
    return [dict(zip(dol, t)) for t in zip(*dol.values())]


def convert_macos_path(env_path):
    """
    On MacOs the user is supposed to provide a application.app file to env_path.
    However the actual binary is in application.app/Contents/Macos/application.
    This helper function converts the path to the path of the actual binary.

    Example input: ./Demo.app
    Example output: ./Demo.app/Contents/Macos/Demo
    """

    filenames = re.findall(r"[^\/]+(?=\.)", env_path)
    assert len(filenames) == 1, "An error occured while converting the env path for MacOS."
    return env_path + "/Contents/MacOS/" + filenames[0]


class ActionSpaceProcessor:
    # can convert tuple action dists to a single continuous action distribution
    # eg (Box(a), Box(b)) -> Box(a+b)
    # (Box(a), Discrete(2)) -> Box(a+2)
    # etc
    # does not yet work with discrete dists of n>2
    def __init__(self, action_space: gym.spaces.Tuple, convert) -> None:
        self._original_action_space = action_space
        self._convert = convert

        if convert:
            if isinstance(action_space, gym.spaces.Tuple):
                space: gym.spaces.Discrete = action_space.spaces[0]
                self.converted_action_space = gym.spaces.Discrete(space.n)

    @property
    def action_space(self):
        if not self._convert:
            return self._original_action_space

        return self.converted_action_space

    def to_original_dist(self, action):
        if not self._convert:
            return action

        original_action = []

        for space in self._original_action_space.spaces:
            if isinstance(space, gym.spaces.Discrete):
                original_action.append(action)
            else:
                raise NotImplementedError("This utils.py version supports only a single discrete action with any size.")

        return original_action


def can_import(module_name):
    return not cant_import(module_name)


def cant_import(module_name):
    try:
        importlib.import_module(module_name)
        return False
    except ImportError:
        return True
