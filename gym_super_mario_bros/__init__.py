"""Registration code of Gym environments in this package."""
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.smb_game import LevelMode
from gym_super_mario_bros.smb_game import MarioGame
from gym_super_mario_bros.smb_random_stages_env import SuperMarioBrosRandomStagesEnv
from gym_super_mario_bros.smb_vec_env import VectorSuperMarioBrosEnv
from gym_super_mario_bros._registration import make


# define the outward facing API of this package
__all__ = [
    make.__name__,
    LevelMode.__name__,
    MarioGame.__name__,
    SuperMarioBrosEnv.__name__,
    SuperMarioBrosRandomStagesEnv.__name__,
    VectorSuperMarioBrosEnv.__name__,
]
