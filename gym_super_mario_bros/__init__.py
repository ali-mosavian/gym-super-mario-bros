"""Registration code of Gym environments in this package."""
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.smb_random_stages_env import SuperMarioBrosRandomStagesEnv
from gym_super_mario_bros._registration import make


# define the outward facing API of this package
__all__ = [
    make.__name__,
    SuperMarioBrosEnv.__name__,
    SuperMarioBrosRandomStagesEnv.__name__,
]
