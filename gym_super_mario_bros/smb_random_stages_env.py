"""An OpenAI Gym Super Mario Bros. environment that randomly selects levels."""
from typing import Any
from typing import Dict
from typing import Union
from typing import SupportsFloat

import numpy as np
import gymnasium as gym

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv


class SuperMarioBrosRandomStagesEnv(gym.Env):
    """A Super Mario Bros. environment that randomly selects levels."""

    # relevant meta-data about the environment
    metadata = SuperMarioBrosEnv.metadata

    # the legal range of rewards for each step
    reward_range = SuperMarioBrosEnv.reward_range

    # observation space for the environment is static across all instances
    observation_space = SuperMarioBrosEnv.observation_space

    # action space is a bitmap of button press values for the 8 NES buttons
    action_space = SuperMarioBrosEnv.action_space

    def __init__(self, rom_mode='vanilla', stages=None):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            stages (list): select stages at random from a specific subset

        Returns:
            None

        """
        # create a dedicated random number generator for the environment
        self.np_random = np.random.RandomState()
        # setup the environments
        self.envs = []
        # iterate over the worlds in the game, i.e., {1, ..., 8}
        for world in range(1, 9):
            # append a new list to put this world's stages into
            self.envs.append([])
            # iterate over the stages in the world, i.e., {1, ..., 4}
            for stage in range(1, 5):
                # create the target as a tuple of the world and stage
                target = (world, stage)
                # create the environment with the given ROM mode
                env = SuperMarioBrosEnv(rom_mode=rom_mode, target=target)
                # add the environment to the stage list for this world
                self.envs[-1].append(env)
        # create a placeholder for the current environment
        self.env = self.envs[0][0]
        # create a placeholder for the image viewer to render the screen
        self.viewer = None
        # create a placeholder for the subset of stages to choose
        self.stages = stages

    @property
    def screen(self):
        """Return the screen from the underlying environment"""
        return self.env.screen

    def seed(self, seed: int | None = None) -> list[int]:
        """
        Set the seed for this environment's random number generator.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.

        """
        # if there is no seed, return an empty list
        if seed is None:
            return []
        # set the random number seed for the NumPy random number generator
        self.np_random.seed(seed)
        # return the list of seeds used by RNG(s) in the environment
        return [seed]


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v7.5.0

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`) and
                the read-only attribute `np_random_seed`.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset
                and the env's :attr:`np_random_seed` will *not* be altered.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # Seed the RNG for this environment.
        self.seed(seed)
        
        # Get the collection of stages to sample from
        stages = self.stages
        if options is not None and 'stages' in options:
            stages = options['stages']
        
        # Select a random level
        if stages is not None and len(stages) > 0:
            level = self.np_random.choice(stages)
            world, stage = level.split('-')
            world = int(world) - 1
            stage = int(stage) - 1
        else:
            world = self.np_random.randint(1, 9) - 1
            stage = self.np_random.randint(1, 5) - 1
        
        # Set the environment based on the world and stage.
        self.env = self.envs[world][stage]

        # reset the environment
        return self.env.reset(
            seed=seed,
            options=options,
        )


    def step(
        self, 
        action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Run one frame of the NES and return the relevant observation data.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        .. versionchanged:: 7.5.0

            The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
            to users when the environment had terminated or truncated which is critical for reinforcement learning
            bootstrapping algorithms.

        Args:
            action (int): an action provided by the agent to update the environment state.

        Returns:
            observation (np.ndarray): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barto Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.

        """
        return self.env.step(action)

    def close(self):
        """Close the environment."""
        # make sure the environment hasn't already been closed
        if self.env is None:
            raise ValueError('env has already been closed.')
        # iterate over each list of stages
        for stage_lists in self.envs:
            # iterate over each stage
            for stage in stage_lists:
                # close the environment
                stage.close()
        # close the environment permanently
        self.env = None
        # if there is an image viewer open, delete it
        if self.viewer is not None:
            self.viewer.close()

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): the mode to render with:
            - human: render to the current display
            - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
              representing RGB values for an x-by-y pixel image

        Returns:
            a numpy array if mode is 'rgb_array', None otherwise

        """
        return SuperMarioBrosEnv.render(self, mode=mode)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        return self.env.get_keys_to_action()

    def get_action_meanings(self):
        """Return the list of strings describing the action space actions."""
        return self.env.get_action_meanings()
    
    def load_state(self, snapshot):
        """Load the state of the environment from a snapshot."""
        self.env.load_state(snapshot)

    def dump_state(self):
        """Save the state of the environment to a snapshot."""
        return self.env.dump_state()
    
    @property
    def _viewer(self):
        """Return the snapshot of the environment."""
        return self.viewer
    
    @_viewer.setter
    def _viewer(self, viewer):
        """Set the viewer."""
        self.viewer = viewer
    
    @property
    def _emulator(self):
        """Return the emulator."""
        return self.env._emulator
    
    @property
    def _screen_buffer(self):
        """Return the screen buffer."""
        return self.env._screen_buffer

    @property
    def _memory_buffer(self):
        """Return the memory buffer."""
        return self.env._memory_buffer
    
# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosRandomStagesEnv.__name__]
