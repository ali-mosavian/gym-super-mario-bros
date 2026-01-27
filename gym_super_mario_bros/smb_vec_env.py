"""A vectorized Super Mario Bros environment using C++ parallel emulation."""
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.vector.utils import batch_space

from nes_py.emulator import VectorEmulator

from gym_super_mario_bros.roms import rom_path
from gym_super_mario_bros.roms import decode_target


# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: "fireball", {0: "small", 1: "tall"})

# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]

# RAM addresses for enemy types on the screen
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]

# enemies whose context indicate that a stage change will occur
_STAGE_OVER_ENEMIES = np.array([0x2D, 0x31])


class VectorSuperMarioBrosEnv:
    """
    A vectorized Super Mario Bros environment using C++ parallel emulation.
    
    This environment steps multiple Mario games in parallel using the
    VectorEmulator, which uses C++ threads for true parallelism. It provides
    a gymnasium VectorEnv-like interface.
    
    Args:
        num_envs: Number of parallel environments
        rom_mode: ROM mode ('vanilla', 'downsample', 'pixel', 'rectangle')
        lost_levels: Whether to use Lost Levels ROM
        target: Optional (world, stage) tuple for single-stage environments
    
    Example:
        >>> env = VectorSuperMarioBrosEnv(num_envs=8)
        >>> obs, info = env.reset()
        >>> obs, rewards, terminated, truncated, info = env.step(actions)
    """

    # the legal range of rewards for each step
    reward_range = (-500, 15)

    def __init__(
        self,
        num_envs: int,
        rom_mode: str = "vanilla",
        lost_levels: bool = False,
        target: Optional[Tuple[int, int]] = None,
    ):
        self._num_envs = num_envs
        self._rom_mode = rom_mode
        self._lost_levels = lost_levels

        # decode the ROM path based on mode and lost levels flag
        self._rom_path = rom_path(lost_levels, rom_mode)

        # set the target world, stage, and area variables
        target_decoded = decode_target(target, lost_levels)
        self._target_world, self._target_stage, self._target_area = target_decoded

        # create the vectorized emulator
        self._emulator = VectorEmulator(self._rom_path, num_envs)

        # per-environment state tracking
        self._time_last = np.zeros(num_envs, dtype=np.int32)
        self._x_position_last = np.zeros(num_envs, dtype=np.int32)
        self._done = np.ones(num_envs, dtype=bool)  # Start as done until reset
        self._snapshots: List[Optional[np.ndarray]] = [None] * num_envs

        # gymnasium spaces
        self.single_observation_space = Box(
            low=0,
            high=255,
            shape=(self._emulator.height, self._emulator.width, 3),
            dtype=np.uint8,
        )
        self.single_action_space = Discrete(256)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        # initialize all environments
        self._initialize_all()

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return self._num_envs

    @property
    def is_single_stage_env(self) -> bool:
        """Return True if this environment is a stage environment."""
        return self._target_world is not None and self._target_area is not None

    # =========================================================================
    # Memory access helpers (vectorized)
    # =========================================================================

    def _get_ram(self, idx: int) -> np.ndarray:
        """Get RAM buffer for environment idx."""
        return self._emulator.memory_buffer(idx)

    def _read_mem_range(self, idx: int, address: int, length: int) -> int:
        """Read a range of bytes where each byte is a 10's place figure."""
        ram = self._get_ram(idx)
        return int("".join(map(str, ram[address : address + length])))

    def _get_level(self, idx: int) -> int:
        ram = self._get_ram(idx)
        return int(ram[0x075F] * 4 + ram[0x075C])

    def _get_world(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x075F] + 1)

    def _get_stage(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x075C] + 1)

    def _get_area(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x0760] + 1)

    def _get_score(self, idx: int) -> int:
        return self._read_mem_range(idx, 0x07DE, 6)

    def _get_time(self, idx: int) -> int:
        return self._read_mem_range(idx, 0x07F8, 3)

    def _get_coins(self, idx: int) -> int:
        return self._read_mem_range(idx, 0x07ED, 2)

    def _get_life(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x075A])

    def _get_x_position(self, idx: int) -> int:
        ram = self._get_ram(idx)
        return int(ram[0x6D]) * 0x100 + int(ram[0x86])

    def _get_y_pixel(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x03B8])

    def _get_y_viewport(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x00B5])

    def _get_y_position(self, idx: int) -> int:
        y_viewport = self._get_y_viewport(idx)
        y_pixel = self._get_y_pixel(idx)
        if y_viewport < 1:
            return 255 + (255 - y_pixel)
        return 255 - y_pixel

    def _get_player_status(self, idx: int) -> str:
        return _STATUS_MAP[self._get_ram(idx)[0x0756]]

    def _get_player_state(self, idx: int) -> int:
        return int(self._get_ram(idx)[0x000E])

    def _is_dying(self, idx: int) -> bool:
        return self._get_player_state(idx) == 0x0B or self._get_y_viewport(idx) > 1

    def _is_dead(self, idx: int) -> bool:
        return self._get_player_state(idx) == 0x06

    def _is_game_over(self, idx: int) -> bool:
        return self._get_life(idx) == 0xFF

    def _is_busy(self, idx: int) -> bool:
        return self._get_player_state(idx) in _BUSY_STATES

    def _is_world_over(self, idx: int) -> bool:
        return self._get_ram(idx)[0x0770] == 2

    def _is_stage_over(self, idx: int) -> bool:
        ram = self._get_ram(idx)
        for address in _ENEMY_TYPE_ADDRESSES:
            if ram[address] in _STAGE_OVER_ENEMIES:
                return ram[0x001D] == 3
        return False

    def _flag_get(self, idx: int) -> bool:
        return self._is_world_over(idx) or self._is_stage_over(idx)

    # =========================================================================
    # RAM Hacks
    # =========================================================================

    def _write_stage(self, idx: int):
        """Write the stage data to RAM to overwrite loading the next stage."""
        ram = self._get_ram(idx)
        ram[0x075F] = self._target_world - 1
        ram[0x075C] = self._target_stage - 1
        ram[0x0760] = self._target_area - 1

    def _runout_prelevel_timer(self, idx: int):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self._get_ram(idx)[0x07A0] = 0

    def _skip_change_area(self, idx: int):
        """Skip change area animations by running down timers."""
        ram = self._get_ram(idx)
        change_area_timer = ram[0x06DE]
        if 1 < change_area_timer < 255:
            ram[0x06DE] = 1

    def _skip_occupied_states(self, idx: int):
        """Skip occupied states by running out a timer and skipping frames."""
        while self._is_busy(idx) or self._is_world_over(idx):
            self._runout_prelevel_timer(idx)
            self._frame_advance_single(idx, 0)

    def _skip_start_screen(self, idx: int):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self._frame_advance_single(idx, 8)
        self._frame_advance_single(idx, 0)
        # Press start until the game starts
        while self._get_time(idx) == 0:
            self._frame_advance_single(idx, 8)
            if self.is_single_stage_env:
                self._write_stage(idx)
            self._frame_advance_single(idx, 0)
            self._runout_prelevel_timer(idx)
        # set the last time to now
        self._time_last[idx] = self._get_time(idx)
        # after the start screen idle to skip some extra frames
        while self._get_time(idx) >= self._time_last[idx]:
            self._time_last[idx] = self._get_time(idx)
            self._frame_advance_single(idx, 8)
            self._frame_advance_single(idx, 0)

    def _skip_end_of_world(self, idx: int):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over(idx):
            time = self._get_time(idx)
            while self._get_time(idx) == time:
                self._frame_advance_single(idx, 0)

    def _kill_mario(self, idx: int):
        """Skip a death animation by forcing Mario to death."""
        self._get_ram(idx)[0x000E] = 0x06
        self._frame_advance_single(idx, 0)

    # =========================================================================
    # Frame advance helpers
    # =========================================================================

    def _frame_advance_single(self, idx: int, action: int):
        """Advance a single frame for one environment."""
        actions = np.zeros(self._num_envs, dtype=np.uint8)
        actions[idx] = action
        self._emulator.step(actions)

    # =========================================================================
    # Reward functions
    # =========================================================================

    def _get_x_reward(self, idx: int) -> float:
        """Return the reward based on left right movement between steps."""
        x_pos = self._get_x_position(idx)
        reward = x_pos - self._x_position_last[idx]
        self._x_position_last[idx] = x_pos
        # Resolve issue where after death the x position resets
        if reward < -5 or reward > 5:
            return 0
        return reward

    def _get_time_penalty(self, idx: int) -> float:
        """Return the reward for the in-game clock ticking."""
        time = self._get_time(idx)
        reward = time - self._time_last[idx]
        self._time_last[idx] = time
        # Time can only decrease, positive reward results from reset
        if reward > 0:
            return 0
        return reward

    def _get_death_penalty(self, idx: int) -> float:
        """Return the reward earned by dying."""
        if self._is_dying(idx) or self._is_dead(idx):
            return -500
        return 0

    def _get_reward(self, idx: int) -> float:
        """Return the total reward for this step."""
        return self._get_x_reward(idx) + self._get_time_penalty(idx) + self._get_death_penalty(idx)

    def _get_done(self, idx: int) -> bool:
        """Return True if the episode is over, False otherwise."""
        if self.is_single_stage_env:
            return self._is_dying(idx) or self._is_dead(idx) or self._flag_get(idx)
        return self._is_game_over(idx)

    def _get_info(self, idx: int) -> Dict[str, Any]:
        """Return the info for this environment."""
        return dict(
            coins=self._get_coins(idx),
            flag_get=self._flag_get(idx),
            life=self._get_life(idx),
            score=self._get_score(idx),
            stage=self._get_stage(idx),
            status=self._get_player_status(idx),
            time=self._get_time(idx),
            world=self._get_world(idx),
            x_pos=self._get_x_position(idx),
            y_pos=self._get_y_position(idx),
        )

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_single(self, idx: int):
        """Initialize a single environment (skip start screen, create backup)."""
        self._emulator.reset_env(idx)
        self._skip_start_screen(idx)
        # Create backup snapshot for this environment
        # Note: VectorEmulator doesn't have dump_state per-env, so we skip this
        # The backup will be handled differently

    def _initialize_all(self):
        """Initialize all environments."""
        self._emulator.reset()
        for idx in range(self._num_envs):
            self._skip_start_screen(idx)
            self._time_last[idx] = self._get_time(idx)
            self._x_position_last[idx] = self._get_x_position(idx)

    # =========================================================================
    # Gymnasium VectorEnv interface
    # =========================================================================

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments and return initial observations.

        Returns:
            observations: Stacked observations from all environments
            infos: Dictionary with per-environment info
        """
        # Reset all emulators
        self._emulator.reset()

        # Initialize each environment
        for idx in range(self._num_envs):
            self._time_last[idx] = 0
            self._x_position_last[idx] = 0
            self._skip_start_screen(idx)
            self._time_last[idx] = self._get_time(idx)
            self._x_position_last[idx] = self._get_x_position(idx)
            self._done[idx] = False

        # Get observations
        observations = np.stack(self._emulator.screen_buffer())

        # Collect info
        infos = self._collect_infos()

        return observations, infos

    def reset_single(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset a single environment."""
        self._emulator.reset_env(idx)
        self._time_last[idx] = 0
        self._x_position_last[idx] = 0
        self._skip_start_screen(idx)
        self._time_last[idx] = self._get_time(idx)
        self._x_position_last[idx] = self._get_x_position(idx)
        self._done[idx] = False

        return self._emulator.screen_buffer(idx), self._get_info(idx)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step all environments with the given actions.

        Args:
            actions: Array of actions, one per environment

        Returns:
            observations: Stacked observations
            rewards: Array of rewards
            terminated: Array of termination flags
            truncated: Array of truncation flags (always False)
            infos: Dictionary with per-environment info
        """
        actions = np.asarray(actions, dtype=np.uint8)

        # Step all emulators in parallel (C++ threads)
        self._emulator.step(actions)

        # Collect results
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        terminated = np.zeros(self._num_envs, dtype=bool)

        for idx in range(self._num_envs):
            # Get reward
            rewards[idx] = self._get_reward(idx)
            rewards[idx] = np.clip(rewards[idx], self.reward_range[0], self.reward_range[1])

            # Get done flag
            self._done[idx] = self._get_done(idx)
            terminated[idx] = self._done[idx]

            # Handle post-step logic
            if not self._done[idx]:
                if self._is_dying(idx):
                    self._kill_mario(idx)
                if not self.is_single_stage_env:
                    self._skip_end_of_world(idx)
                self._skip_change_area(idx)
                self._skip_occupied_states(idx)

        # Get observations
        observations = np.stack(self._emulator.screen_buffer())

        # Collect info
        infos = self._collect_infos()

        return observations, rewards, terminated, np.zeros(self._num_envs, dtype=bool), infos

    def _collect_infos(self) -> Dict[str, Any]:
        """Collect info from all environments into a batched dict."""
        # For gymnasium compatibility, return dict with arrays
        infos: Dict[str, Any] = {
            "coins": np.array([self._get_coins(i) for i in range(self._num_envs)]),
            "flag_get": np.array([self._flag_get(i) for i in range(self._num_envs)]),
            "life": np.array([self._get_life(i) for i in range(self._num_envs)]),
            "score": np.array([self._get_score(i) for i in range(self._num_envs)]),
            "stage": np.array([self._get_stage(i) for i in range(self._num_envs)]),
            "status": [self._get_player_status(i) for i in range(self._num_envs)],
            "time": np.array([self._get_time(i) for i in range(self._num_envs)]),
            "world": np.array([self._get_world(i) for i in range(self._num_envs)]),
            "x_pos": np.array([self._get_x_position(i) for i in range(self._num_envs)]),
            "y_pos": np.array([self._get_y_position(i) for i in range(self._num_envs)]),
        }
        return infos

    def close(self):
        """Close the environment."""
        if self._emulator is not None:
            del self._emulator
            self._emulator = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# explicitly define the outward facing API of this module
__all__ = [VectorSuperMarioBrosEnv.__name__]
