"""A vectorized Super Mario Bros environment using C++ parallel emulation."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gymnasium.vector.utils import batch_space

from nes_py.emulator import VectorEmulator

from gym_super_mario_bros.roms import decode_target
from gym_super_mario_bros.roms import rom_path
from gym_super_mario_bros.smb_game import LevelMode
from gym_super_mario_bros.smb_game import MarioGame
from gym_super_mario_bros.smb_game import get_level_list
from gym_super_mario_bros.smb_game import get_next_level
from gym_super_mario_bros.smb_game import get_random_level
from gym_super_mario_bros.smb_game import warm_up_level
from gym_super_mario_bros.smb_game import warm_up_levels


class VectorSuperMarioBrosEnv:
    """
    A vectorized Super Mario Bros environment using C++ parallel emulation.
    
    This environment steps multiple Mario games in parallel using the
    VectorEmulator, which uses C++ threads for true parallelism. It provides
    a gymnasium VectorEnv-like interface with auto-reset support.
    
    Args:
        num_envs: Number of parallel environments
        rom_mode: ROM mode ('vanilla', 'downsample', 'pixel', 'rectangle')
        lost_levels: Whether to use Lost Levels ROM
        level_mode: Level selection mode:
            - LevelMode.SINGLE: Play one specific level (requires target)
            - LevelMode.SEQUENTIAL: Play levels in order, advancing on completion
            - LevelMode.RANDOM: Random level on each reset
        target: (world, stage) tuple for SINGLE mode, or starting level for SEQUENTIAL
        auto_reset: If True, automatically reset terminated environments
    
    Example:
        >>> # Single level mode
        >>> env = VectorSuperMarioBrosEnv(num_envs=8, level_mode=LevelMode.SINGLE, target=(1, 1))
        
        >>> # Random levels mode  
        >>> env = VectorSuperMarioBrosEnv(num_envs=8, level_mode=LevelMode.RANDOM)
        
        >>> # Sequential levels mode
        >>> env = VectorSuperMarioBrosEnv(num_envs=8, level_mode=LevelMode.SEQUENTIAL)
        
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
        level_mode: LevelMode = LevelMode.SINGLE,
        target: Optional[Tuple[int, int]] = None,
        auto_reset: bool = True,
    ):
        self._num_envs = num_envs
        self._rom_mode = rom_mode
        self._lost_levels = lost_levels
        self._level_mode = level_mode
        self._auto_reset = auto_reset

        # Decode ROM path
        self._rom_path = rom_path(lost_levels, rom_mode)

        # Handle target based on level mode
        if level_mode == LevelMode.SINGLE:
            if target is None:
                target = (1, 1)  # Default to 1-1
            target_decoded = decode_target(target, lost_levels)
            self._default_level = target_decoded
        else:
            # For RANDOM/SEQUENTIAL, target is optional starting point
            if target is not None:
                target_decoded = decode_target(target, lost_levels)
                self._default_level = target_decoded
            else:
                self._default_level = (1, 1, 1)

        # Level snapshot cache: (world, stage, area) -> snapshot bytes
        self._level_snapshots: Dict[Tuple[int, int, int], np.ndarray] = {}

        # IMPORTANT: Pre-warm level snapshots BEFORE creating the main emulator.
        # Creating many temp VectorEmulators after the main emulator causes thread
        # synchronization issues that make load_state hang. By creating snapshots
        # first, we avoid this problem.
        self._prewarm_snapshots()

        # Create the vectorized emulator (after prewarming snapshots)
        self._emulator = VectorEmulator(self._rom_path, num_envs)

        # Random generator for level selection
        self._rng = np.random.default_rng()

        # Create MarioGame instances for each environment
        self._games: List[MarioGame] = []
        for idx in range(num_envs):
            ram = self._emulator.memory_buffer(idx)
            game = MarioGame(ram=ram)
            self._games.append(game)

        # Per-environment current level tracking
        self._current_levels: List[Tuple[int, int, int]] = [self._default_level] * num_envs

        # Done tracking
        self._done = np.ones(num_envs, dtype=bool)

        # Gymnasium spaces
        self.single_observation_space = Box(
            low=0,
            high=255,
            shape=(self._emulator.height, self._emulator.width, 3),
            dtype=np.uint8,
        )
        self.single_action_space = Discrete(256)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

        # Initialize all environments
        self._initialize_all()

    @property
    def num_envs(self) -> int:
        """Return the number of environments."""
        return self._num_envs

    @property
    def level_mode(self) -> LevelMode:
        """Return the level selection mode."""
        return self._level_mode

    # =========================================================================
    # Level snapshot management
    # =========================================================================

    def _prewarm_snapshots(self):
        """Pre-create snapshots for levels based on mode.
        
        Uses parallel snapshot creation when multiple levels are needed.
        """
        if self._level_mode == LevelMode.SINGLE:
            levels_to_create = [self._default_level]
        elif self._level_mode == LevelMode.SEQUENTIAL:
            levels_to_create = [self._default_level]
        else:  # RANDOM - pre-create all levels
            levels_to_create = get_level_list(self._lost_levels)

        if len(levels_to_create) == 1:
            # Single level - use simple sequential method
            self._create_snapshot_for_level_fresh(levels_to_create[0])
        else:
            # Multiple levels - create all in parallel
            self._create_snapshots_parallel(levels_to_create)

    def _create_snapshots_parallel(self, levels: List[Tuple[int, int, int]]):
        """Create snapshots for multiple levels in parallel using VectorEmulator.
        
        This is much faster than creating them sequentially because all
        emulators step in parallel using C++ threads.
        """
        import gc
        
        num_levels = len(levels)
        
        # Create a VectorEmulator with one env per level
        temp_vec = VectorEmulator(self._rom_path, num_levels)
        
        # Create MarioGame instances for RAM access
        games: List[MarioGame] = []
        for idx in range(num_levels):
            temp_vec.reset_env(idx)
            ram = temp_vec.memory_buffer(idx)
            games.append(MarioGame(ram=ram))
        
        # Warm up all levels in parallel
        warm_up_levels(games, levels, temp_vec.step)
        
        # Capture all snapshots
        for idx in range(num_levels):
            snapshot = temp_vec.dump_state(idx).copy()
            self._level_snapshots[levels[idx]] = snapshot
        
        # Cleanup
        del games
        del temp_vec
        gc.collect()

    def _create_snapshot_for_level_fresh(self, level: Tuple[int, int, int]) -> np.ndarray:
        """Create a snapshot for a single level using a fresh temporary emulator.
        
        Used for single-level modes or on-demand snapshot creation.
        """
        import gc
        
        if level in self._level_snapshots:
            return self._level_snapshots[level]
        
        # Create fresh emulator for this level
        temp_vec = VectorEmulator(self._rom_path, 1)
        temp_vec.reset_env(0)
        temp_ram = temp_vec.memory_buffer(0)
        temp_game = MarioGame(ram=temp_ram)
        
        # Warm up to the target level
        warm_up_level(temp_game, level, lambda a: temp_vec.step_single(0, a))
        
        # Save snapshot and copy the state data so it outlives temp_vec
        snapshot = temp_vec.dump_state(0).copy()
        self._level_snapshots[level] = snapshot
        
        # Clean up temp emulator - ensure proper cleanup
        del temp_ram
        del temp_game
        del temp_vec
        gc.collect()
        
        return snapshot

    def _get_or_create_snapshot(self, level: Tuple[int, int, int]) -> np.ndarray:
        """Get or create a snapshot for the given level."""
        if level in self._level_snapshots:
            return self._level_snapshots[level]
        return self._create_snapshot_for_level_fresh(level)

    # =========================================================================
    # Level selection
    # =========================================================================

    def _select_level_for_reset(self, idx: int) -> Tuple[int, int, int]:
        """Select level for environment reset based on level mode."""
        if self._level_mode == LevelMode.SINGLE:
            return self._default_level
        elif self._level_mode == LevelMode.RANDOM:
            return get_random_level(self._rng, self._lost_levels)
        else:  # SEQUENTIAL
            return self._current_levels[idx]

    def _advance_level(self, idx: int):
        """Advance to next level (for SEQUENTIAL mode after flag_get)."""
        if self._level_mode == LevelMode.SEQUENTIAL:
            game = self._games[idx]
            self._current_levels[idx] = get_next_level(
                game.world, game.stage, self._lost_levels
            )

    # =========================================================================
    # Frame advance helpers
    # =========================================================================

    def _frame_advance_single(self, idx: int, action: int):
        """Advance a single frame for one environment."""
        actions = np.zeros(self._num_envs, dtype=np.uint8)
        actions[idx] = action
        self._emulator.step(actions)

    # =========================================================================
    # Skip helpers
    # =========================================================================

    def _skip_occupied_states(self, idx: int):
        """Skip occupied states by running out timer and skipping frames."""
        game = self._games[idx]
        while game.is_busy or game.is_world_over:
            game.runout_prelevel_timer()
            self._frame_advance_single(idx, 0)

    def _skip_end_of_world(self, idx: int):
        """Skip the cutscene at end of world."""
        game = self._games[idx]
        if game.is_world_over:
            time = game.time
            while game.time == time:
                self._frame_advance_single(idx, 0)

    def _kill_mario(self, idx: int):
        """Skip death animation by forcing dead state."""
        game = self._games[idx]
        game.kill_mario()
        self._frame_advance_single(idx, 0)

    # =========================================================================
    # Initialization
    # =========================================================================

    def _initialize_single(self, idx: int, level: Tuple[int, int, int]):
        """Initialize a single environment by loading level snapshot."""
        game = self._games[idx]

        # Get or create snapshot for this level
        snapshot = self._get_or_create_snapshot(level)

        # Load snapshot into this environment
        self._emulator.load_state(idx, snapshot)

        # Track current level
        self._current_levels[idx] = level

        # Initialize reward tracking
        game.reset_reward_state()

        # Mark as not done
        self._done[idx] = False

    def _initialize_all(self):
        """Initialize all environments."""
        for idx in range(self._num_envs):
            level = self._select_level_for_reset(idx)
            self._initialize_single(idx, level)

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

        Args:
            seed: Optional random seed
            options: Optional reset options

        Returns:
            observations: Stacked observations from all environments
            infos: Dictionary with per-environment info
        """
        # Set seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Initialize each environment with appropriate level
        for idx in range(self._num_envs):
            level = self._select_level_for_reset(idx)
            self._initialize_single(idx, level)

        # Get observations and info
        observations = np.stack(self._emulator.screen_buffer())
        infos = self._collect_infos()

        return observations, infos

    def reset_single(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset a single environment."""
        level = self._select_level_for_reset(idx)
        self._initialize_single(idx, level)
        return self._emulator.screen_buffer(idx).copy(), self._games[idx].get_info()

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
            game = self._games[idx]

            # Compute reward
            rewards[idx] = game.compute_reward()
            rewards[idx] = np.clip(rewards[idx], self.reward_range[0], self.reward_range[1])

            # Check if done
            self._done[idx] = game.done
            terminated[idx] = self._done[idx]

            # Handle post-step logic for non-terminated environments
            if not self._done[idx]:
                if game.is_dying:
                    self._kill_mario(idx)
                if self._level_mode != LevelMode.SINGLE:
                    self._skip_end_of_world(idx)
                game.skip_change_area()
                self._skip_occupied_states(idx)

        # Collect info before potential auto-reset
        infos = self._collect_infos()

        # Handle auto-reset for terminated environments
        if self._auto_reset:
            for idx in range(self._num_envs):
                if terminated[idx]:
                    game = self._games[idx]

                    # Store final observation in info
                    if "final_observation" not in infos:
                        infos["final_observation"] = [None] * self._num_envs
                    infos["final_observation"][idx] = self._emulator.screen_buffer(idx).copy()

                    # Store final info
                    if "final_info" not in infos:
                        infos["final_info"] = [None] * self._num_envs
                    infos["final_info"][idx] = game.get_info()

                    # Advance level if flag was reached (SEQUENTIAL mode)
                    if game.flag_get:
                        self._advance_level(idx)

                    # Reset the environment
                    level = self._select_level_for_reset(idx)
                    self._initialize_single(idx, level)

        # Get observations (after potential auto-reset)
        observations = np.stack(self._emulator.screen_buffer())

        return observations, rewards, terminated, np.zeros(self._num_envs, dtype=bool), infos

    def _collect_infos(self) -> Dict[str, Any]:
        """Collect info from all environments into a batched dict."""
        infos: Dict[str, Any] = {
            "coins": np.array([g.coins for g in self._games]),
            "flag_get": np.array([g.flag_get for g in self._games]),
            "life": np.array([g.life for g in self._games]),
            "score": np.array([g.score for g in self._games]),
            "stage": np.array([g.stage for g in self._games]),
            "status": [g.player_status for g in self._games],
            "time": np.array([g.time for g in self._games]),
            "world": np.array([g.world for g in self._games]),
            "x_pos": np.array([g.x_position for g in self._games]),
            "y_pos": np.array([g.y_position for g in self._games]),
        }
        return infos

    def close(self):
        """Close the environment."""
        if hasattr(self, "_emulator") and self._emulator is not None:
            del self._emulator
            self._emulator = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


# explicitly define the outward facing API of this module
__all__ = [VectorSuperMarioBrosEnv.__name__]
