"""Mario game logic - shared between single and vector environments.

This module contains the pure game logic for Super Mario Bros, operating
on RAM buffers. It handles:
- RAM reading (positions, scores, states)
- State queries (is_dying, is_dead, flag_get)
- Reward computation (x-progress, time penalty, death penalty)
- RAM hacks (skip screens, kill mario, write stage)
- Info dict generation

This follows SRP by separating game logic from environment concerns.
"""
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# Status register value to string mapping
_STATUS_MAP = defaultdict(lambda: "fireball", {0: "small", 1: "tall"})

# Player states indicating Mario is "busy" (can't be controlled)
_BUSY_STATES = frozenset([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07])

# RAM addresses for enemy types on screen
_ENEMY_TYPE_ADDRESSES = (0x0016, 0x0017, 0x0018, 0x0019, 0x001A)

# Enemies indicating stage completion (Bowser=0x2D, Flagpole=0x31)
_STAGE_OVER_ENEMIES = frozenset([0x2D, 0x31])

# RAM addresses
class RAM:
    """RAM address constants for Super Mario Bros."""
    WORLD = 0x075F
    STAGE = 0x075C
    AREA = 0x0760
    SCORE_START = 0x07DE
    TIME_START = 0x07F8
    COINS_START = 0x07ED
    LIFE = 0x075A
    X_PAGE = 0x6D
    X_POS = 0x86
    Y_PIXEL = 0x03B8
    Y_VIEWPORT = 0x00B5
    PLAYER_STATUS = 0x0756
    PLAYER_STATE = 0x000E
    GAMEPLAY_MODE = 0x0770
    FLOAT_STATE = 0x001D
    PRELEVEL_TIMER = 0x07A0
    CHANGE_AREA_TIMER = 0x06DE


# =============================================================================
# Level definitions
# =============================================================================

class LevelMode(Enum):
    """Level selection mode for vector environments."""
    SINGLE = "single"      # Play one specific level
    SEQUENTIAL = "sequential"  # Play levels in order
    RANDOM = "random"      # Random level selection


# All levels in Super Mario Bros (world, stage, area)
SMB_LEVELS: List[Tuple[int, int, int]] = [
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),
    (2, 1, 1), (2, 2, 1), (2, 3, 1), (2, 4, 1),
    (3, 1, 1), (3, 2, 1), (3, 3, 1), (3, 4, 1),
    (4, 1, 1), (4, 2, 1), (4, 3, 1), (4, 4, 1),
    (5, 1, 1), (5, 2, 1), (5, 3, 1), (5, 4, 1),
    (6, 1, 1), (6, 2, 1), (6, 3, 1), (6, 4, 1),
    (7, 1, 1), (7, 2, 1), (7, 3, 1), (7, 4, 1),
    (8, 1, 1), (8, 2, 1), (8, 3, 1), (8, 4, 1),
]

# Lost Levels (SMB2 Japan) levels
SMB_LOST_LEVELS: List[Tuple[int, int, int]] = [
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),
    (2, 1, 1), (2, 2, 1), (2, 3, 1), (2, 4, 1),
    (3, 1, 1), (3, 2, 1), (3, 3, 1), (3, 4, 1),
    (4, 1, 1), (4, 2, 1), (4, 3, 1), (4, 4, 1),
    (5, 1, 1), (5, 2, 1), (5, 3, 1), (5, 4, 1),
    (6, 1, 1), (6, 2, 1), (6, 3, 1), (6, 4, 1),
    (7, 1, 1), (7, 2, 1), (7, 3, 1), (7, 4, 1),
    (8, 1, 1), (8, 2, 1), (8, 3, 1), (8, 4, 1),
    (9, 1, 1), (9, 2, 1), (9, 3, 1), (9, 4, 1),
]


# =============================================================================
# MarioGame - Core game logic
# =============================================================================

@dataclass
class MarioGame:
    """
    Encapsulates Mario game state reading and manipulation for a single environment.
    
    This class operates on a RAM buffer (numpy array) and provides:
    - Properties for reading game state (positions, scores, etc.)
    - Methods for state queries (is_dying, is_dead, etc.)
    - Methods for reward computation
    - Methods for RAM manipulation (hacks to skip animations, etc.)
    
    Args:
        ram: The RAM buffer (numpy array of 2048 bytes)
        target_world: Target world for single-stage mode (1-8, or None)
        target_stage: Target stage for single-stage mode (1-4, or None)
        target_area: Target area for single-stage mode (1-5, or None)
        death_penalty: Penalty for dying (default -500)
    """
    ram: np.ndarray
    target_world: Optional[int] = None
    target_stage: Optional[int] = None
    target_area: Optional[int] = None
    death_penalty: float = -500.0
    
    # Reward tracking state (mutable)
    _time_last: int = field(default=0, repr=False)
    _x_position_last: int = field(default=0, repr=False)

    # =========================================================================
    # Properties - RAM reading
    # =========================================================================

    @property
    def is_single_stage(self) -> bool:
        """Return True if this is a single-stage environment."""
        return self.target_world is not None and self.target_area is not None

    @property
    def level(self) -> int:
        """Return the level index (0-31)."""
        return int(self.ram[RAM.WORLD] * 4 + self.ram[RAM.STAGE])

    @property
    def world(self) -> int:
        """Return the current world (1-8)."""
        return int(self.ram[RAM.WORLD] + 1)

    @property
    def stage(self) -> int:
        """Return the current stage (1-4)."""
        return int(self.ram[RAM.STAGE] + 1)

    @property
    def area(self) -> int:
        """Return the current area (1-5)."""
        return int(self.ram[RAM.AREA] + 1)

    @property
    def score(self) -> int:
        """Return the current score (0-999990)."""
        return self._read_mem_range(RAM.SCORE_START, 6)

    @property
    def time(self) -> int:
        """Return the time remaining (0-999)."""
        return self._read_mem_range(RAM.TIME_START, 3)

    @property
    def coins(self) -> int:
        """Return the number of coins (0-99)."""
        return self._read_mem_range(RAM.COINS_START, 2)

    @property
    def life(self) -> int:
        """Return the number of lives remaining."""
        return int(self.ram[RAM.LIFE])

    @property
    def x_position(self) -> int:
        """Return the horizontal position in the level."""
        return int(self.ram[RAM.X_PAGE]) * 0x100 + int(self.ram[RAM.X_POS])

    @property
    def y_pixel(self) -> int:
        """Return the vertical pixel position."""
        return int(self.ram[RAM.Y_PIXEL])

    @property
    def y_viewport(self) -> int:
        """Return the y viewport (1=visible, 0=above, >1=below/falling)."""
        return int(self.ram[RAM.Y_VIEWPORT])

    @property
    def y_position(self) -> int:
        """Return the vertical position (distance from bottom)."""
        if self.y_viewport < 1:
            return 255 + (255 - self.y_pixel)
        return 255 - self.y_pixel

    @property
    def player_status(self) -> str:
        """Return the player status as a string ('small', 'tall', 'fireball')."""
        return _STATUS_MAP[self.ram[RAM.PLAYER_STATUS]]

    @property
    def player_state(self) -> int:
        """Return the player state byte."""
        return int(self.ram[RAM.PLAYER_STATE])

    # =========================================================================
    # State queries
    # =========================================================================

    @property
    def is_dying(self) -> bool:
        """Return True if Mario is in dying animation."""
        return self.player_state == 0x0B or self.y_viewport > 1

    @property
    def is_dead(self) -> bool:
        """Return True if Mario is dead."""
        return self.player_state == 0x06

    @property
    def is_game_over(self) -> bool:
        """Return True if game over (no lives left)."""
        return self.life == 0xFF

    @property
    def is_busy(self) -> bool:
        """Return True if Mario is busy (can't be controlled)."""
        return self.player_state in _BUSY_STATES

    @property
    def is_world_over(self) -> bool:
        """Return True if world is complete."""
        return self.ram[RAM.GAMEPLAY_MODE] == 2

    @property
    def is_stage_over(self) -> bool:
        """Return True if stage is complete (flag reached)."""
        for address in _ENEMY_TYPE_ADDRESSES:
            if self.ram[address] in _STAGE_OVER_ENEMIES:
                return self.ram[RAM.FLOAT_STATE] == 3
        return False

    @property
    def flag_get(self) -> bool:
        """Return True if Mario reached the flag/goal."""
        return self.is_world_over or self.is_stage_over

    @property
    def done(self) -> bool:
        """Return True if episode is over."""
        if self.is_single_stage:
            return self.is_dying or self.is_dead or self.flag_get
        return self.is_game_over

    # =========================================================================
    # Reward computation
    # =========================================================================

    def reset_reward_state(self):
        """Reset the reward tracking state (call on episode reset)."""
        self._time_last = self.time
        self._x_position_last = self.x_position

    def compute_reward(self) -> float:
        """Compute and return the total reward for this step."""
        return self._x_reward() + self._time_penalty() + self._death_penalty()

    def _x_reward(self) -> float:
        """Return reward based on horizontal movement."""
        x_pos = self.x_position
        reward = x_pos - self._x_position_last
        self._x_position_last = x_pos
        # Filter large jumps (death resets position)
        if reward < -5 or reward > 5:
            return 0
        return reward

    def _time_penalty(self) -> float:
        """Return penalty for time passing."""
        time = self.time
        reward = time - self._time_last
        self._time_last = time
        # Time only decreases; positive means reset occurred
        if reward > 0:
            return 0
        return reward

    def _death_penalty(self) -> float:
        """Return penalty for dying."""
        if self.is_dying or self.is_dead:
            return self.death_penalty
        return 0

    # =========================================================================
    # RAM hacks
    # =========================================================================

    def write_stage(self):
        """Write target stage to RAM (for single-stage mode)."""
        if self.target_world is not None:
            self.ram[RAM.WORLD] = self.target_world - 1
        if self.target_stage is not None:
            self.ram[RAM.STAGE] = self.target_stage - 1
        if self.target_area is not None:
            self.ram[RAM.AREA] = self.target_area - 1

    def set_level(self, world: int, stage: int, area: int = 1):
        """Set the current level directly."""
        self.ram[RAM.WORLD] = world - 1
        self.ram[RAM.STAGE] = stage - 1
        self.ram[RAM.AREA] = area - 1

    def runout_prelevel_timer(self):
        """Force pre-level timer to 0 to skip waiting."""
        self.ram[RAM.PRELEVEL_TIMER] = 0

    def skip_change_area(self):
        """Skip area change animations."""
        timer = self.ram[RAM.CHANGE_AREA_TIMER]
        if 1 < timer < 255:
            self.ram[RAM.CHANGE_AREA_TIMER] = 1

    def kill_mario(self):
        """Force Mario to dead state (skip death animation)."""
        self.ram[RAM.PLAYER_STATE] = 0x06

    # =========================================================================
    # Info generation
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Return info dict for gymnasium compatibility."""
        return {
            "coins": self.coins,
            "flag_get": self.flag_get,
            "life": self.life,
            "score": self.score,
            "stage": self.stage,
            "status": self.player_status,
            "time": self.time,
            "world": self.world,
            "x_pos": self.x_position,
            "y_pos": self.y_position,
        }

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _read_mem_range(self, address: int, length: int) -> int:
        """Read bytes as decimal digits (e.g., score stored as 6 separate bytes)."""
        return int("".join(map(str, self.ram[address:address + length])))


# =============================================================================
# Level selection helpers
# =============================================================================

def get_level_list(lost_levels: bool = False) -> List[Tuple[int, int, int]]:
    """Get the list of levels for the given game."""
    return SMB_LOST_LEVELS if lost_levels else SMB_LEVELS


def get_random_level(
    rng: np.random.Generator,
    lost_levels: bool = False,
) -> Tuple[int, int, int]:
    """Get a random level."""
    levels = get_level_list(lost_levels)
    idx = rng.integers(len(levels))
    return levels[idx]


def get_next_level(
    current_world: int,
    current_stage: int,
    lost_levels: bool = False,
) -> Tuple[int, int, int]:
    """Get the next level in sequence (wraps around)."""
    levels = get_level_list(lost_levels)
    # Find current level index
    for i, (w, s, a) in enumerate(levels):
        if w == current_world and s == current_stage:
            next_idx = (i + 1) % len(levels)
            return levels[next_idx]
    # Default to first level if not found
    return levels[0]


# Exports
__all__ = [
    "MarioGame",
    "LevelMode",
    "RAM",
    "SMB_LEVELS",
    "SMB_LOST_LEVELS",
    "get_level_list",
    "get_random_level",
    "get_next_level",
]
