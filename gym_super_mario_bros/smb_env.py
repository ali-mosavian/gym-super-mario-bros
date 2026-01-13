"""An OpenAI Gym environment for Super Mario Bros. and Lost Levels."""
from collections import defaultdict
from collections import deque
from typing import Callable

import numpy as np
from nes_py import NESEnv

from gym_super_mario_bros.roms import rom_path
from gym_super_mario_bros.roms import decode_target


# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: 'fireball', {0:'small', 1: 'tall'})


# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]


# RAM addresses for enemy types on the screen
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]


# enemies whose context indicate that a stage change will occur (opposed to an
# enemy that implies a stage change wont occur -- i.e., a vine)
# Bowser = 0x2D
# Flagpole = 0x31
_STAGE_OVER_ENEMIES = np.array([0x2D, 0x31])


def cast_return_type_to(type_) -> Callable[[Callable], Callable]:
    def wrapper(func) -> Callable:
        def wrapper_inner(self, *args, **kwargs):
            return_value = func(self, *args, **kwargs)
            return type_(return_value)
        return wrapper_inner
    return wrapper


class SuperMarioBrosEnv(NESEnv):
    """An environment for playing Super Mario Bros with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-51, 106)

    # Reward configuration constants
    TIME_PENALTY = -0.1          # Every step costs, no camping!
    PROGRESS_SCALE = 0.5         # Per-pixel bonus for new forward progress
    MOMENTUM_WINDOW = 10         # Frames to average for momentum calculation
    MOMENTUM_THRESHOLD = 1.5     # Min avg speed to start getting momentum bonus
    MOMENTUM_SCALE = 0.3         # Bonus scales: (avg_speed - threshold) * scale
    POWERUP_LOSS_PENALTY = -15.0 # Penalty for losing powerup (tall/fire → small)
    DEATH_PENALTY = -50.0        # Heavy penalty for dying
    FLAG_REWARD = 100.0          # Ultimate goal reward

    def __init__(self, rom_mode='vanilla', lost_levels=False, target=None):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            lost_levels (bool): whether to load the ROM with lost levels.
                - False: load original Super Mario Bros.
                - True: load Super Mario Bros. Lost Levels
            target (tuple): a tuple of the (world, stage) to play as a level

        Returns:
            None

        """
        # decode the ROM path based on mode and lost levels flag
        rom = rom_path(lost_levels, rom_mode)
        # initialize the super object with the ROM path
        super(SuperMarioBrosEnv, self).__init__(rom)
        # set the target world, stage, and area variables
        target = decode_target(target, lost_levels)
        self._target_world, self._target_stage, self._target_area = target
        # setup a variable to keep track of the last frames time
        self._time_last = 0
        # setup a variable to keep track of the last frames x position
        self._x_position_last = 0
        # track the furthest x position ever reached (only reward new progress)
        self._x_position_max = 0
        # track if flag was already rewarded this episode
        self._flag_rewarded = False
        # rolling window of recent speeds for momentum calculation
        self._speed_history: deque[float] = deque(maxlen=self.MOMENTUM_WINDOW)
        # track previous player status for powerup loss detection
        self._status_last: str = "small"
        # reset the emulator
        self.reset()
        # skip the start screen
        self._skip_start_screen()
        # create a backup state to restore from on subsequent calls to reset
        self._backup()

    @property
    def is_single_stage_env(self):
        """Return True if this environment is a stage environment."""
        return self._target_world is not None and self._target_area is not None

    # MARK: Memory access

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.

        Args:
            address (int): the address to read from as a 16 bit integer
            length: the number of sequential bytes to read

        Note:
            this method is specific to Mario where three GUI values are stored
            in independent memory slots to save processing time
            - score has 6 10's places
            - coins has 2 10's places
            - time has 3 10's places

        Returns:
            the integer value of this 10's place representation

        """
        return int(''.join(map(str, self.ram[address:address + length])))

    @property
    @cast_return_type_to(int)
    def _level(self):
        """Return the level of the game."""
        return self.ram[0x075f] * 4 + self.ram[0x075c]

    @property
    @cast_return_type_to(int)
    def _world(self):
        """Return the current world (1 to 8)."""
        return self.ram[0x075f] + 1

    @property
    @cast_return_type_to(int)
    def _stage(self):
        """Return the current stage (1 to 4)."""
        return self.ram[0x075c] + 1

    @property
    @cast_return_type_to(int)
    def _area(self):
        """Return the current area number (1 to 5)."""
        return self.ram[0x0760] + 1

    @property
    @cast_return_type_to(int)
    def _score(self):
        """Return the current player score (0 to 999990)."""
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x07de, 6)

    @property
    @cast_return_type_to(int)
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        return self._read_mem_range(0x07f8, 3)

    @property
    @cast_return_type_to(int)
    def _coins(self):
        """Return the number of coins collected (0 to 99)."""
        # coins are represented as a figure with 2 10's places
        return self._read_mem_range(0x07ed, 2)

    @property
    @cast_return_type_to(int)
    def _life(self):
        """Return the number of remaining lives."""
        return self.ram[0x075a]

    @property
    @cast_return_type_to(int)
    def _x_position(self):
        """Return the current horizontal position."""
        # add the current page 0x6d to the current x
        return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])

    @property
    @cast_return_type_to(int)
    def _left_x_position(self):
        """Return the number of pixels from the left of the screen."""
        # TODO: resolve RuntimeWarning: overflow encountered in ubyte_scalars
        # subtract the left x position 0x071c from the current x 0x86
        # return (self.ram[0x86] - self.ram[0x071c]) % 256
        return np.uint8(int(self.ram[0x86]) - int(self.ram[0x071c])) % 256

    @property
    @cast_return_type_to(int)
    def _y_pixel(self):
        """Return the current vertical position."""
        return self.ram[0x03b8]

    @property
    @cast_return_type_to(int)
    def _y_viewport(self):
        """
        Return the current y viewport.

        Note:
            1 = in visible viewport
            0 = above viewport
            > 1 below viewport (i.e. dead, falling down a hole)
            up to 5 indicates falling into a hole

        """
        return self.ram[0x00b5]

    @property
    @cast_return_type_to(int)
    def _y_position(self):
        """Return the current vertical position."""
        # check if Mario is above the viewport (the score board area)
        if self._y_viewport < 1:
            # y position overflows so we start from 255 and add the offset
            return 255 + (255 - self._y_pixel)
        # invert the y pixel into the distance from the bottom of the screen
        return 255 - self._y_pixel

    @property
    @cast_return_type_to(str)
    def _player_status(self):
        """Return the player status as a string."""
        return _STATUS_MAP[self.ram[0x0756]]

    @property
    @cast_return_type_to(int)
    def _player_state(self):
        """
        Return the current player state.

        Note:
            0x00 : Leftmost of screen
            0x01 : Climbing vine
            0x02 : Entering reversed-L pipe
            0x03 : Going down a pipe
            0x04 : Auto-walk
            0x05 : Auto-walk
            0x06 : Dead
            0x07 : Entering area
            0x08 : Normal
            0x09 : Cannot move
            0x0B : Dying
            0x0C : Palette cycling, can't move

        """
        return self.ram[0x000e]

    @property
    @cast_return_type_to(bool)
    def _is_dying(self):
        """Return True if Mario is in dying animation, False otherwise."""
        return self._player_state == 0x0b or self._y_viewport > 1

    @property
    @cast_return_type_to(bool)
    def _is_dead(self):
        """Return True if Mario is dead, False otherwise."""
        return self._player_state == 0x06

    @property
    @cast_return_type_to(bool)
    def _is_game_over(self):
        """Return True if the game has ended, False otherwise."""
        # the life counter will get set to 255 (0xff) when there are no lives
        # left. It goes 2, 1, 0 for the 3 lives of the game
        return self._life == 0xff

    @property
    @cast_return_type_to(bool)
    def _is_busy(self):
        """Return boolean whether Mario is busy with in-game garbage."""
        return self._player_state in _BUSY_STATES

    @property
    @cast_return_type_to(bool)
    def _is_world_over(self):
        """Return a boolean determining if the world is over."""
        # 0x0770 contains GamePlay mode:
        # 0 => Demo
        # 1 => Standard
        # 2 => End of world
        return self.ram[0x0770] == 2

    @property
    @cast_return_type_to(bool)
    def _is_stage_over(self):
        """Return a boolean determining if the level is over."""
        # iterate over the memory addresses that hold enemy types
        for address in _ENEMY_TYPE_ADDRESSES:
            # check if the byte is either Bowser (0x2D) or a flag (0x31)
            # this is to prevent returning true when Mario is using a vine
            # which will set the byte at 0x001D to 3
            if self.ram[address] in _STAGE_OVER_ENEMIES:
                # player float state set to 3 when sliding down flag pole
                return self.ram[0x001D] == 3

        return False

    @property
    @cast_return_type_to(bool)
    def _flag_get(self):
        """Return a boolean determining if the agent reached a flag."""
        return self._is_world_over or self._is_stage_over

    # MARK: RAM Hacks

    def _write_stage(self):
        """Write the stage data to RAM to overwrite loading the next stage."""
        self.ram[0x075f] = self._target_world - 1
        self.ram[0x075c] = self._target_stage - 1
        self.ram[0x0760] = self._target_area - 1

    def _runout_prelevel_timer(self):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self.ram[0x07A0] = 0

    def _skip_change_area(self):
        """Skip change area animations by by running down timers."""
        change_area_timer = self.ram[0x06DE]
        if change_area_timer > 1 and change_area_timer < 255:
            self.ram[0x06DE] = 1

    def _skip_occupied_states(self):
        """Skip occupied states by running out a timer and skipping frames."""
        while self._is_busy or self._is_world_over:
            self._runout_prelevel_timer()
            self.frame_advance(0)

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self.frame_advance(8)
        self.frame_advance(0)
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            self.frame_advance(8)
            # if we're in the single stage, environment, write the stage data
            if self.is_single_stage_env:
                self._write_stage()
            self.frame_advance(0)
            # run-out the prelevel timer to skip the animation
            self._runout_prelevel_timer()
        # set the last time to now
        self._time_last = self._time
        # after the start screen idle to skip some extra frames
        while self._time >= self._time_last:
            self._time_last = self._time
            self.frame_advance(8)
            self.frame_advance(0)

    def _skip_end_of_world(self):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over:
            # get the current game time to reference
            time = self._time
            # loop until the time is different
            while self._time == time:
                # frame advance with NOP
                self.frame_advance(0)

    def _kill_mario(self):
        """Skip a death animation by forcing Mario to death."""
        # force Mario's state to dead
        self.ram[0x000e] = 0x06
        # step forward one frame
        self.frame_advance(0)

    # MARK: Reward Function

    def _calculate_movement_rewards(self):
        """
        Calculate movement-based rewards (progress, momentum).

        Returns a tuple of (progress_reward, momentum_reward).
        Updates internal state for position and speed history tracking.

        Only rewards NEW forward progress beyond the furthest point ever reached.
        Going back and forth does NOT give rewards.
        """
        current_x = self._x_position
        delta = current_x - self._x_position_last
        self._x_position_last = current_x

        # resolve an issue where after death the x position resets
        # the x delta is typically has at most magnitude of 3, 5 is a safe bound
        if delta < -5 or delta > 5:
            self._speed_history.clear()
            return 0.0, 0.0

        # Track speed for momentum calculation (even when not making new progress)
        current_speed = max(0.0, float(delta))
        self._speed_history.append(current_speed)

        # Only reward progress BEYOND the furthest point ever reached
        new_progress = current_x - self._x_position_max
        if new_progress > 0:
            self._x_position_max = current_x
            # Per-pixel bonus for new forward progress
            progress_reward = new_progress * self.PROGRESS_SCALE
        else:
            progress_reward = 0.0

        # Momentum bonus: scales with avg speed above threshold
        # Higher sustained speed = bigger bonus
        if len(self._speed_history) >= self.MOMENTUM_WINDOW:
            avg_speed = sum(self._speed_history) / len(self._speed_history)
            if avg_speed >= self.MOMENTUM_THRESHOLD:
                # Scales linearly: at avg 1.5 -> 0, at avg 3.0 -> 0.45
                momentum_reward = (avg_speed - self.MOMENTUM_THRESHOLD) * self.MOMENTUM_SCALE
            else:
                momentum_reward = 0.0
        else:
            momentum_reward = 0.0

        return progress_reward, momentum_reward

    @property
    def _time_penalty(self):
        """Return the constant time penalty for each step."""
        return self.TIME_PENALTY

    @property
    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._is_dying or self._is_dead:
            return self.DEATH_PENALTY
        return 0.0

    @property
    def _powerup_loss_penalty(self):
        """Return penalty for losing powerup (tall/fireball -> small)."""
        current_status = self._player_status
        previous_status = self._status_last
        self._status_last = current_status

        # Penalize going from powered up to small
        if previous_status in ("tall", "fireball") and current_status == "small":
            return self.POWERUP_LOSS_PENALTY
        return 0.0

    @property
    def _flag_reward(self):
        """Return +FLAG_REWARD when the flag is reached (once per episode)."""
        if self._flag_get and not self._flag_rewarded:
            self._flag_rewarded = True
            return self.FLAG_REWARD
        return 0.0

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._x_position_last = 0
        self._x_position_max = 0
        self._flag_rewarded = False
        self._speed_history.clear()
        self._status_last = "small"

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._x_position_last = self._x_position
        self._x_position_max = self._x_position
        self._flag_rewarded = False
        self._speed_history.clear()
        self._status_last = self._player_status

    def _did_restore(self):
        self._done = self._get_done()
        self._time_last = self._time
        self._x_position_last = self._x_position
        self._x_position_max = self._x_position
        self._flag_rewarded = False
        self._speed_history.clear()
        self._status_last = self._player_status        

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return
        # if mario is dying, then cut to the chase and kill hi,
        if self._is_dying:
            self._kill_mario()
        # skip world change scenes (must call before other skip methods)
        if not self.is_single_stage_env:
            self._skip_end_of_world()
        # skip area change (i.e. enter pipe, flag get, etc.)
        self._skip_change_area()
        # skip occupied states like the black screen between lives that shows
        # how many lives the player has left
        self._skip_occupied_states()

    def _get_reward(self):
        """
        Return the reward after a step occurs.

        Reward components:
        - time_penalty:   -0.1        Every step costs, no camping!
        - progress:       +0.5/pixel  Bonus for new forward progress
        - momentum:       scales      Bonus for sustained speed (avg >= 1.5 over 10 frames)
        - powerup_loss:   -15.0       Penalty for losing powerup (tall/fire → small)
        - death:          -50.0       Heavy penalty
        - flag:           +100.0      Ultimate goal
        """
        progress, momentum = self._calculate_movement_rewards()
        return (
            self._time_penalty
            + progress
            + momentum
            + self._powerup_loss_penalty
            + self._death_penalty
            + self._flag_reward
        )

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        if self.is_single_stage_env:
            return self._is_dying or self._is_dead or self._flag_get
        return self._is_game_over

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            coins=self._coins,
            flag_get=self._flag_get,
            life=self._life,
            score=self._score,
            stage=self._stage,
            status=self._player_status,
            time=self._time,
            world=self._world,
            x_pos=self._x_position,
            y_pos=self._y_position,
        )


# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosEnv.__name__]
