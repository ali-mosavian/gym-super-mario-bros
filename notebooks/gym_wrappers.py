import time
from hashlib import md5
from typing import Union
from typing import Literal
from typing import Generic
from typing import Callable
from typing import NamedTuple
from typing import SupportsFloat
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
import lz4.block as lz4
from gymnasium.core import ActType
from gymnasium.core import ObsType
from gymnasium.core import WrapperActType
from gymnasium.core import WrapperObsType


@dataclass(init=False)
class CompressedFrame:
    """Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.

    Note:
        This object should only be converted to numpy array just before forward pass.
    """
    shape: tuple[int, ...]
    dtype: np.dtype

    _hash: int
    _frame: Union[np.ndarray, bytes]
    _decoder: Union[Callable[[bytes], np.ndarray], Callable[[np.ndarray], np.ndarray]]

    def __init__(self, frame: np.ndarray, compression: Union[None, Literal['fast', 'high']] = None):
        """CompressedFrame for a set of frames and if to apply lz4.

        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally

        Raises:
            DependencyNotInstalled: lz4 is not installed
        """        
        
        self.shape = frame.shape
        self.dtype = frame.dtype   

        self._frame = frame
        self._decoder = lambda x: x
        self._hash = int.from_bytes(md5(frame.tobytes()).digest(), 'little')
        
        if compression is None:
            return
        
        match compression:
            case 'fast':
                mode = 'fast'
            case 'high':
                mode = 'high_compression'
            case _:
                raise ValueError(f"Invalid compression mode: {compression}")
        
        self._frame = lz4.compress(frame.tobytes(), mode=mode)
        self._decoder = lambda x: (
            np.frombuffer(
                lz4.decompress(x),                 
                dtype=frame.dtype
            )
            .reshape(frame.shape)
        )

    def __hash__(self):
        return self._hash
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"CompressedFrame(shape={self.shape}, hash={self._hash:x}, dtype={self.dtype})"
        

    def __array__(self, dtype=None):
        """Gets a numpy array of stacked frames with specific dtype.

        Args:
            dtype: The dtype of the stacked frames

        Returns:
            The array of stacked frames with dtype
        """
        return self._decoder(self._frame)

    def __len__(self):
        """Returns the number of frame stacks.

        Returns:
            The number of frame stacks
        """
        return self.shape[0]

    def __eq__(self, other: 'CompressedFrame'):
        """Checks that the current frames are equal to the other object."""
        return self._frame == other._frame
    
    
@dataclass(init=False)
class SkipFrame(gym.Wrapper[WrapperObsType, WrapperActType, ObsType, ActType]):
    """A wrapper that skips frames by repeating actions and accumulating rewards.
    
    This wrapper helps reduce the frequency of agent decisions by repeating the same action
    for multiple frames and accumulating the rewards. It can optionally render the skipped
    frames at a controlled playback speed.
    
    Attributes:
        _skip (int): Number of frames to skip between agent decisions
        render_frames (bool): Whether to render the intermediate skipped frames
        next_render_time (float | None): Timestamp for next frame render
        playback_speed (float): Speed multiplier for frame rendering
    """
    _skip: int
    render_frames: bool
    next_render_time: float | None
    playback_speed: float
    
    def __init__(
        self, 
        env: gym.Env,
        skip: int,
        render_frames: bool = False,
        playback_speed: float = 1.0
    ) -> None:
        """Initialize the SkipFrame wrapper.
        
        Args:
            env: The Gymnasium environment to wrap
            skip: Number of frames to skip between agent decisions
            render_frames: Whether to render the intermediate skipped frames
            playback_speed: Speed multiplier for frame rendering (>1 is faster)
        """
        super().__init__(env)
        self._skip = skip
        self.render_frames = render_frames
        self.next_render_time: float | None = None
        self.playback_speed = playback_speed

    def step(
        self, 
        action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict]:
        """Execute the same action for multiple frames and accumulate rewards.
        
        The action is repeated for `skip` frames or until the episode ends. Rewards
        are summed across the skipped frames. If render_frames is True, intermediate
        frames are rendered at the specified playback speed.
        
        Args:
            action: The action to repeat for multiple frames
            
        Returns:
            tuple containing:
                np.ndarray: The final observation after skipping frames
                float: The accumulated reward across skipped frames
                bool: Whether the episode terminated
                bool: Whether the episode was truncated
                dict: Additional information from the environment
        """
        done = False
        total_reward = 0.0

        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            s_, r, done, truncated, info = self.env.step(action)
            total_reward += r
            if done or truncated:
                break

            if not self.render_frames:
                continue
            
            # Render the frame at the specified playback speed
            self.next_render_time = self.next_render_time or time.time()
            while time.time() < self.next_render_time:
                time.sleep(1/self.metadata['render_fps']/2)
            
            self.next_render_time += 1/self.metadata['render_fps']/self.playback_speed
            self.env.render()

        return s_, total_reward, done, truncated, info


class Observation(NamedTuple, Generic[WrapperActType]):
    state: CompressedFrame
    action: WrapperActType    
    next_state: CompressedFrame
    reward: SupportsFloat
    terminated: bool
    truncated: bool
    info: dict


@dataclass(init=False)
class ObservationRecorder(gym.Wrapper[np.ndarray, WrapperActType, np.ndarray, WrapperActType]):
    """A wrapper that records observations from a Gymnasium environment.
    
    This wrapper captures and stores the state transitions, actions, rewards, and other
    information from each step of the environment. States are stored as CompressedFrames
    to optimize memory usage.
    
    Attributes:
        action_space (gym.spaces.Discrete): The discrete action space of the environment
        observations (list[Observation]): List of recorded observations during interaction
    """    
    action_space: gym.spaces.Discrete
    observations: list[Observation[WrapperActType]]

    def __init__(self, env: gym.Env) -> None:
        """Initialize the ObservationRecorder wrapper.
        
        Args:
            env (gym.Env): The Gymnasium environment to wrap
        """
        super().__init__(env)
        self.observations = list()
        self.action_space = env.action_space

    def step(self, action: WrapperActType) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Execute one environment step and record the observation.
        
        Records the current state, action, next state, reward and other information
        from the environment step in the observations list.
        
        Args:
            action (int): The action to take in the environment
            
        Returns:
            tuple containing:
                np.ndarray: The next state observation
                float: The reward received
                bool: Whether the episode terminated
                bool: Whether the episode was truncated
                dict: Additional information from the environment
        """
        state = CompressedFrame(self.env.unwrapped.screen.copy(), compression='high')        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state_compressed = CompressedFrame(next_state.copy(), compression='high')        
        
        self.observations.append(
            Observation(
                state=state,
                action=action, 
                next_state=next_state_compressed, 
                reward=reward, 
                terminated=terminated, 
                truncated=truncated, 
                info=info
            )
        )
        
        return next_state, reward, terminated, truncated, info