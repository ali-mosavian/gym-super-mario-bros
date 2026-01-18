from typing import Set
from typing import List
from typing import Generic
from typing import Optional
from typing import Generator
from typing import SupportsFloat
from dataclasses import field
from dataclasses import dataclass

from gymnasium.core import WrapperActType

from gym_wrappers import Observation
from gym_wrappers import CompressedFrame



@dataclass(frozen=True)
class Transition(Generic[WrapperActType]):
    """Represents a single transition in a trajectory.
    
    A transition consists of an action taken in a state, the reward received,
    and the resulting next state.
    
    Args:
        action: The action taken in the state
        reward: The reward received for taking the action
        next_state: The resulting state after taking the action
    """
    action: WrapperActType    
    reward: SupportsFloat
    next_state: 'TrajectoryNode'

    def __hash__(self) -> int:
        """Calculate hash value for the transition.
        
        Returns:
            int: Hash value combining next_state, action and reward hashes
        """
        return (
            hash(self.next_state) 
            + hash(self.action) 
            + hash(self.reward)
        )

    def __eq__(self, other: object) -> bool:
        """Check equality between two transitions.
        
        Args:
            other: Another object to compare with
            
        Returns:
            bool: True if both transitions have same next_state, action and reward
        """
        if not isinstance(other, Transition):
            return NotImplemented
        return (
            self.next_state == other.next_state and 
            self.action == other.action and 
            self.reward == other.reward
        )


@dataclass(frozen=False)
class TrajectoryNode(Generic[WrapperActType]):
    """Represents a node in a trajectory tree.
    
    Each node contains a state and its possible transitions to other states.
    
    Args:
        terminal: Whether this state is terminal
        state: The compressed frame representing the state
        children: List of possible transitions from this state
    """
    terminal: bool
    state: Optional[CompressedFrame]    
    children: list[Transition[WrapperActType]] = field(default_factory=list)

    def __hash__(self) -> int:
        """Calculate hash value for the node.
        
        Returns:
            int: Hash value of the state
        """
        return hash(self.state)

    @staticmethod
    def from_observations(
        observations: list[Observation[WrapperActType]]
    ) -> 'TrajectoryNode':
        """Create a trajectory tree from a list of observations.
        
        Args:
            observations: List of observations containing state transitions
            
        Returns:
            TrajectoryNode: Root node of the constructed trajectory tree
        """
        trajectories = []
        seen_nodes = dict()

        for obs in observations:
            state_hash = hash(obs.state) 
            if state_hash not in seen_nodes:
                node = TrajectoryNode(state=obs.state, terminal=obs.terminated)
                seen_nodes[state_hash] = node
                trajectories.append(node)

            next_state_hash = hash(obs.next_state)
            if next_state_hash not in seen_nodes:
                seen_nodes[next_state_hash] = TrajectoryNode(state=obs.next_state, terminal=obs.terminated)

            state_node = seen_nodes[state_hash]
            next_state_node = seen_nodes[next_state_hash]

            t = Transition(
                action=obs.action, 
                reward=obs.reward,                 
                next_state=next_state_node
            )
            if t not in set(state_node.children):
                state_node.children.append(t)

        return TrajectoryNode(
            state=None,
            terminal=False,            
            children=trajectories
        )


@dataclass(frozen=True)
class SubTrajectory(Generic[WrapperActType]):
    """Represents a sub-trajectory with its actions, states, and rewards.
    
    Args:
        states: List of compressed frames representing states
        actions: List of actions taken
        rewards: List of rewards received
    """
    states: List[CompressedFrame] = field(default_factory=list)
    actions: List[WrapperActType] = field(default_factory=list)
    rewards: List[SupportsFloat] = field(default_factory=list)

    def append(
        self, 
        state: Optional[CompressedFrame] = None, 
        action: Optional[WrapperActType] = None, 
        reward: Optional[SupportsFloat] = None, 
        max_length: Optional[int] = None
    ) -> 'SubTrajectory':
        """Append new state, action, and reward to the trajectory.
        
        Args:
            state: New state to append
            action: New action to append
            reward: New reward to append
            max_length: Maximum length to maintain (will truncate if exceeded)
            
        Returns:
            SubTrajectory: New trajectory with appended elements
        """
        states = [*self.states, state] if state is not None else [*self.states]
        actions = [*self.actions, action] if action is not None else [*self.actions]
        rewards = [*self.rewards, reward] if reward is not None else [*self.rewards]

        return SubTrajectory(
            states=states[-max_length:],
            actions=actions[-max_length+1:],
            rewards=rewards[-max_length+1:]
        )
    
    def __len__(self) -> int:
        """Get length of trajectory.
        
        Returns:
            int: Number of states in the trajectory
        """
        return len(self.states)

    @classmethod
    def from_trajectory(
        cls, 
        node: 'TrajectoryNode', 
        n: int, 
        path: Optional['SubTrajectory'] = None, 
        visited: Set['TrajectoryNode'] = set()
    ) -> Generator['SubTrajectory', None, None]:
        """Generate all possible sub-trajectories of length n from a trajectory tree.
        
        Args:
            node: Current node in the trajectory tree
            n: Desired length of sub-trajectories
            path: Current path being built
            visited: Set of visited nodes to avoid cycles
            
        Yields:
            SubTrajectory: Each possible sub-trajectory of length n
        """
        if node.state in visited:
            return

        path = path or SubTrajectory()        
        path = path.append(state=node.state, max_length=n)
        if len(path) == n:
            yield path        
        
        for transition in node.children:
            action = transition.action
            reward = transition.reward            
            next_state = transition.next_state
            
            yield from SubTrajectory.from_trajectory(
                next_state, 
                n, 
                path.append(
                    max_length=n,
                    action=action, 
                    reward=reward,                
                ), 
                visited | {node.state}
            )