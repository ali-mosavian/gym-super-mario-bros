import math
import random
import hashlib
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple
from typing import Optional

import numpy as np
import gymnasium as gym


@dataclass
class MCTSNode:
    state_dump: np.ndarray
    state_hash: str
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    visits: int = 0
    reward: float = 0.0
    children: List['MCTSNode'] = field(default_factory=list)
    terminal: bool = False

    def is_fully_expanded(self, action_space_size: int) -> bool:
        '''
        Check if all possible actions have been expanded for this node.
        '''
        return len(self.children) == action_space_size

    def best_child(self, exploration_param: float = 1.41) -> 'MCTSNode':
        '''
        Select the best child node using the UCT formula.
        '''
        return max(
            self.children,
            key=lambda child: uct_value(
                total_visits=self.visits,
                node_wins=child.reward,
                node_visits=child.visits,
                exploration_param=exploration_param,
            ),
        )

    def find_child_with_action(self, action: int) -> Optional['MCTSNode']:
        '''
        Find child node that corresponds to a specific action.
        '''
        for child in self.children:
            if child.action == action:
                return child
        return None


def hash_state(state: np.ndarray) -> str:
    '''
    Generates a hash for the given np.ndarray state.
    '''
    return hashlib.md5(state.tobytes()).hexdigest()


def uct_value(
    total_visits: int,
    node_wins: float,
    node_visits: int,
    exploration_param: float = 1.41
) -> float:
    '''
    Calculates the UCT value for a node.
    '''
    if node_visits == 0:
        return float('inf')
    return (node_wins / node_visits) + exploration_param * math.sqrt(
        math.log(total_visits) / node_visits
    )


class MCTS:
    def __init__(self) -> None:
        self.visited_states: set[str] = set()
        self.root: Optional[MCTSNode] = None

    def is_state_visited(self, image: np.ndarray) -> bool:
        '''
        Checks if a state has been visited by comparing its image hash.
        '''
        return hash_state(image) in self.visited_states

    def select(
        self,
        node: MCTSNode,
        env: gym.Env,
        action_space_size: int
    ) -> MCTSNode:
        '''
        Traverse the tree using UCT until a node is found for expansion.
        During traversal, keep environment state in sync with selected child nodes.
        '''
        snapshot: np.ndarray = env.unwrapped.dump_state()

        while node.children and node.is_fully_expanded(action_space_size) and not node.terminal:
            node = node.best_child()
            env.unwrapped.load_state(node.state_dump)
            if node.action is not None:
                _, _, terminated, truncated, _ = env.step(node.action)
                if terminated or truncated:
                    node.terminal = True
                    break

        env.unwrapped.load_state(snapshot)
        return node

    def expand(
        self,
        node: MCTSNode,
        env: gym.Env,
        action_space_size: int
    ) -> Optional[MCTSNode]:
        '''
        Expand a node by creating a new child for an untried action,
        skipping already-visited states.
        '''
        if node.terminal:
            return None

        untried_actions: List[int] = [
            a for a in range(action_space_size)
            if all(child.action != a for child in node.children)
        ]
        if not untried_actions:
            return None

        action: int = random.choice(untried_actions)
        snapshot: np.ndarray = env.unwrapped.dump_state()
        state: np.ndarray
        reward: float
        terminated: bool
        truncated: bool
        state, reward, terminated, truncated, _ = env.step(action)

        child_terminal: bool = False
        if terminated or truncated:
            child_terminal = True

        state_hash: str = hash_state(state)
        if state_hash in self.visited_states:
            env.unwrapped.load_state(snapshot)
            return None

        self.visited_states.add(state_hash)

        child_node: MCTSNode = MCTSNode(
            state_dump=env.unwrapped.dump_state(),
            state_hash=state_hash,
            parent=node,
            action=action,
            terminal=child_terminal
        )
        node.children.append(child_node)
        env.unwrapped.load_state(snapshot)

        return child_node

    def simulate(
        self,
        env: gym.Env,
        max_trajectory_length: int
    ) -> Tuple[float, List[int]]:
        '''
        Perform a simulation (rollout) to estimate the reward and find good action sequences.
        Uses a random rollout policy by default.
        '''
        actions: List[int] = []
        total_reward: float = 0.0
        snapshot: np.ndarray = env.unwrapped.dump_state()

        for _ in range(max_trajectory_length):
            action: int = int(env.action_space.sample())
            _, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            total_reward += reward
            if terminated or truncated:
                #print(f'Terminal state, reward: {reward}')
                break

        env.unwrapped.load_state(snapshot)
        return total_reward, actions
    
    def simulate2(
        self,
        env: gym.Env,
        max_trajectory_length: int
    ) -> Tuple[float, List[int]]:
        '''
        Perform a simulation (rollout) to estimate the reward and find good action sequences,
        delegating to a recursive helper function for clarity.
        '''
        snapshot: np.ndarray = env.unwrapped.dump_state()

        total_reward, actions = self._simulate_recursive(
            env=env,
            step=0,
            max_steps=max_trajectory_length,
            snapshot=snapshot,
            actions=[],
            total_reward=0.0,
            best_partial_reward=0.0,
            best_actions=[]
        )

        env.unwrapped.load_state(snapshot)
        return total_reward, actions

    def _simulate_recursive(
        self,
        env: gym.Env,
        step: int,
        max_steps: int,
        snapshot: np.ndarray,
        actions: List[int],
        total_reward: float,
        best_partial_reward: float,
        best_actions: List[int]
    ) -> Tuple[float, List[int]]:
        '''
        A recursive helper function for simulation.
        If the agent reaches a terminal state, it backtracks to the best partial path discovered so far.
        '''
        if step >= max_steps:
            return total_reward, actions

        action: int = int(env.action_space.sample())
        _, reward, terminated, truncated, _ = env.step(action)

        updated_actions: List[int] = [*actions, action]
        updated_reward: float = total_reward + reward

        updated_best_partial_reward: float = best_partial_reward
        updated_best_actions: List[int] = [*best_actions]

        if updated_reward > best_partial_reward:            
            updated_best_partial_reward = updated_reward
            updated_best_actions = updated_actions.copy()
            if not (terminated or truncated):
                snapshot = env.unwrapped.dump_state()

        if terminated or truncated:
            env.unwrapped.load_state(snapshot)
            return updated_best_partial_reward, updated_best_actions

        return self._simulate_recursive(
            env=env,
            step=step + 1,
            max_steps=max_steps,
            snapshot=snapshot,
            actions=updated_actions,
            total_reward=updated_reward,
            best_partial_reward=updated_best_partial_reward,
            best_actions=updated_best_actions
        )    

    def backpropagate(
        self,
        node: MCTSNode,
        reward: float
    ) -> None:
        '''
        Backpropagate the reward to all ancestors.
        '''
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def search(
        self,
        env: gym.Env,
        root: MCTSNode,
        num_simulations: int,
        max_trajectory_length: int
    ) -> Tuple[float, List[int]]:
        '''
        Perform MCTS to find the best action sequence.
        '''
        action_space_size: int = env.action_space.n

        best_sequence: List[int] = []
        best_reward: float = float('-inf')
        
        for _ in range(num_simulations):
            node: MCTSNode = root
            snapshot: np.ndarray = env.unwrapped.dump_state()

            for depth in range(max_trajectory_length):
                node = self.select(node, env, action_space_size)
                new_node: Optional[MCTSNode] = self.expand(node, env, action_space_size)
                if not new_node:
                    break

                rollout_reward, roll_actions = self.simulate(env, max_trajectory_length - depth)

                if rollout_reward > best_reward:
                    best_reward = rollout_reward
                    best_sequence = roll_actions

                self.backpropagate(new_node, rollout_reward)



            env.unwrapped.load_state(snapshot)

        #best_sequence: List[int] = []
        #current_node: MCTSNode = root
        #while current_node.children:
        #    current_node = max(current_node.children, key=lambda child: child.reward)
        #    if current_node.action is not None:
        #        best_sequence.append(current_node.action)

        return best_reward, best_sequence

    def update_root_after_action(self, action: int, env: gym.Env) -> None:
        '''
        Update the root node after an action is taken.
        '''
        if self.root is None:
            return

        # Find the child node corresponding to the action
        new_root: Optional[MCTSNode] = self.root.find_child_with_action(action)
        
        if new_root is not None:
            # Detach from parent and update as new root
            new_root.parent = None
            self.root = new_root
        else:
            # If we can't find the child (shouldn't happen in normal use), create new root
            state_hash: str = hash_state(env.unwrapped.screen)
            self.root = MCTSNode(
                state_dump=env.unwrapped.dump_state(),
                state_hash=state_hash
            )

    def do_rollout(
        self,
        env: gym.Env,
        num_simulations: int = 10,
        max_trajectory_length: int = 25
    ) -> Tuple[float, List[int]]:
        '''
        Perform a rollout using MCTS, reusing existing tree if available.
        '''
        snapshot: np.ndarray = env.unwrapped.dump_state()
        
        # Create root node only if we don't have one
        if self.root is None:
            initial_state: np.ndarray = env.unwrapped.screen
            self.root = MCTSNode(
                state_dump=snapshot,
                state_hash=hash_state(initial_state)
            )

        best_reward, best_sequence = self.search(
            env=env,
            root=self.root,
            num_simulations=num_simulations,
            max_trajectory_length=max_trajectory_length
        )
        env.unwrapped.load_state(snapshot)

        return best_reward, best_sequence
    

