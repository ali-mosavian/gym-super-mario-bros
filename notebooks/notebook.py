# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .pyenv
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import time
from typing import Union
from typing import Literal
from typing import Callable
from typing import Optional
from typing import NamedTuple
from dataclasses import field
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from mcts import MCTS


# %%
@dataclass(init=False)
class LazyFrame:
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
        """Lazyframe for a set of frames and if to apply lz4.

        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally

        Raises:
            DependencyNotInstalled: lz4 is not installed
        """        
        from hashlib import md5
        self.shape = frame.shape
        self.dtype = frame.dtype        
        self._frame = frame
        self._hash = int.from_bytes(md5(frame.tobytes()).digest(), 'little')
        self._decoder = lambda x: x
        

        if compression is None:
            return
        
        try:
            import lz4.block as lz4
        except ImportError:
            raise DependencyNotInstalled(
                "lz4 is not installed, run `pip install lz4`"
            )
        
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
        return f"LazyFrame(shape={self.shape}, hash={self._hash:x}, dtype={self.dtype})"
        

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

    def __eq__(self, other):
        """Checks that the current frames are equal to the other object."""
        return self.__array__() == other
    

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip, render_frames: bool = False, playback_speed: float = 1.0):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.render_frames = render_frames
        self.next_render_time = None
        self.playback_speed = playback_speed

    def step(self, action):
        """Repeat action, and sum reward"""
        done = False
        total_reward = 0.0

        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            s_, r, done, truncated, info = self.env.step(action)
            total_reward += r
            if done or truncated:
                break

            if self.render_frames:
                self.next_render_time = self.next_render_time or time.time()
                while time.time() < self.next_render_time:
                    time.sleep(1/self.metadata['render_fps']/2)
                self.next_render_time += 1/self.metadata['render_fps']/self.playback_speed
                self.env.render()

        return s_, total_reward, done, truncated, info


class Observation(NamedTuple):
    state: LazyFrame
    action: int    
    next_state: LazyFrame
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass(init=False)
class ObservationRecorder(gym.Wrapper):
    """Record the observations of the environment"""    
    action_space: gym.spaces.Discrete
    observations: list[Observation] = field(default_factory=list)
    record: bool = True
    

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observations = list()
        self.action_space = env.action_space


    def step(self, action: int):
        state = LazyFrame(self.env.unwrapped.screen.copy(), compression='high')        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state_lf = LazyFrame(next_state.copy(), compression='high')
        
        
        if self.record:            
            self.observations.append(
                Observation(
                    state=state,
                    action=action, 
                    next_state=next_state_lf, 
                    reward=reward, 
                    terminated=terminated, 
                    truncated=truncated, 
                    info=info
                )
            )
            
        return next_state, reward, terminated, truncated, info


# %%
from mcts import MCTS
import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

#env = gym.make('SuperMarioBros-1-1-v0')
env = gym.make('SuperMarioBrosRandomStages-v0')

actions = SIMPLE_MOVEMENT
env = JoypadSpace(env, actions)


class Transition(NamedTuple):
    state: LazyFrame
    action: Optional[int] = None    
    reward: Optional[float] = None
    terminal: Optional[bool] = None


transitions = []

try:
    mcts = MCTS()
    env = SkipFrame(env, skip=8, render_frames=False, playback_speed=1)
    env = ObservationRecorder(env)

    done = True    
    actions = []
    num_games = 0

    last_state = None
    while num_games < 50:
        if done:
            last_state, _ = env.reset()
            transitions.append(Transition(state=LazyFrame(last_state, compression='high')))
            num_games += 1
        #env.record = False

        if len(actions) == 0:
            reward, actions = mcts.do_rollout(env, num_simulations=5, max_trajectory_length=15)
            print(reward, actions)
            actions = actions

        if len(actions) == 0:
            actions = [env.action_space.sample()]

        #env.record = True
        env.render_frames = True
        env.next_render_time = None
        action, *actions = actions
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated   
        env.render_frames = False
        
        transitions.append(Transition(action=action, state=LazyFrame(state, compression='high'), reward=reward, terminal=done))
        last_state = state

except KeyboardInterrupt:
    pass

# %%

from trajectory import TrajectoryNode, SubTrajectory
sub_trajectories =[
    t
    for trajectory in TrajectoryNode.from_observations(env.observations).children
    for t in SubTrajectory.from_trajectory(trajectory, 7)
]
len(sub_trajectories)

# %%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import StackDataset


dataset = StackDataset(
    states=[s.states for s in sub_trajectories],
    actions=[s.actions for s in sub_trajectories],
    rewards=[s.rewards for s in sub_trajectories]
)

dataset_split = random_split(
    dataset, 
    [0.7, 0.15, 0.15], 
    generator=torch.Generator().manual_seed(42)
)

train_ds, test_ds, val_ds = [
    DataLoader(
        ds, 
        batch_size=16, 
        shuffle=True, 
        #num_workers=0,
        #prefetch_factor=2,
        collate_fn=lambda x: (
            np.array(
                [
                    item['states']
                    for item in x
                ]
            ),
            np.array(
                [
                    np.array(item['actions']) 
                    for item in x
                ]
            ),
            np.array(
                [
                    np.array(item['rewards']) 
                    for item in x
                ]
            )
        )
    ) 
    for ds in dataset_split
]



# %%
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from patch import PatchEmbeddingModel
from cnn import CNNVAE, AutoEncoder
from loss import combined_quality_loss, psnr_loss
from patch import PatchAutoEncoder


def contrastive_diversity_loss(z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B, T, D = z.shape
    flat_emb = z.view(B * T, D)
    # Normalize embeddings
    normalized_emb = F.normalize(flat_emb, p=2, dim=1)
    # Compute similarity matrix
    sim_matrix = torch.mm(normalized_emb, normalized_emb.t()) / temperature
    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(sim_matrix.shape[0]).to(sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)


def determinant_diversity_loss(embeddings: torch.Tensor) -> torch.Tensor:
    B, T, D = embeddings.shape
    flat_emb = embeddings.view(B * T, D)
    # Normalize embeddings
    normalized_emb = F.normalize(flat_emb, p=2, dim=1)
    # Compute Gram matrix
    gram = torch.mm(normalized_emb, normalized_emb.t())
    # Minimize negative log determinant (maximizes determinant)
    return -torch.logdet(gram + torch.eye(gram.shape[0]).to(gram.device) * 1e-6)



class NextStatePredictor(BaseModel):    
    def __init__(self, embedding_dim: int, num_actions: int, num_heads: int):
        super().__init__()        
        self.num_actions = num_actions

        self.state_encoder = PatchEmbeddingModel(
            img_size=(240, 256),
            img_channels=3,
            patch_size=16,
            embed_dim=embedding_dim,
            patch_embed_dim=32,
            num_heads=8,
            dropout=0.1,
            reduction='mean'
        )

        self.state_decoder = IJEPADecoder(embedding_dim=embedding_dim)

        # Adapt the state_action_pairs to the MHA layer
        self.mha_adapter = nn.Sequential(
            nn.LazyLinear(embedding_dim*2),
            nn.BatchNorm1d(embedding_dim*2),
            nn.GELU(),            
            nn.Dropout(0.1),
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        

        self.next_emb_predictor = nn.Sequential(
            nn.LazyLinear(embedding_dim*4),
            nn.BatchNorm1d(embedding_dim*4),
            nn.GELU(),            
            nn.Dropout(0.1),
            nn.Linear(embedding_dim*4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Sigmoid(),
        )

        self.reward_predictor = nn.Sequential(
            nn.LazyLinear(embedding_dim*4),
            nn.BatchNorm1d(embedding_dim*4),
            nn.GELU(),            
            nn.Dropout(0.1),
            nn.Linear(embedding_dim*4, 1),            
        )        

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        


    def get_positional_embeddings(self, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional embeddings."""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, seq_len, d_model, device=device)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        return pos_enc


    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor,
        return_loss: bool = False,
        return_rewards: bool = False,
        return_reconstructed: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for the next state predictor.
        
        Args:
            states: Input states (batch, time_steps, height, width, channels)
            actions: Input actions (batch, time_steps)
            return_loss: Whether to return the loss
        
        Returns:
            Predicted next_emb and optionally next_frame, reward and loss (in that order)
        """
        B, T, H, W, C = states.shape

        
        # Compress states to embeddings
        flat_states = states.view(B * T, H, W, C).permute(0, 3, 1, 2)
        embeddings = self.state_encoder(flat_states)
        
        # Reshape embeddings & states
        embeddings = embeddings.view(B, T, -1)
        orig_states = flat_states.view(B, T, C, H, W)
        
        # Convert actions to one-hot encoding
        action_onehot = F.one_hot(actions, num_classes=self.num_actions)
        
        # Prepare sequences
        curr_state_embs = embeddings[:, :-1]
        next_state_embs = embeddings[:, 1:]
        
        # Add action encodings to the current state embeddings
        state_action_pairs = torch.cat([curr_state_embs, action_onehot], dim=-1)

        # Apply adapter to the state-action pairs to prepare for MHA
        B, T, D = state_action_pairs.shape
        x = self.mha_adapter(state_action_pairs.view(B*T, D))
        x = x.view(B, T, -1)
        
        # Add positional embeddings to the adapter output
        B, T, D = x.shape
        pos_emb = self.get_positional_embeddings(T, D, x.device)
        x = x + pos_emb
        x = self.layer_norm1(x)

        # Apply causal self-attention to capture past information in 
        # preparation for predicting the next state
        causal_mask = torch.ones((T, T), device=state_action_pairs.device, dtype=torch.bool)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = x + attn_out
        x = self.layer_norm2(x)

        # Predict the next state embeddings
        B, T, D = x.shape
        next_state_embs_pred = (
            self
            .next_emb_predictor(
                x
                .view(B*T, D)
            )
            .view(B, T, -1)
        )

        # Predict the rewards
        reward_preds = self.reward_predictor(x.view(B*T, D))
        reward_preds = reward_preds.view(B, T)
        
        # Reconstruct the states from the predicted next states
        B, T, H, W, C = states.shape
        state_reconstruction = self.state_decoder(embeddings.view(B*T, -1))
        state_reconstruction = state_reconstruction.view(B, T, C, H, W).contiguous()
        
        # Determine what to return
        return_values = (next_state_embs_pred[:, -1],)
        
        if return_reconstructed:
            return_values = (*return_values, state_reconstruction.permute(0, 1, 3, 4, 2).contiguous())

        if return_rewards:
            return_values = (*return_values, reward_preds[:, -1])

        if not return_loss:
            return return_values
        
        loss = self.loss(
            next_state_embs, 
            next_state_embs_pred, 
            embeddings, 
            orig_states, 
            state_reconstruction,
            rewards,
            reward_preds
        )

        return (*return_values, loss)

    def loss(
        self,         
        true_states: torch.Tensor,
        pred_states: torch.Tensor, 
        embeddings: torch.Tensor,
        orig_states: torch.Tensor,
        reconstructed: torch.Tensor,
        rewards: torch.Tensor,
        pred_rewards: torch.Tensor
    ) -> torch.Tensor:
        """Calculate combined loss with reconstruction, embedding and prediction terms.
        
        Args:
            true_states: True next state embeddings
            pred_states: Predicted next state embeddings
            embeddings: State embeddings
            orig_states: Original input states
            reconstructed: Reconstructed states
            
        Returns:
            Combined loss value
        """
        # Prediction loss
        pred_loss = torch.sqrt(F.mse_loss(pred_states, true_states))
        
        # Force the embeddings to be uncorrelated, this is necessary for the
        # next state predictor to learn the correct dynamics and not collapse
        #B, T, D = embeddings.shape
        #flat_emb = embeddings.view(B * T, D)
        #emb_corr = torch.corrcoef(flat_emb)
       # emb_loss = F.mse_loss(emb_corr, torch.eye(emb_corr.shape[0]).to(emb_corr.device))
        emb_loss = contrastive_diversity_loss(embeddings)

        # Reconstruction quality loss
        B, T, C, H, W = reconstructed.shape
        recon_loss = psnr_loss(
            reconstructed.reshape(B*T, C, H, W), 
            orig_states.reshape(B*T, C, H, W),             
        )
        recon_loss = 1 - torch.clamp(recon_loss, min=0.0, max=50.0)/50.0

        # Reward loss
        reward_loss = torch.sqrt(F.mse_loss(pred_rewards, rewards))
        print(f'pred_loss: {pred_loss.item():.4f}, emb_loss: {emb_loss.item():.4f}, recon_loss: {recon_loss.item():.4f}, reward_loss: {reward_loss.item():.4f}')
       
        return 0.1*recon_loss + pred_loss #+ 0.1*reward_loss


#model = NextStatePredictor(embedding_dim=128*4, num_actions=env.action_space.n, num_heads=8)
model = AutoEncoder(dropout=0.0, decorrelation=True).to(torch.bfloat16)
#model = PatchAutoEncoder()
#model = model.to('cpu')

s,a,r = next(iter(train_ds))
s = torch.tensor(s, dtype=torch.bfloat16)[:16]
a = torch.tensor(a, dtype=torch.long)[:16]
r = torch.tensor(r, dtype=torch.bfloat16)[:16]
#next_emb, next_frame, reward, loss = model(s, a, r, return_loss=True, return_reconstructed=True, return_rewards=True)
with torch.no_grad():
    B, T, H, W, C = s.shape
    x_reconstructed, loss = model(s.reshape(B*T, H, W, C), return_loss=True)
#loss.backward()

s.reshape(B*T, H, W, C).shape, x_reconstructed.shape,  loss
model = torch.compile(model, mode='max-autotune')


#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# %%
print(model.num_parameters())
print(model.encoder.num_parameters())
print(model.decoder.num_parameters())


# %%
model.encoder

# %%
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

def get_lr_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
    max_lr: float = 1e-3
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a combined linear warmup and cosine decay scheduler with safeguards.
    
    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate after decay.
        max_lr: Maximum learning rate at peak of warmup.
    
    Returns:
        A LambdaLR scheduler that increases LR linearly from min_lr to max_lr 
        during warmup, then decays it via a cosine schedule down to min_lr.
    """
    def lr_lambda(current_step: int) -> float:
        # Safeguard to avoid division by zero
        warmup_steps_safe = max(1, warmup_steps)
        total_steps_safe = max(1, total_steps)

        # ----- Linear Warmup -----
        if current_step < warmup_steps_safe:            
            # Linearly scale from min_lr to max_lr
            progress = float(current_step) / float(warmup_steps_safe)
            lr = min_lr + (max_lr - min_lr) * progress
            return lr

        # ----- Cosine Decay -----
        # Progress after warmup in [0, 1]
        progress = float(current_step - warmup_steps_safe) / float(max(1, total_steps_safe - warmup_steps_safe))
        progress = min(1.0, max(0.0, progress))  # ensure it's within [0, 1]
        
        # Cosine decay from max_lr down to min_lr
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (max_lr - min_lr) * cosine_decay
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def available_device() -> torch.device:
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    return torch.device('cuda' if has_cuda else 'mps' if has_mps else 'cpu')


from IPython import display

def train(
    train_ds: DataLoader, 
    model: nn.Module, 
    num_epochs: int = 100, 
    batch_size: int = 32, 
    warmup_steps: int = 10, 
    start_lr: float = 1e-4, 
    min_lr: float = 1e-6,
    dtype: torch.dtype = torch.float,
    val_ds: DataLoader = None
) -> list[float]:
    """Training loop for the model.
    
    Args:
        train_ds: Training dataset
        model: Neural network model
        num_epochs: Number of training epochs
    
    Returns:
        List of losses per epoch
    """
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=start_lr)
    #scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=len(x)*25, min_lr=min_lr, max_lr=start_lr)
    train_losses = []
    val_losses = []

    # Training loop
    losses: list[float] = []
    device = torch.device(available_device())
    model = model.to(device, dtype=dtype)
    
    f, ax = plt.subplots(figsize=(10, 6))
    disp_obj = display.display(f, display_id=True)

    
    epoch_bar = tqdm(range(num_epochs), desc='Epoch', unit='epoch')
    for epoch in epoch_bar:
        epoch_loss = 0.0
        num_batches = 0
        
        with tqdm(train_ds, desc='Training', unit='batch', leave=False, disable=True) as batch_bar:
            for batch in batch_bar:                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass            
                s, a, r = batch
                s = torch.as_tensor(s, dtype=dtype, device=device)
                a = torch.as_tensor(a, dtype=torch.long, device=device)
                r = torch.as_tensor(r, dtype=dtype, device=device)
                B, L, H, W, C = s.shape
                *_, loss = model(s.reshape(B*L, H, W, C), return_loss=True)            
                
                # Backward pass and optimize
                if loss == float('nan'):
                    print('nan loss')
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                #scheduler.step()

                # Track loss            
                #current_lr = scheduler.get_last_lr()[0]
                current_lr = optimizer.param_groups[0]['lr']
                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                del s
                del a
                del r
                del loss
                batch_bar.set_postfix(loss=batch_loss, lr=current_lr)
            
        if val_ds is not None:
            with tqdm(val_ds, desc='Evaluating', unit='batch', leave=False, disable=True) as eval_bar:
                val_loss = None
                with torch.no_grad():
                    val_loss = 0.0
                    val_num_batches = 0                
                    for eval_batch in eval_bar:
                        s_val, a_val, r_val = eval_batch
                        s_val = torch.as_tensor(s_val, dtype=dtype, device=device)
                        a_val = torch.as_tensor(a_val, dtype=torch.long, device=device)
                        r_val = torch.as_tensor(r_val, dtype=dtype, device=device)
                        B, L, H, W, C = s_val.shape
                        *_, loss = model(s_val.reshape(B*L, H, W, C), return_loss=True)
                        
                        val_loss += loss.item()
                        val_num_batches += 1
                    val_loss /= val_num_batches        
        
        # Calculate average loss for epoch
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        val_losses.append(val_loss)
        epoch_bar.set_postfix(lr=current_lr, loss=avg_epoch_loss, val_loss=val_loss)

        # Clear the current figure and create new plot
        ax.clear()
        
        # Plot updated losses
        ax.plot(train_losses, label='Train')
        ax.plot(val_losses, label='Val')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title(f'Epoch {epoch+1}/{num_epochs}')
        disp_obj.update(f)
        


        #print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}')
        # Print progress        
    
    return losses

# Example usage:

#states = dict()
#for s in [*[o.state for o in env.observations], *[o.next_state for o in env.observations]]:
#    states[hash(s)] = s


#model = NextStatePredictor(embedding_dim=128*4, num_actions=env.action_space.n)
#model.eval()
losses = train(
    train_ds, 
    model, 
    batch_size=128, 
    num_epochs=1000, 
    warmup_steps=10, 
    start_lr=1e-3, 
    min_lr=1e-4, 
    dtype=torch.bfloat16, 
    val_ds=val_ds
)




# %%
# %%time
s, a, r = next(iter(test_ds))
s = torch.tensor(s, dtype=torch.bfloat16, device=available_device())
a = torch.tensor(a, dtype=torch.long, device=available_device())
r = torch.tensor(r, dtype=torch.bfloat16, device=available_device())

with torch.no_grad():
    model.eval()
    #next_emb, next_frames, rewards, loss = model(s, a, r, return_loss=True, return_reconstructed=True, return_rewards=True)
    B, T, H, W, C = s.shape
    recon = model(s.reshape(B*T, H, W, C))
    z = model.encoder(s.reshape(B*T, H, W, C).permute(0, 3, 1, 2)).cpu().to(torch.float).numpy()
    #next_frames = model.state_decoder.forward(next_emb)*255

B, T, H, W, C = s.shape
x1 = (s).cpu().to(torch.uint8).numpy().reshape(B*T, H, W, C)
x2 = (recon[0]).cpu().to(torch.uint8).numpy().reshape(B*T, H, W, C)

x1.shape, x2.shape

# %%
import pandas as pd

pd.DataFrame(z).sample(5).T.round().plot(figsize=(40, 6))


# %%
from PIL import Image
from IPython.display import display


def create_side_by_side(img1: np.ndarray, img2: np.ndarray, scale: int = 3) -> Image.Image:
    """
    Create a side by side image comparison.
    
    Args:
        img1: First image array
        img2: Second image array
        scale: Upscale factor
    
    Returns:
        PIL Image with both images side by side
    """
    # Convert arrays to PIL Images and resize
    pil1 = Image.fromarray(np.uint8(img1)).resize((256 * scale, 240 * scale))
    pil2 = Image.fromarray(np.uint8(img2)).resize((256 * scale, 240 * scale))
    
    # Create new image with space for both
    total_width = pil1.width * 2
    max_height = pil1.height
    combined = Image.new('RGB', (total_width, max_height))
    
    # Paste images side by side
    combined.paste(pil1, (0, 0))
    combined.paste(pil2, (pil1.width, 0))
    
    return combined

# Display side by side comparisons
for i in range(0, len(x1), 8):
    comparison = create_side_by_side(x1[i], x2[i], scale=2)
    display(comparison)


# %%
states = dict()
for s in [*[o.state for o in env.observations], *[o.next_state for o in env.observations]]:
    states[hash(s)] = s

len(states)

# %%
state = model.state_dict()
NextStatePredictor(embedding_dim=128*4, num_actions=env.action_space.n, num_heads=8).load_state_dict(state)

# %%

# %%
corr = torch.corrcoef(torch.tensor(z))
corr = corr*(1-torch.eye(corr.shape[0], device=corr.device))
corr.mean()

