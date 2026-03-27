"""CNN-based action predictor: learns which actions cause state changes.

Inspired by the 1st-place ARC-AGI-3 solution (StochasticGoose).
Architecture:
- Input: 16-channel one-hot encoded 64x64 grid
- 4-layer CNN: 32 -> 64 -> 128 -> 256 channels
- Two output heads:
  (a) Sigmoid probabilities for ACTION1-5 (will this action change the frame?)
  (b) 64x64 spatial heatmap for ACTION6 click coordinates
- Trained online during exploration on (state, action) -> frame_changed labels
- Used to bias exploration toward productive actions

Key design decisions from 1st place:
- Binary classification (will this change the state?), not reward prediction
- Experience buffer with hash-based dedup (~200K pairs)
- Reset model weights when advancing to new levels
- Light entropy regularization
"""

import hashlib
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger()

# Number of discrete actions (ACTION1-5). ACTION6 handled by spatial head.
NUM_SIMPLE_ACTIONS = 5
# Grid colors
NUM_COLORS = 16
GRID_SIZE = 64


class ActionPredictorCNN(nn.Module):
    """4-layer CNN that predicts which actions will change the game state."""

    def __init__(self) -> None:
        super().__init__()

        # Encoder: 4 conv layers with batch norm
        self.conv1 = nn.Conv2d(NUM_COLORS, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

        # Head A: simple action probabilities (ACTION1-5)
        # After 3 stride-2 convs: 64 -> 32 -> 16 -> 8
        self.action_pool = nn.AdaptiveAvgPool2d(1)
        self.action_fc = nn.Linear(256, NUM_SIMPLE_ACTIONS)

        # Head B: spatial heatmap for ACTION6 click coordinates
        # Upsample back to 64x64
        self.spatial_up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.spatial_up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.spatial_up3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 16, 64, 64) one-hot encoded grid

        Returns:
            action_probs: (batch, 5) sigmoid probabilities for ACTION1-5
            spatial_map: (batch, 1, 64, 64) heatmap for ACTION6 click target
        """
        # Encoder
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        # Head A: action probabilities
        pooled = self.action_pool(h).flatten(1)
        action_probs = torch.sigmoid(self.action_fc(pooled))

        # Head B: spatial heatmap
        s = F.relu(self.spatial_up1(h))
        s = F.relu(self.spatial_up2(s))
        spatial_map = torch.sigmoid(self.spatial_up3(s))

        return action_probs, spatial_map


def grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    """Convert a 64x64 grid of color indices (0-15) to 16-channel one-hot encoding."""
    h, w = grid.shape
    onehot = np.zeros((NUM_COLORS, h, w), dtype=np.float32)
    for c in range(NUM_COLORS):
        onehot[c] = (grid == c).astype(np.float32)
    return onehot


class ExperienceBuffer:
    """Hash-deduplicated experience buffer for online CNN training.

    Stores (state, action, did_change) tuples with deduplication
    based on (state_hash, action) pairs.
    """

    def __init__(self, max_size: int = 200_000) -> None:
        self.max_size = max_size
        self.states: list[np.ndarray] = []  # one-hot encoded (16, 64, 64)
        self.actions: list[int] = []  # action index (0-4 for simple, 5 for click)
        self.click_coords: list[Optional[tuple[int, int]]] = []  # (x, y) for ACTION6
        self.labels: list[float] = []  # 1.0 = changed, 0.0 = no change
        self._seen: set[str] = set()

    def add(
        self,
        state: np.ndarray,
        action_idx: int,
        did_change: bool,
        click_xy: Optional[tuple[int, int]] = None,
    ) -> bool:
        """Add experience. Returns True if it was new (not a duplicate)."""
        # Hash for dedup
        state_hash = hashlib.md5(state.tobytes()).hexdigest()[:12]
        key = f"{state_hash}_{action_idx}"
        if click_xy:
            key += f"_{click_xy[0]}_{click_xy[1]}"

        if key in self._seen:
            return False

        self._seen.add(key)
        self.states.append(state)
        self.actions.append(action_idx)
        self.click_coords.append(click_xy)
        self.labels.append(1.0 if did_change else 0.0)

        # Evict oldest if over capacity
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.click_coords.pop(0)
            self.labels.pop(0)

        return True

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.click_coords.clear()
        self.labels.clear()
        self._seen.clear()


class ActionPredictor:
    """High-level interface for the CNN action predictor.

    Handles:
    - Online training from experience buffer
    - Predicting action-change probabilities for a given state
    - Model reset on level changes
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = ActionPredictorCNN().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.buffer = ExperienceBuffer()

        self._train_steps = 0
        self._last_loss = 0.0

    def predict(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict change probability for each action given current grid.

        Args:
            grid: 64x64 numpy array of color values (0-15)

        Returns:
            action_probs: (5,) probabilities for ACTION1-5 causing a change
            spatial_map: (64, 64) heatmap for ACTION6 click effectiveness
        """
        self.model.eval()
        onehot = grid_to_onehot(grid)
        x = torch.from_numpy(onehot).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs, spatial_map = self.model(x)

        return (
            action_probs[0].cpu().numpy(),
            spatial_map[0, 0].cpu().numpy(),
        )

    def record(
        self,
        grid: np.ndarray,
        action_idx: int,
        did_change: bool,
        click_xy: Optional[tuple[int, int]] = None,
    ) -> None:
        """Record an experience for later training."""
        onehot = grid_to_onehot(grid)
        self.buffer.add(onehot, action_idx, did_change, click_xy)

    def train_step(self, batch_size: int = 64, entropy_weight: float = 0.01) -> float:
        """Train one batch from the experience buffer.

        Returns the loss value.
        """
        if len(self.buffer) < batch_size:
            return 0.0

        self.model.train()

        # Sample a random batch
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        batch_states = np.array([self.buffer.states[i] for i in indices])
        batch_actions = np.array([self.buffer.actions[i] for i in indices])
        batch_labels = np.array([self.buffer.labels[i] for i in indices])

        states_t = torch.from_numpy(batch_states).to(self.device)
        labels_t = torch.from_numpy(batch_labels).float().to(self.device)

        action_probs, spatial_map = self.model(states_t)

        # Loss for simple actions (ACTION1-5, idx 0-4)
        simple_mask = batch_actions < NUM_SIMPLE_ACTIONS
        loss = torch.tensor(0.0, device=self.device)

        if simple_mask.any():
            simple_indices = np.where(simple_mask)[0]
            action_indices = batch_actions[simple_indices]

            # Gather the predicted probability for the actual action taken
            pred = action_probs[
                torch.from_numpy(simple_indices).long().to(self.device),
                torch.from_numpy(action_indices).long().to(self.device),
            ]
            target = labels_t[torch.from_numpy(simple_indices).long().to(self.device)]
            loss += F.binary_cross_entropy(pred, target)

        # Loss for click actions (ACTION6, idx 5)
        click_mask = batch_actions == NUM_SIMPLE_ACTIONS
        if click_mask.any():
            click_indices = np.where(click_mask)[0]
            for idx in click_indices:
                buf_idx = indices[idx]
                xy = self.buffer.click_coords[buf_idx]
                if xy is not None:
                    x, y = xy
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        pred_val = spatial_map[idx, 0, y, x]
                        target_val = labels_t[idx]
                        loss += F.binary_cross_entropy(pred_val.unsqueeze(0), target_val.unsqueeze(0))

        # Entropy regularization: encourage exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-8) +
                     (1 - action_probs) * torch.log(1 - action_probs + 1e-8)).mean()
        loss -= entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        self._last_loss = loss.item()
        return self._last_loss

    def train_epoch(self, n_steps: int = 10, batch_size: int = 64) -> float:
        """Train for multiple steps. Returns average loss."""
        if len(self.buffer) < batch_size:
            return 0.0
        losses = [self.train_step(batch_size) for _ in range(n_steps)]
        return sum(losses) / len(losses) if losses else 0.0

    def reset_model(self) -> None:
        """Reset model weights (e.g., on level change). Keep buffer."""
        self.model = ActionPredictorCNN().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self._train_steps = 0
        logger.info("CNN model weights reset")

    def reset_all(self) -> None:
        """Reset both model and buffer."""
        self.reset_model()
        self.buffer.clear()
        logger.info("CNN model and buffer reset")

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "buffer_size": len(self.buffer),
            "train_steps": self._train_steps,
            "last_loss": round(self._last_loss, 4),
        }
