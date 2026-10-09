#!/usr/bin/env python3
"""
TopoMario inference system for playing Super Mario Bros.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import os
import json

from .model import TopoMario, TopoMarioConfig


@dataclass
class InferenceSettings:
    """Settings for TopoMario inference."""
    checkpoint_dir: str = "checkpoints_topomario"
    checkpoint_name: str = "last"
    device: Optional[str] = None
    temperature: float = 1.0
    top_k: int = 5


@dataclass
class InferenceReport:
    """Report from inference run."""
    action: int
    action_name: str
    confidence: float
    q_values: list
    state_hash: str


class InferencePipeline:
    """Pipeline for running TopoMario inference."""

    ACTION_NAMES = [
        "NOOP",
        "RIGHT",
        "RIGHT_A",
        "RIGHT_B",
        "RIGHT_A_B",
        "A",
        "LEFT",
    ]

    def __init__(self, settings: InferenceSettings):
        self.settings = settings
        self.device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        config = TopoMarioConfig()
        self.model = TopoMario(config).to(self.device)

        checkpoint_path = os.path.join(
            self.settings.checkpoint_dir,
            self.settings.checkpoint_name,
            "model.safetensors"
        )

        if os.path.exists(checkpoint_path):
            self.model.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")

        self.model.eval()

    def execute(self, screen: np.ndarray, semantic: np.ndarray) -> InferenceReport:
        """Run inference on game state."""
        screen_tensor = torch.from_numpy(screen).float().unsqueeze(0).to(self.device)
        # screen already normalized to [0,1] by the environment

        semantic_tensor = torch.from_numpy(semantic).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _, _ = self.model(screen_tensor, semantic_tensor)

            if self.settings.temperature > 0:
                logits = logits / self.settings.temperature

            if self.settings.top_k > 0:
                values, indices = torch.topk(logits, self.settings.top_k)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, indices, values)
                logits = mask

            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()

            q_values = logits[0].cpu().tolist()

        state_hash = hash(screen.tobytes()) % 10000

        return InferenceReport(
            action=action,
            action_name=self.ACTION_NAMES[action] if action < len(self.ACTION_NAMES) else f"ACTION_{action}",
            confidence=confidence,
            q_values=q_values,
            state_hash=str(state_hash)
        )

    def get_action(self, screen: np.ndarray, semantic: np.ndarray) -> int:
        """Get action for given game state."""
        report = self.execute(screen, semantic)
        return report.action


def run_inference(
    screen: np.ndarray,
    semantic: np.ndarray,
    checkpoint_dir: str = "checkpoints_topomario",
    checkpoint_name: str = "last",
    device: Optional[str] = None,
    temperature: float = 1.0,
    top_k: int = 5,
) -> InferenceReport:
    """Run inference and return report."""
    settings = InferenceSettings(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        device=device,
        temperature=temperature,
        top_k=top_k,
    )
    pipeline = InferencePipeline(settings)
    return pipeline.execute(screen, semantic)


def get_action(
    screen: np.ndarray,
    semantic: np.ndarray,
    checkpoint_dir: str = "checkpoints_topomario",
    checkpoint_name: str = "last",
    device: Optional[str] = None,
) -> int:
    """Get action for given game state."""
    settings = InferenceSettings(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    pipeline = InferencePipeline(settings)
    return pipeline.get_action(screen, semantic)
