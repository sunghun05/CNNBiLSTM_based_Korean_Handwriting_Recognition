from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """환경에서 얻은 transition을 섞어서 학습하기 위한 경험 리플레이 버퍼입니다."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state, action, reward: float, next_state, done: bool) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).reshape(self.state_dim),
                np.asarray(action, dtype=np.float32).reshape(self.action_dim),
                float(reward),
                np.asarray(next_state, dtype=np.float32).reshape(self.state_dim),
                float(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """현재 layer state를 입력받아 pruning ratio를 출력합니다."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 128) -> None:
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.max_action


class Critic(nn.Module):
    """state와 action을 함께 보고 해당 action의 Q-value를 예측합니다."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


@dataclass
class DDPGConfig:
    state_dim: int
    action_dim: int = 1
    max_action: float = 0.6
    hidden_dim: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    replay_capacity: int = 10000


class DDPGAgent:
    def __init__(self, config: DDPGConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

        # online network는 실제 학습 대상, target network는 Bellman target 안정화용입니다.
        self.actor = Actor(config.state_dim, config.action_dim, config.max_action, config.hidden_dim).to(device)
        self.actor_target = Actor(config.state_dim, config.action_dim, config.max_action, config.hidden_dim).to(device)
        self.critic = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic_target = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay_buffer = ReplayBuffer(config.replay_capacity, config.state_dim, config.action_dim)

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()

        # 결정론적 정책이 한쪽으로 굳지 않도록 탐색 noise를 더합니다.
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
        return np.clip(action, 0.0, self.config.max_action).astype(np.float32)

    def update(self, batch_size: int) -> Dict[str, float]:
        if len(self.replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, self.device)

        # Critic은 Bellman target과 현재 Q-value의 MSE를 줄이는 방향으로 학습합니다.
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.config.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor는 Critic이 평가하는 Q-value가 커지도록 정책을 업데이트합니다.
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.__dict__,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        # target network를 천천히 따라오게 해서 학습 진동을 줄입니다.
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.config.tau)
            target_param.data.add_(self.config.tau * source_param.data)
