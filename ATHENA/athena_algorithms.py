#!/usr/bin/env python3
"""
ATHENA RL Algorithms

Modification 5: SAC (Soft Actor-Critic) replaces TD3.
  - Stochastic policy with entropy regularisation
  - Automatic temperature (alpha) tuning
  Reference: Haarnoja et al. (2018)

Modification 6: AdaptiveConfidenceGate replaces the fixed 70/30 blend.
  - A small network learns when to trust RL vs supervised signals
  - Inputs: market features, regime probs, recent performance
  - Output: gate in [0, 1]

The multi-agent structure (coordinator, position sizer, regime agent)
is reused from ARTEMIS with the same architecture.
"""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Coordinator (same as ARTEMIS)
# ---------------------------------------------------------------------------

class ATHENACoordinatorAgent(nn.Module):
    def __init__(self, n_models=5, market_features=10, hidden_size=128):
        super().__init__()
        self.weight_network = nn.Sequential(
            nn.Linear(n_models + market_features, hidden_size),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, n_models), nn.Softmax(dim=-1),
        )
        self.performance_tracker = nn.Sequential(
            nn.Linear(n_models, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, model_predictions, market_features):
        combined = torch.cat([model_predictions, market_features], dim=-1)
        weights = self.weight_network(combined)
        perf = self.performance_tracker(model_predictions)
        return weights, perf


# ---------------------------------------------------------------------------
# Position Sizer (same as ARTEMIS)
# ---------------------------------------------------------------------------

class ATHENAPositionSizerAgent(nn.Module):
    def __init__(self, market_features=20, hidden_size=96):
        super().__init__()
        self.risk_assessor = nn.Sequential(
            nn.Linear(market_features, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.volatility_predictor = nn.Sequential(
            nn.Linear(market_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.position_calculator = nn.Sequential(
            nn.Linear(market_features + 2, 64), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, market_features):
        risk = self.risk_assessor(market_features)
        vol = self.volatility_predictor(market_features)
        pos_in = torch.cat([market_features, risk, vol], dim=-1)
        pos_mult = self.position_calculator(pos_in)
        return pos_mult * (1.0 - risk * 0.5), risk, vol


# ---------------------------------------------------------------------------
# Regime Agent (same as ARTEMIS)
# ---------------------------------------------------------------------------

class ATHENARegimeAgent(nn.Module):
    def __init__(self, input_size, hidden_size=80, n_regimes=5):
        super().__init__()
        self.n_regimes = n_regimes
        self.short_gru = nn.GRU(input_size, hidden_size // 2, 1, batch_first=True)
        self.medium_gru = nn.GRU(input_size, hidden_size // 2, 1, batch_first=True)
        self.long_gru = nn.GRU(input_size, hidden_size // 2, 1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_regimes),
        )
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)
        seq_len = x.shape[1]
        short_seq = x[:, -min(10, seq_len):]
        medium_seq = x[:, -min(30, seq_len):]
        _, sh = self.short_gru(short_seq)
        _, mh = self.medium_gru(medium_seq)
        _, lh = self.long_gru(x)
        combined = torch.cat([sh[-1], mh[-1], lh[-1]], dim=-1)
        regime_probs = F.softmax(self.classifier(combined), dim=-1)
        conf = self.confidence(combined)
        return regime_probs, conf, None


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ATHENAReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# SAC Agent  (Modification 5 – replaces TD3)
# ---------------------------------------------------------------------------

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ATHENASACAgent(nn.Module):
    """
    Soft Actor-Critic with automatic entropy tuning.
    Replaces the deterministic TD3 policy in ARTEMIS.
    """

    def __init__(self, state_size, action_size, hidden_size=256, lr=1e-4, device=None):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Stochastic actor (outputs mean + log_std)
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_size // 2, action_size)
        self.actor_log_std = nn.Linear(hidden_size // 2, action_size)

        # Twin critics
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Automatic entropy tuning
        self.target_entropy = -float(action_size)
        self.log_alpha = nn.Parameter(torch.zeros(1))

        self.actor_optimizer = optim.Adam(
            list(self.actor_backbone.parameters()) +
            list(self.actor_mean.parameters()) +
            list(self.actor_log_std.parameters()),
            lr=lr,
        )
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr * 2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr * 2)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.memory = ATHENAReplayBuffer(100000)
        self.gamma = 0.99
        self.tau = 0.005
        self.max_position = 1.0
        self.risk_threshold = 0.02

        self.to(self.device)

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_size + self.action_size, self.hidden_size),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample_action(self, state_tensor):
        """Reparameterised Gaussian sample → (action, log_prob)."""
        backbone_out = self.actor_backbone(state_tensor)
        mean = self.actor_mean(backbone_out)
        log_std = self.actor_log_std(backbone_out).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # Compute log-prob with tanh squashing correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act(self, state, add_noise=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self._sample_action(state_t)
        action = action.cpu().numpy().flatten()
        action = np.clip(action, -self.max_position, self.max_position)
        vol = np.std(state[-20:]) if len(state) >= 20 else 0.01
        if vol > self.risk_threshold:
            action *= 0.7
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size=256):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # --- Update critics ---
        with torch.no_grad():
            next_action, next_log_prob = self._sample_action(next_states)
            tq1 = self.critic1_target(torch.cat([next_states, next_action], 1))
            tq2 = self.critic2_target(torch.cat([next_states, next_action], 1))
            target_q = torch.min(tq1, tq2) - self.alpha.detach() * next_log_prob
            target_q = rewards + (~dones) * self.gamma * target_q

        for critic, opt in [(self.critic1, self.critic1_optimizer),
                            (self.critic2, self.critic2_optimizer)]:
            q = critic(torch.cat([states, actions], 1))
            loss = F.mse_loss(q, target_q)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- Update actor ---
        new_action, log_prob = self._sample_action(states)
        q1 = self.critic1(torch.cat([states, new_action], 1))
        actor_loss = (self.alpha.detach() * log_prob - q1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update alpha ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft-update targets ---
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, source, target):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# ---------------------------------------------------------------------------
# Adaptive Confidence Gate  (Modification 6 – replaces fixed 70/30)
# ---------------------------------------------------------------------------

class AdaptiveConfidenceGate(nn.Module):
    """
    Learns when to trust supervised vs RL signals based on market state.

    Output gate ∈ [0, 1]:
      high  → trust supervised more
      low   → trust RL more

    enhanced_signal = gate * supervised + (1-gate) * rl
    """

    def __init__(self, market_dim=20, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(market_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, market_features):
        """market_features: (B, market_dim) or (market_dim,)"""
        if market_features.dim() == 1:
            market_features = market_features.unsqueeze(0)
        return self.net(market_features).squeeze(-1)


# ---------------------------------------------------------------------------
# Multi-Agent System (uses SAC instead of TD3)
# ---------------------------------------------------------------------------

class ATHENAMultiAgentSystem:
    def __init__(self, state_size, n_models=5, market_features=20, device=None):
        self.state_size = state_size
        self.n_models = n_models
        self.market_features = market_features
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.coordinator = ATHENACoordinatorAgent(n_models, market_features // 2).to(self.device)
        self.position_sizer = ATHENAPositionSizerAgent(market_features).to(self.device)
        self.regime_agent = ATHENARegimeAgent(state_size).to(self.device)
        self.sac_agent = ATHENASACAgent(state_size + market_features, 1, device=self.device)

        self.performance_history = []
        self.episode_rewards = []

    def act(self, state, model_predictions, market_features):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preds_t = torch.FloatTensor(model_predictions).unsqueeze(0).to(self.device)
        market_t = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)

        weights, _ = self.coordinator(preds_t, market_t[:, :10])
        pos_mult, _, _ = self.position_sizer(market_t)
        regime_probs, _, _ = self.regime_agent(state_t.unsqueeze(0))

        combined_state = np.concatenate([state, market_features])
        base_action = self.sac_agent.act(combined_state)

        weighted_pred = torch.sum(preds_t * weights, dim=1)

        action = (
            0.6 * base_action[0] + 0.4 * weighted_pred.item()
        ) * pos_mult.item()

        dominant = torch.argmax(regime_probs).item()
        if dominant == 1:
            action *= 0.7
        elif dominant == 3:
            action *= 0.8

        return np.clip(action, -0.8, 0.8)

    def update(self, state, action, reward, next_state, done,
               model_predictions, market_features):
        cs = np.concatenate([state, market_features])
        cns = np.concatenate([next_state, market_features])
        self.sac_agent.remember(cs, [action], reward, cns, done)
        self.sac_agent.replay()

        self.episode_rewards.append(reward)
        if done:
            self.performance_history.append(sum(self.episode_rewards))
            self.episode_rewards = []

    def get_performance_metrics(self):
        if not self.performance_history:
            return {'avg_return': 0, 'std_return': 0, 'best_return': 0}
        return {
            'avg_return': np.mean(self.performance_history[-100:]),
            'std_return': np.std(self.performance_history[-100:]),
            'best_return': np.max(self.performance_history),
            'recent_trend': (
                np.mean(self.performance_history[-10:])
                if len(self.performance_history) >= 10 else 0
            ),
        }
