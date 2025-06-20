#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import copy

class ARTEMISCoordinatorAgent(nn.Module):
    """ARTEMIS Coordinator Agent: Manages ensemble model coordination"""
    def __init__(self, n_models=5, market_features=10, hidden_size=128):
        super().__init__()
        self.n_models = n_models
        self.market_features = market_features
        
        # Model weight predictor
        self.weight_network = nn.Sequential(
            nn.Linear(n_models + market_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_models),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracker
        self.performance_tracker = nn.Sequential(
            nn.Linear(n_models, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, model_predictions, market_features):
        # Combine predictions and market context
        combined_input = torch.cat([model_predictions, market_features], dim=-1)
        
        # Generate adaptive weights
        weights = self.weight_network(combined_input)
        
        # Predict ensemble performance
        performance = self.performance_tracker(model_predictions)
        
        return weights, performance

class ARTEMISPositionSizerAgent(nn.Module):
    """ARTEMIS Position Sizer Agent: Dynamic risk-adjusted position sizing"""
    def __init__(self, market_features=20, hidden_size=96):
        super().__init__()
        
        # Risk assessment network
        self.risk_assessor = nn.Sequential(
            nn.Linear(market_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Volatility predictor
        self.volatility_predictor = nn.Sequential(
            nn.Linear(market_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Position size calculator
        self.position_calculator = nn.Sequential(
            nn.Linear(market_features + 2, 64),  # +2 for risk and volatility
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, market_features):
        # Assess market risk
        risk_level = self.risk_assessor(market_features)
        
        # Predict volatility
        volatility = self.volatility_predictor(market_features)
        
        # Calculate optimal position size
        position_input = torch.cat([market_features, risk_level, volatility], dim=-1)
        position_multiplier = self.position_calculator(position_input)
        
        # Scale position based on risk (more conservative in high risk)
        risk_adjusted_position = position_multiplier * (1.0 - risk_level * 0.5)
        
        return risk_adjusted_position, risk_level, volatility

class ARTEMISRegimeAgent(nn.Module):
    """ARTEMIS Regime Agent: Enhanced market regime detection"""
    def __init__(self, input_size, hidden_size=80, n_regimes=5):
        super().__init__()
        self.n_regimes = n_regimes
        
        # Multi-scale temporal analysis
        self.short_term_gru = nn.GRU(input_size, hidden_size//2, 1, batch_first=True)
        self.medium_term_gru = nn.GRU(input_size, hidden_size//2, 1, batch_first=True)
        self.long_term_gru = nn.GRU(input_size, hidden_size//2, 1, batch_first=True)
        
        # Regime classification
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_regimes)
        )
        
        # Regime confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Regime transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_size + hidden_size//2, 64),
            nn.ReLU(),
            nn.Linear(64, n_regimes * n_regimes)
        )
        
    def forward(self, x):
        # Handle different input dimensions
        if len(x.shape) == 2:
            # (batch_size, features) -> (batch_size, 1, features)
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            # (batch_size, 1, 1, features) -> (batch_size, 1, features)
            x = x.squeeze(1)
        
        batch_size, seq_len, features = x.shape
        
        # Multi-scale analysis
        # Handle cases where sequence is shorter than expected
        short_seq = x[:, -min(10, seq_len):]  # Last 10 steps or available
        medium_seq = x[:, -min(30, seq_len):]  # Last 30 steps or available
        
        short_out, short_hidden = self.short_term_gru(short_seq)
        medium_out, medium_hidden = self.medium_term_gru(medium_seq)
        long_out, long_hidden = self.long_term_gru(x)  # Full sequence
        
        # Combine temporal scales
        combined_features = torch.cat([
            short_hidden[-1],
            medium_hidden[-1], 
            long_hidden[-1]
        ], dim=-1)
        
        # Regime classification
        regime_logits = self.regime_classifier(combined_features)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Confidence estimation
        confidence = self.confidence_estimator(combined_features)
        
        # Transition probabilities
        transition_logits = self.transition_predictor(combined_features)
        transition_probs = F.softmax(transition_logits.view(batch_size, self.n_regimes, self.n_regimes), dim=-1)
        
        return regime_probs, confidence, transition_probs

class ARTEMISTD3Agent(nn.Module):
    """ARTEMIS Advanced TD3 RL Agent with enhanced features"""
    def __init__(self, state_size, action_size, hidden_size=256, lr=1e-4):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        
        # Actor Networks (with enhanced architecture)
        self.actor = self._build_actor()
        self.actor_target = copy.deepcopy(self.actor)
        
        # Twin Critic Networks
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr*2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr*2)
        
        # Enhanced replay buffer
        self.memory = ARTEMISReplayBuffer(100000)
        
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.update_counter = 0
        
        # Risk management
        self.max_position = 1.0
        self.risk_threshold = 0.02
        
    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.action_size),
            nn.Tanh()
        )
    
    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_size + self.action_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1)
        )
    
    def act(self, state, add_noise=True):
        """Select action with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.policy_noise, size=self.action_size)
            action = action + noise
            action = np.clip(action, -1, 1)
        
        # Apply risk management
        action = self._apply_risk_management(action, state)
        
        return action
    
    def _apply_risk_management(self, action, state):
        """Apply risk management constraints"""
        # Limit maximum position size
        action = np.clip(action, -self.max_position, self.max_position)
        
        # Reduce position in high volatility
        volatility = self._estimate_volatility(state)
        if volatility > self.risk_threshold:
            action *= 0.7
        
        return action
    
    def _estimate_volatility(self, state):
        """Simple volatility estimation from state"""
        if len(state.shape) > 1:
            return torch.std(state[:, -20:]).item() if state.shape[1] > 20 else 0.01
        return 0.01
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size=256):
        """Train the agent"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # Update critics
        self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor (delayed)
        self.update_counter += 1
        if self.update_counter % self.policy_freq == 0:
            self._update_actor(states)
            self._soft_update_targets()
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            # Target Q values
            target_q1 = self.critic1_target(torch.cat([next_states, next_actions], 1))
            target_q2 = self.critic2_target(torch.cat([next_states, next_actions], 1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (~dones) * self.gamma * target_q
        
        # Current Q values
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
    
    def _update_actor(self, states):
        """Update actor network"""
        actions = self.actor(states)
        actor_loss = -self.critic1(torch.cat([states, actions], 1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ARTEMISReplayBuffer:
    """Enhanced replay buffer for ARTEMIS RL agents"""
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

class ARTEMISMultiAgentSystem:
    """ARTEMIS Multi-Agent System: Coordinates all RL agents"""
    def __init__(self, state_size, n_models=5, market_features=20):
        self.state_size = state_size
        self.n_models = n_models
        self.market_features = market_features
        
        # Initialize agents
        self.coordinator_agent = ARTEMISCoordinatorAgent(n_models, market_features//2)
        self.position_sizer_agent = ARTEMISPositionSizerAgent(market_features)
        self.regime_agent = ARTEMISRegimeAgent(state_size)
        self.td3_agent = ARTEMISTD3Agent(state_size + market_features, 1)
        
        # Performance tracking
        self.performance_history = []
        self.episode_rewards = []
        
    def act(self, state, model_predictions, market_features):
        """Coordinated action selection"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        predictions_tensor = torch.FloatTensor(model_predictions).unsqueeze(0)
        market_tensor = torch.FloatTensor(market_features).unsqueeze(0)
        
        # Get ensemble weights from coordinator
        weights, ensemble_performance = self.coordinator_agent(predictions_tensor, market_tensor[:, :10])
        
        # Get position sizing
        position_multiplier, risk_level, volatility = self.position_sizer_agent(market_tensor)
        
        # Get regime information
        regime_probs, regime_confidence, _ = self.regime_agent(state_tensor.unsqueeze(0))
        
        # Get base action from TD3
        combined_state = np.concatenate([state, market_features])
        base_action = self.td3_agent.act(combined_state)
        
        # Combine all agent outputs
        weighted_prediction = torch.sum(predictions_tensor * weights, dim=1)
        
        # Final action with risk management
        risk_adjusted_action = (
            0.6 * base_action[0] + 
            0.4 * weighted_prediction.item()
        ) * position_multiplier.item()
        
        # Apply regime-based adjustments
        dominant_regime = torch.argmax(regime_probs).item()
        if dominant_regime == 1:  # Bear market
            risk_adjusted_action *= 0.7
        elif dominant_regime == 3:  # High volatility
            risk_adjusted_action *= 0.8
        
        return np.clip(risk_adjusted_action, -0.8, 0.8)
    
    def update(self, state, action, reward, next_state, done, model_predictions, market_features):
        """Update all agents"""
        # Store experience for TD3
        combined_state = np.concatenate([state, market_features])
        combined_next_state = np.concatenate([next_state, market_features])
        
        self.td3_agent.remember(combined_state, [action], reward, combined_next_state, done)
        
        # Train TD3 agent
        self.td3_agent.replay()
        
        # Track performance
        self.episode_rewards.append(reward)
        
        if done:
            episode_return = sum(self.episode_rewards)
            self.performance_history.append(episode_return)
            self.episode_rewards = []
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        if not self.performance_history:
            return {'avg_return': 0, 'std_return': 0, 'best_return': 0}
        
        return {
            'avg_return': np.mean(self.performance_history[-100:]),
            'std_return': np.std(self.performance_history[-100:]),
            'best_return': np.max(self.performance_history),
            'recent_trend': np.mean(self.performance_history[-10:]) if len(self.performance_history) >= 10 else 0
        }

# Legacy compatibility
def create_advanced_td3_agent(state_size, action_size, lr=1e-4):
    """Legacy compatibility function"""
    return ARTEMISTD3Agent(state_size, action_size, lr=lr)

def create_multi_agent_system(state_size, n_models=5, market_features=20):
    """Legacy compatibility function"""
    return ARTEMISMultiAgentSystem(state_size, n_models, market_features) 