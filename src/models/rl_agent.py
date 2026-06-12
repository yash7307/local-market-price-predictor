"""
RL-Style Pricing Agent
-----------------------
Implements a rule-based pricing agent that mimics Reinforcement Learning
concepts (state, action, reward) without requiring stable-baselines3.

State  : [price_ratio, inventory_ratio, is_weekend, is_festival, temperature_norm]
Actions: DECREASE (-10%), KEEP (0%), INCREASE (+10%)
Reward : daily_revenue - stockout_penalty - overstock_penalty

To upgrade to true RL (PPO/DQN), uncomment the stable-baselines3 section
and replace RuleBasedAgent with SB3Agent.

TODO:
    - Implement proper Gymnasium environment for SB3 integration.
    - Add multi-step episode simulation.
    - Log reward curves for training visualization.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


# ── Action Space ─────────────────────────────────────────────────────────────

ACTION_DECREASE = 0
ACTION_KEEP     = 1
ACTION_INCREASE = 2
ACTION_NAMES    = ["Decrease (-10%)", "Keep Price", "Increase (+10%)"]
PRICE_CHANGE    = {ACTION_DECREASE: -0.10, ACTION_KEEP: 0.0, ACTION_INCREASE: 0.10}


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class PricingState:
    """Normalized state vector for the pricing agent."""
    current_price:    float   # ₹
    competitor_price: float   # ₹
    inventory:        int     # units
    is_weekend:       int     # 0 or 1
    is_festival:      int     # 0 or 1
    temperature:      float   # °C

    def to_vector(self) -> np.ndarray:
        """Convert state to a normalized numpy vector."""
        price_ratio      = self.current_price / max(self.competitor_price, 1)
        inventory_norm   = self.inventory / 500.0
        temperature_norm = (self.temperature - 10) / 40.0
        return np.array([
            price_ratio,
            inventory_norm,
            float(self.is_weekend),
            float(self.is_festival),
            temperature_norm,
        ], dtype=np.float32)


# ── Environment ───────────────────────────────────────────────────────────────

class PricingEnvironment:
    """
    Simulated daily pricing environment.

    The agent sets a price each day, the environment returns
    simulated demand → revenue → reward.
    """

    def __init__(
        self,
        demand_model=None,
        initial_price: float = 120.0,
        price_min: float = 50.0,
        price_max: float = 500.0,
        stockout_penalty: float = 50.0,
        overstock_penalty: float = 0.1,
    ):
        self.demand_model     = demand_model
        self.initial_price    = initial_price
        self.price_min        = price_min
        self.price_max        = price_max
        self.stockout_penalty = stockout_penalty
        self.overstock_penalty= overstock_penalty
        self.reset()

    def reset(self) -> PricingState:
        """Reset environment to initial state."""
        self.current_price = self.initial_price
        self.step_count    = 0
        self.total_reward  = 0.0
        self.history: list[dict] = []
        return self._get_state()

    def _get_state(self) -> PricingState:
        rng = np.random.default_rng(self.step_count)
        return PricingState(
            current_price    = self.current_price,
            competitor_price = rng.uniform(90, 160),
            inventory        = int(rng.integers(20, 200)),
            is_weekend       = int(self.step_count % 7 >= 5),
            is_festival      = int(rng.random() < 0.07),
            temperature      = float(25 + 8 * np.sin(2 * np.pi * self.step_count / 365)),
        )

    def step(self, action: int) -> tuple[PricingState, float, bool]:
        """
        Apply action, simulate demand, compute reward.

        Returns:
            (next_state, reward, done)
        """
        # Apply price change
        change_pct         = PRICE_CHANGE[action]
        new_price          = self.current_price * (1 + change_pct)
        self.current_price = float(np.clip(new_price, self.price_min, self.price_max))

        state = self._get_state()

        # Simulate demand (uses ML model if available, else rule-based)
        demand = self._simulate_demand(state)

        # Compute revenue and penalties
        revenue = self.current_price * demand
        stockout   = self.stockout_penalty * max(0, demand - state.inventory)
        overstock  = self.overstock_penalty * max(0, state.inventory - demand)
        reward     = revenue - stockout - overstock

        self.total_reward += reward
        self.step_count   += 1

        self.history.append({
            "step":    self.step_count,
            "price":   round(self.current_price, 2),
            "demand":  round(demand, 1),
            "revenue": round(revenue, 2),
            "reward":  round(reward, 2),
            "action":  ACTION_NAMES[action],
        })

        done = self.step_count >= 365  # one year episode
        return state, reward, done

    def _simulate_demand(self, state: PricingState) -> float:
        """Compute demand using ML model or fallback formula."""
        if self.demand_model is not None:
            from src.models.demand_model import FEATURES
            row = {
                "price":            state.current_price,
                "competitor_price": state.competitor_price,
                "is_weekend":       state.is_weekend,
                "is_festival":      state.is_festival,
                "inventory":        state.inventory,
                "month":            (self.step_count % 365 // 30) + 1,
                "day_of_week":      self.step_count % 7,
                "temperature":      state.temperature,
            }
            import pandas as _pd
            return max(0, float(self.demand_model.predict(_pd.DataFrame([row])[FEATURES])[0]))

        # Fallback: simple rule-based demand
        base = 100
        demand = (
            base
            - 0.5 * state.current_price
            + 0.3 * state.competitor_price
            + 10  * state.is_weekend
            + 25  * state.is_festival
            + np.random.normal(0, 5)
        )
        return max(0.0, float(demand))

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ── Rule-Based Agent ──────────────────────────────────────────────────────────

class RuleBasedAgent:
    """
    A heuristic pricing agent that follows simple business rules.
    Useful as a baseline to compare against true RL agents.

    Rules:
        1. If inventory is high (>150) AND current_price >= competitor_price → DECREASE
        2. If is_festival or is_weekend AND current_price < competitor_price → INCREASE
        3. If current_price > competitor_price * 1.1 → DECREASE
        4. Otherwise → KEEP
    """

    def select_action(self, state: PricingState) -> int:
        """Choose an action given the current state."""
        ratio = state.current_price / max(state.competitor_price, 1)

        if state.inventory > 150 and ratio >= 1.0:
            return ACTION_DECREASE

        if (state.is_festival or state.is_weekend) and ratio < 1.0:
            return ACTION_INCREASE

        if ratio > 1.10:
            return ACTION_DECREASE

        return ACTION_KEEP

    def run_episode(self, env: PricingEnvironment) -> pd.DataFrame:
        """Run a full 365-step episode and return history."""
        state = env.reset()
        done  = False
        while not done:
            action = self.select_action(state)
            state, _, done = env.step(action)
        return env.get_history_df()


# ─────────────────────────────────────────────────────────────────────────────
# To use Stable-Baselines3 PPO (uncomment + install stable-baselines3):
#
# import gymnasium as gym
# from stable_baselines3 import PPO
#
# class SB3PricingEnv(gym.Env):
#     ...  # wrap PricingEnvironment as a gym.Env
#
# def train_ppo(env: SB3PricingEnv, timesteps: int = 100_000):
#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=timesteps)
#     return model
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    env   = PricingEnvironment()
    agent = RuleBasedAgent()
    hist  = agent.run_episode(env)
    print(f"Total Reward (365 days): ₹{env.total_reward:,.0f}")
    print(hist.tail(10).to_string(index=False))
