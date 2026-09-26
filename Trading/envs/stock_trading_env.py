from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """Simula compras y ventas con una acción discreta.

    Acciones: ``0 = mantener``, ``1 = comprar`` y ``2 = vender``.
    la recompensa base es el retorno porcentual del portafolio:
    ``(Portfolio_Value_t - Portfolio_Value_t-1) / Portfolio_Value_t-1``.
    """

    metadata = {"render_modes": ["human"]}
    HOLD, BUY, SELL = 0, 1, 2

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
        trade_fraction: float = 0.10,
        reward_scaling: float = 1.0,
        volatility_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
        feature_cols: list[str] | None = None,
    ) -> None:
        super().__init__()
        if df.empty:
            raise ValueError("El DataFrame de precios está vacío.")
        if "Close" not in df.columns:
            raise ValueError("El DataFrame debe contener la columna 'Close'.")
        if (df["Close"] <= 0).any():
            raise ValueError("La columna 'Close' debe contener precios positivos.")
        if initial_cash <= 0:
            raise ValueError("initial_cash debe ser mayor que cero.")
        if not 0 <= commission:
            raise ValueError("commission no puede ser negativa.")
        if not 0 < trade_fraction <= 1:
            raise ValueError("trade_fraction debe estar entre 0 y 1.")

        self.df = df.reset_index(drop=True).copy()
        if "Return" not in self.df:
            self.df["Return"] = self.df["Close"].pct_change().fillna(0.0)
        self.df = self.df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        self.initial_cash = float(initial_cash)
        self.commission = float(commission)
        self.trade_fraction = float(trade_fraction)
        self.reward_scaling = float(reward_scaling)
        self.volatility_penalty = float(volatility_penalty)
        self.drawdown_penalty = float(drawdown_penalty)

        default_features = [
            "Close", "Return", "RSI", "MACD", "MACD_Signal",
            "SMA_20", "SMA_50", "Volatility",
        ]
        self.feature_cols = (
            [column for column in default_features if column in self.df]
            if feature_cols is None
            else list(feature_cols)
        )
        missing = [column for column in self.feature_cols if column not in self.df]
        if missing:
            raise ValueError(f"Columnas no encontradas en df: {missing}")
        if not self.feature_cols:
            raise ValueError("No hay columnas válidas para la observación.")

        features = self.df[self.feature_cols].to_numpy(dtype=np.float64)
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(3 + len(self.feature_cols),),
            dtype=np.float32,
        )
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.prev_portfolio = self.initial_cash
        self.peak_portfolio = self.initial_cash
        self.portfolio_history: list[float] = []
        self.action_history: list[int] = []
        self.price_history: list[float] = []

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def _get_obs(self, price: float, portfolio: float) -> np.ndarray:
        account = np.array(
            [
                self.cash / self.initial_cash,
                self.shares * price / self.initial_cash,
                portfolio / self.initial_cash,
            ],
            dtype=np.float64,
        )
        market = (
            self.df.loc[self.current_step, self.feature_cols].to_numpy(dtype=np.float64)
            - self.feature_mean
        ) / self.feature_std
        return np.clip(np.concatenate((account, market)), -10, 10).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.prev_portfolio = self.initial_cash
        self.peak_portfolio = self.initial_cash
        first_price = float(self.df.loc[0, "Close"])
        self.portfolio_history = [self.initial_cash]
        self.action_history = []
        self.price_history = [first_price]
        return self._get_obs(first_price, self.initial_cash), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")

        price = float(self.df.loc[self.current_step, "Close"])
        executed_action = self.HOLD
        if action == self.BUY:
            budget = self.cash * self.trade_fraction
            shares_to_buy = int(budget // (price * (1 + self.commission)))
            if shares_to_buy:
                self.cash -= shares_to_buy * price * (1 + self.commission)
                self.shares += shares_to_buy
                executed_action = self.BUY
        elif action == self.SELL:
            shares_to_sell = min(
                int(self.shares),
                max(1, int(self.shares * self.trade_fraction)) if self.shares else 0,
            )
            if shares_to_sell:
                self.cash += shares_to_sell * price * (1 - self.commission)
                self.shares -= shares_to_sell
                executed_action = self.SELL

        portfolio = self._portfolio_value(price)
        step_return = (portfolio - self.prev_portfolio) / max(self.prev_portfolio, 1e-8)
        self.peak_portfolio = max(self.peak_portfolio, portfolio)
        drawdown = (self.peak_portfolio - portfolio) / max(self.peak_portfolio, 1e-8)
        reward = step_return
        reward -= self.volatility_penalty * abs(step_return)
        reward -= self.drawdown_penalty * drawdown
        reward *= self.reward_scaling
        self.prev_portfolio = portfolio

        self.portfolio_history.append(portfolio)
        self.action_history.append(executed_action)
        self.price_history.append(price)
        self.current_step += 1
        terminated = portfolio <= 0
        truncated = self.current_step >= len(self.df)
        if truncated:
            self.current_step = len(self.df) - 1
        next_price = float(self.df.loc[self.current_step, "Close"])
        observation = self._get_obs(next_price, portfolio)
        info = {
            "portfolio": portfolio,
            "cash": self.cash,
            "shares": self.shares,
            "price": price,
            "action_requested": int(action),
            "action_executed": executed_action,
            "return": step_return,
            "drawdown": drawdown,
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        price = self.price_history[-1] if self.price_history else float("nan")
        portfolio = (
            self.portfolio_history[-1]
            if self.portfolio_history
            else float("nan")
        )
        print(
            f"Paso {self.current_step} | Precio ${price:.2f} | "
            f"Efectivo ${self.cash:.2f} | Acciones {self.shares} | "
            f"Portafolio ${portfolio:.2f}"
        )
