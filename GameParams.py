from dataclasses import dataclass

import numpy as np


@dataclass
class GameParams:
    D_full: float = 110.0
    alpha: float = 0.2

    def __post_init__(self) -> None:
        self.d_max =  self.D_full
        self.d_min = -self.D_full
        self.x_min = 0
        self.x_max = self.D_full
        # Scoring / absorbing regions (closed intervals in the model)
        self.E_A_lo = (1-self.alpha) * self.D_full
        self.E_A_hi = self.D_full
        self.E_B_lo = 0
        self.E_B_hi = self.alpha * self.D_full

    def in_E_A(self, x: float) -> bool:
        return self.E_A_lo <= x <= self.E_A_hi

    def in_E_B(self, x: float) -> bool:
        return self.E_B_lo <= x <= self.E_B_hi

    def clamp_field(self, x: float) -> float:
        return float(np.clip(x, self.x_min, self.x_max))

    def position_after_throw(self, x: float, d: float, attack_sign: int) -> float:
        """Disc moves along the attack direction after a completed throw."""
        return  min(max(self.E_B_hi, x + float(attack_sign) * float(d)), self.E_A_lo)