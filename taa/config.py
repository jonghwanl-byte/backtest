"""전략/백테스트 설정."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

# ----------------------------------------------------------------------------
# 기본 유니버스
# ----------------------------------------------------------------------------
TICKERS: Tuple[str, ...] = ("QQQ", "TLT", "GLD")

TICKER_NAMES = {
    "QQQ": "QQQ (나스닥 100)",
    "TLT": "TLT (미국 장기채)",
    "GLD": "GLD (실물 금)",
}

# 각 ETF 실제 상장일 (표본 외 검증 가능 구간 판단용)
INCEPTION = {"QQQ": "1999-03-10", "TLT": "2002-07-30", "GLD": "2004-11-18"}

CASH_PROXY = "^IRX"  # 13주 T-Bill 할인율(연율 %). 현금 파킹 수익률 대용.


@dataclass(frozen=True)
class StrategyConfig:
    """전략 파라미터. 이 조합 하나가 '케이스' 하나에 해당."""

    base_weights: Dict[str, float] = field(
        default_factory=lambda: {"QQQ": 0.80, "TLT": 0.10, "GLD": 0.10}
    )
    # (상단배수, 하단배수). 1.030 = MA 대비 +3.0% 돌파 시 ON
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "QQQ": (1.030, 0.980),
            "TLT": (1.030, 0.980),
            "GLD": (1.025, 0.970),
        }
    )
    ma_windows: Tuple[int, ...] = (20, 120, 200)
    # ON 개수 -> 목표비중 투입 스케일
    scalar_map: Dict[int, float] = field(
        default_factory=lambda: {3: 1.00, 2: 0.50, 1: 0.25, 0: 0.00}
    )

    def label(self) -> str:
        parts = []
        for t in TICKERS:
            if self.base_weights.get(t, 0) > 0:
                up, dn = self.bands[t]
                parts.append(f"{t}{self.base_weights[t]:.0%}[{up - 1:+.1%}/{dn - 1:+.1%}]")
        return " ".join(parts)


@dataclass(frozen=True)
class ExecConfig:
    """체결/비용/세금 가정. 여기가 백테스트 신뢰도를 좌우한다."""

    # ---- 체결 지연 (룩어헤드 방지) ----
    # 0 = T일 종가 신호를 T일 종가에 체결 (룩어헤드! 비교용으로만 사용)
    # 1 = T일 종가 신호를 T+1일 종가에 체결 (현실적 기본값)
    exec_lag: int = 1

    # ---- 거래 비용 ----
    cost_bps: float = 7.0          # 편도 비용(스프레드+수수료), bp
    slippage_bps: float = 3.0      # 추가 슬리피지, bp

    # ---- 리밸런싱 규칙 ----
    # "signal"    : 목표비중이 바뀔 때만 매매 (+ 드리프트 허용치 초과 시)
    # "daily"     : 매일 목표비중으로 되돌림 (원본 백테스트 가정, 비용 과소평가 주의)
    rebalance_mode: str = "signal"
    drift_tolerance: float = 0.03  # 개별 자산 비중이 목표 대비 3%p 벗어나면 리밸런싱

    # ---- 현금 ----
    use_real_cash_rate: bool = True   # False면 flat_cash_rate 사용
    flat_cash_rate: float = 0.02      # 원본 README 가정 (연 2.0%)
    cash_spread: float = 0.0025       # T-Bill 대비 실제 파킹 수익률 차감분

    # ---- 세금 (국내 거주자 해외주식 양도소득세) ----
    apply_tax: bool = True
    tax_rate: float = 0.22
    annual_deduction_usd: float = 1900.0  # 기본공제 250만원 / FX 1300 가정
    tax_payment_month: int = 5            # 다음해 5월 확정신고 납부

    # ---- 계좌 ----
    initial_capital: float = 100_000.0
    fractional_shares: bool = True   # False면 정수 주 단위 (소액계좌 현실성 검증)


@dataclass(frozen=True)
class GridSpec:
    """밴드 격자. 케이스 수 = len(qqq) * len(tlt) * len(gld) * len(weight_scenarios)."""

    up_grid: Sequence[float] = (0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050)
    dn_grid: Sequence[float] = (0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)

    def band_pairs(self) -> list[Tuple[float, float]]:
        return [(1 + u, 1 - d) for u in self.up_grid for d in self.dn_grid]

    def n_bands(self) -> int:
        return len(self.up_grid) * len(self.dn_grid)


# 비중 시나리오 (README에 있던 것 + 위성자산 기여도 검증용)
WEIGHT_SCENARIOS: Dict[str, Dict[str, float]] = {
    "50/25/25": {"QQQ": 0.50, "TLT": 0.25, "GLD": 0.25},
    "60/20/20": {"QQQ": 0.60, "TLT": 0.20, "GLD": 0.20},
    "70/15/15": {"QQQ": 0.70, "TLT": 0.15, "GLD": 0.15},
    "80/10/10": {"QQQ": 0.80, "TLT": 0.10, "GLD": 0.10},
    "100/0/0": {"QQQ": 1.00, "TLT": 0.00, "GLD": 0.00},
}

# 스케일링 룰 변형 (신형 vs 구형 vs 선형 vs 이진)
SCALAR_RULES: Dict[str, Dict[int, float]] = {
    "100/50/25/0": {3: 1.00, 2: 0.50, 1: 0.25, 0: 0.00},
    "100/75/50/0": {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00},
    "100/67/33/0": {3: 1.00, 2: 2 / 3, 1: 1 / 3, 0: 0.00},
    "binary": {3: 1.00, 2: 1.00, 1: 0.00, 0: 0.00},  # 과반 룰
}
