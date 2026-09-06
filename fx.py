"""
fx.py — USD 표시 자산가격을 KRW 기준으로 변환하는 헬퍼.

목적:
    run_study.py 파이프라인을 원화 기준으로 재실행해서
    달러 기준 성과(CAGR 9.9% / MDD -14.5% / Sharpe 0.877)와
    원화 기준 실제 체감 성과의 괴리를 측정한다.

핵심 원칙:
    1) 자산 가격만 환산한다. 현금은 원화이므로 환노출이 없다.
    2) 환율 시계열은 가격 인덱스에 맞춰 reindex + ffill 한다.
    3) 체결 지연(lag) 적용 날짜와 환율 날짜를 반드시 일치시킨다.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# 1. 환율 시계열 로드
# ---------------------------------------------------------------------------

def load_fx_yfinance() -> pd.Series:
    """
    yfinance에서 USD/KRW를 가져온다. 간편하지만 이력이 2003년 전후로 짧고
    간헐적 결측이 있다. 빠른 확인용.
    """
    import yfinance as yf

    df = yf.download("KRW=X", period="max", auto_adjust=False,
                     progress=False, threads=False)
    fx = df["Close"]
    if isinstance(fx, pd.DataFrame):          # yfinance가 MultiIndex를 줄 때
        fx = fx.iloc[:, 0]
    fx.name = "USDKRW"
    return fx.dropna()


def load_fx_fred() -> pd.Series:
    """
    FRED DEXKOUS (Korea/US Foreign Exchange Rate). 1981년부터 제공되므로
    장기 백테스트에는 이쪽이 낫다. 미 연준 공휴일에는 결측(.)이 들어온다.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS"
    df = pd.read_csv(url, parse_dates=[0], index_col=0)
    fx = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    fx.name = "USDKRW"
    return fx


def align_fx(fx: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """
    환율을 가격 인덱스(미국 거래일)에 맞춘다.

    reindex 후 ffill: 환율 데이터가 없는 날은 직전 값을 쓴다.
    bfill은 앞부분 결측 처리용이며, 이 값이 실제 신호에 쓰이는 구간에
    걸치면 안 되므로 워밍업 이후 구간만 사용할 것.
    """
    return fx.reindex(index).ffill().bfill()


# ---------------------------------------------------------------------------
# 2. 가격 변환
# ---------------------------------------------------------------------------

def to_krw(prices_usd: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
    """
    달러 표시 가격 -> 원화 표시 가격.

    prices_usd 는 auto_adjust=True 로 받은 수정주가여야 한다.
    배당 조정은 곱셈이므로 환율을 곱해도 총수익 성질이 유지된다.

    주의: 여기서 나온 원화 가격으로 MA/이격도를 다시 계산하면
    그건 '다른 전략'이 된다. 아래 3번 항목 참조.
    """
    fx_aligned = align_fx(fx, prices_usd.index)
    return prices_usd.mul(fx_aligned, axis=0)


# ---------------------------------------------------------------------------
# 3. 포트폴리오 합성 — 현금 처리가 핵심
# ---------------------------------------------------------------------------

def portfolio_returns_krw(
    weights: pd.DataFrame,          # 자산별 목표비중 (lag 이미 반영된 상태)
    prices_krw: pd.DataFrame,       # to_krw() 결과
    krw_cash_rate_annual: float = 0.0,
) -> pd.Series:
    """
    원화 기준 포트폴리오 일간 수익률.

    현금 비중 = 1 - weights.sum(axis=1)
    현금은 원화이므로 환율 영향을 받지 않는다. 이 부분이
    '자산곡선 통째로 환산' 방식과 결정적으로 갈리는 지점이다.
    """
    asset_ret = prices_krw.pct_change().fillna(0.0)

    w = weights.reindex(asset_ret.index).ffill().fillna(0.0)
    w = w[asset_ret.columns]                       # 컬럼 순서 정렬

    cash_w = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    cash_daily = krw_cash_rate_annual / 252.0

    return (w * asset_ret).sum(axis=1) + cash_w * cash_daily


# ---------------------------------------------------------------------------
# 4. 두 가지 실험 변형
# ---------------------------------------------------------------------------
#
# 변형 A — 신호는 달러 가격, 손익은 원화  ★ 이걸 먼저 하세요
#   의미: "동일 전략을 원화 투자자가 운용하면 무엇을 겪는가"
#   구현: 기존 신호 로직 그대로 두고, 손익 계산만 prices_krw 로 교체.
#   이건 새 전략이 아니라 기존 전략의 정직한 측정이다.
#
# 변형 B — 신호도 원화 가격으로 계산
#   의미: 환율 모멘텀이 신호에 섞여 들어간다.
#   이건 명백히 '다른 전략'이므로 사전 합격기준을 세우고
#   별도 실험으로 다뤄야 한다. 변형 A의 결과를 보기 전에는 하지 말 것.
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. 비교 리포트
# ---------------------------------------------------------------------------

def summarize(returns: pd.Series, label: str) -> dict:
    """CAGR / MDD / Sharpe 계산."""
    curve = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    cagr = curve.iloc[-1] ** (1.0 / years) - 1.0
    mdd = (curve / curve.cummax() - 1.0).min()
    vol = returns.std() * (252 ** 0.5)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else float("nan")
    return {"label": label, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Vol": vol}


def compare(usd_returns: pd.Series, krw_returns: pd.Series) -> pd.DataFrame:
    rows = [summarize(usd_returns, "USD 기준"), summarize(krw_returns, "KRW 기준")]
    return pd.DataFrame(rows).set_index("label")


def subperiod_table(
    usd_returns: pd.Series,
    krw_returns: pd.Series,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """
    구간별 비교. 환율 완충 효과가 실제로 있었는지 확인하는 용도.

    periods 예시:
        {
            "2008 금융위기":   ("2007-10-01", "2009-03-31"),
            "2020 코로나":     ("2020-02-01", "2020-04-30"),
            "2022 동반하락":   ("2022-01-01", "2022-10-31"),
            "2009-10 원화강세": ("2009-03-01", "2010-12-31"),
            "2026 원화강세":   ("2026-06-01", "2026-09-05"),
        }
    """
    out = []
    for name, (s, e) in periods.items():
        u, k = usd_returns.loc[s:e], krw_returns.loc[s:e]
        if len(u) == 0:
            continue
        uc, kc = (1 + u).cumprod(), (1 + k).cumprod()
        out.append({
            "구간": name,
            "USD 수익": uc.iloc[-1] - 1,
            "KRW 수익": kc.iloc[-1] - 1,
            "USD MDD": (uc / uc.cummax() - 1).min(),
            "KRW MDD": (kc / kc.cummax() - 1).min(),
        })
    return pd.DataFrame(out).set_index("구간")
