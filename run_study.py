#!/usr/bin/env python3
"""전체 검증 스위트 실행.

사용 예
-------
    python run_study.py                          # 전체 (권장, 10~30분)
    python run_study.py --quick                  # 격자 축소 빠른 확인
    python run_study.py --stage lag,subperiod    # 특정 검증만
    python run_study.py --synthetic              # 네트워크 없이 엔진 검증
    python run_study.py --grid-mode full         # 3자산 밴드 전조합 (17만+ 케이스)

출력
----
    out/report.md         사람이 읽는 요약
    out/results.xlsx      모든 표 (시트별)
    out/*.csv             원자료
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from taa.config import (  # noqa: E402
    INCEPTION, SCALAR_RULES, TICKERS, WEIGHT_SCENARIOS,
    ExecConfig, GridSpec, StrategyConfig,
)
from taa.data import load_cash_rate, load_prices, synthetic_prices  # noqa: E402
from taa.engine import run_backtest  # noqa: E402
from taa.fastscan import GridScanner, surface  # noqa: E402
from taa.metrics import annual_table, perf_stats  # noqa: E402
from taa.periods import IS_OOS, REGIMES  # noqa: E402
from taa.study import (  # noqa: E402
    benchmark_suite, bootstrap_ci, full_grid, market_type_table,
    overfit_diagnostics, robustness_report, rolling_table, subperiod_table,
    tax_impact, test_execution_lag, vol_matched_comparison, walk_forward,
)

OUT = Path("out")
SHEETS: dict[str, pd.DataFrame] = {}
NOTES: list[str] = []


def emit(name: str, df: pd.DataFrame, title: str = "", note: str = ""):
    if df is None or (hasattr(df, "empty") and df.empty):
        return
    SHEETS[name[:31]] = df
    NOTES.append(f"\n## {title or name}\n\n{df.round(4).to_markdown(index=False)}\n")
    if note:
        NOTES.append(f"> {note}\n")
    print(f"\n=== {title or name} ===")
    print(df.round(4).to_string(index=False)[:4000])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2004-11-18", help="3자산 동시 가능 시작일(GLD 상장)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--grid-mode", default="qqq", choices=["qqq", "full", "linked"])
    ap.add_argument("--stage", default="all")
    ap.add_argument("--flat-cash", action="store_true", help="README식 flat 2% 현금 가정")
    ap.add_argument("--no-tax", action="store_true")
    args = ap.parse_args()
    stages = set(args.stage.split(",")) if args.stage != "all" else None

    def run(s):
        return stages is None or s in stages

    OUT.mkdir(exist_ok=True)

    # ---------------- 데이터 ----------------
    print("[0] 데이터 로딩")
    if args.synthetic:
        prices = synthetic_prices(list(TICKERS))
        cash = pd.Series(0.02 / 252, index=prices.index)
        print("  ** 합성 데이터 (엔진 검증용, 결과 해석 금지) **")
    else:
        prices = load_prices(TICKERS, start="1999-01-01", end=args.end)
        cash = load_cash_rate(prices.index, use_real=not args.flat_cash)
        print(f"  현금 수익률 평균: {cash.mean() * 252:.2%} (연율)")
        print(f"  상장일: {INCEPTION}")

    # 신호는 전체 히스토리로 워밍업, 성과는 start 이후만 집계
    eval_start = pd.Timestamp(args.start)
    mask_eval = np.asarray(prices.index >= eval_start)

    strat = StrategyConfig()
    ex = ExecConfig(apply_tax=not args.no_tax)
    spec = GridSpec(up_grid=(0.015, 0.025, 0.030, 0.040), dn_grid=(0.015, 0.020, 0.030, 0.040)) \
        if args.quick else GridSpec()

    sc = GridScanner(prices, cash, strat.ma_windows, ex.exec_lag,
                     (ex.cost_bps + ex.slippage_bps) / 10_000)
    base_r = sc.series_for(strat.bands, strat.base_weights, strat.scalar_map)[mask_eval]
    cash_eval = cash[mask_eval]
    bench = {k: v[mask_eval] for k, v in benchmark_suite(prices, cash, ex).items()}

    NOTES.append(f"# TAA 전략 검증 리포트\n\n"
                 f"- 평가구간: {base_r.index[0].date()} ~ {base_r.index[-1].date()} "
                 f"({len(base_r)/252:.1f}년)\n"
                 f"- 신호 워밍업: {prices.index[0].date()}부터 (경로의존 상태 수렴)\n"
                 f"- 현금: {'flat 2%' if args.flat_cash else '^IRX 실측'}\n"
                 f"- 세금: {'미반영' if args.no_tax else '해외주식 양도세 22%'}\n")

    # ---------------- 1. 룩어헤드 ----------------
    if run("lag"):
        print("\n[1] 체결지연 민감도 (룩어헤드 검증)")
        df = test_execution_lag(prices, cash, strat, ex)
        emit("1_exec_lag", df, "검증1. 체결 시점 민감도",
             "lag=-1에서만 성적이 좋다면 원본 백테스트는 룩어헤드다. "
             "lag=1(현실) 대비 샤프 차이가 0.3 이상이면 경고.")

    # ---------------- 2. 구간 분해 ----------------
    if run("subperiod"):
        print("\n[2] 구간별 분해")
        emit("2a_IS_OOS", subperiod_table(base_r, cash_eval, IS_OOS, bench["QQQ 매수보유"]),
             "검증2a. 표본내 vs 표본외",
             "README 최적화 구간(2015-2024) 밖에서 성과가 유지되는지가 핵심.")
        emit("2b_regimes", subperiod_table(base_r, cash_eval, REGIMES, bench["QQQ 매수보유"]),
             "검증2b. 매크로 레짐별")
        emit("2c_market_type", market_type_table(base_r, cash_eval, bench["QQQ 매수보유"]),
             "검증2c. 시장 국면별",
             "'완만한 약세장(휩쏘 다발)'이 이 전략의 최대 약점. 여기 숫자를 보라.")
        rt = rolling_table(base_r, cash_eval, years=3, step_months=6)
        emit("2d_rolling3y", rt, "검증2d. 롤링 3년 (중첩)",
             f"요약: {rt.attrs.get('summary')}" if not rt.empty else "")
        emit("2e_annual", annual_table(base_r, bench["QQQ 매수보유"]).reset_index()
             .rename(columns={"index": "year"}), "검증2e. 연도별")

    # ---------------- 3. 격자 + 견고성 ----------------
    grid = None
    if run("grid"):
        print(f"\n[3] 격자 스캔 (mode={args.grid_mode})")
        kw = dict(vary=("QQQ",)) if args.grid_mode == "qqq" else \
             dict(vary=tuple(TICKERS)) if args.grid_mode == "full" else dict(link_bands=True)
        grid = full_grid(prices, cash, spec, ex, strat.ma_windows,
                         WEIGHT_SCENARIOS, SCALAR_RULES, mask=mask_eval, **kw)
        grid.to_csv(OUT / "grid_full.csv", index=False)

        top = grid.nlargest(15, "Sharpe")
        emit("3a_top", top, "검증3a. 상위 15 (참고용)",
             "**1등은 보지 말 것.** 아래 분포와 이웃 점수를 보라.")

        for scen in WEIGHT_SCENARIOS:
            sub = grid.query("scenario == @scen and rule == '100/50/25/0'")
            if len(sub) < 4:
                continue
            rep = robustness_report(sub)
            emit(f"3b_dist_{scen.replace('/', '')}", rep["distribution"].to_frame("value")
                 .reset_index().rename(columns={"index": "stat"}),
                 f"검증3b. 격자 분포 [{scen}]",
                 f"판정: **{rep['verdict']}** (최적셀 {rep['best_cell']}, "
                 f"고립도 {rep['isolation']:.3f} = 최적값 − 이웃평균)")
            if args.grid_mode != "linked":
                emit(f"3c_surf_{scen.replace('/', '')}", rep["surface"].reset_index(),
                     f"검증3c. QQQ 밴드 표면 [{scen}] (행=상단, 열=하단)",
                     "전체가 비슷하면 고원(신뢰 가능), 한 칸만 튀면 첨탑(과최적화).")

        emit("3d_overfit", pd.DataFrame([overfit_diagnostics(grid, int(mask_eval.sum()))]),
             "검증3d. 과최적화 진단 (Deflated Sharpe)",
             "DSR < 0.95면 '수천 번 돌려서 1등 뽑은 것'과 구분 불가.")

        emit("3e_rules", grid.groupby("rule")["Sharpe"].describe().reset_index(),
             "검증3e. 스케일링 룰 비교",
             "샤프 차이가 작다면 '리스크관리 개선'이 아니라 단순 디레버리지.")

    # ---------------- 4. 워크포워드 ----------------
    if run("wf"):
        print("\n[4] 워크포워드 (표본외 실전 기대치)")
        wf_ret, wf_log = walk_forward(prices, cash, spec, ex, strat.base_weights,
                                      strat.scalar_map, strat.ma_windows,
                                      is_years=4, oos_years=1, vary=("QQQ",))
        if not wf_log.empty:
            emit("4a_wf_log", wf_log, "검증4a. 워크포워드 구간별",
                 f"IS→OOS 샤프 열화: **{wf_log.attrs['degradation']:.3f}**, "
                 f"OOS 승률 {wf_log.attrs['hit_rate']:.0%}")
            wf_ev = wf_ret[wf_ret.index >= eval_start]
            emit("4b_wf_perf", pd.DataFrame([perf_stats(wf_ev, cash)]).assign(전략="워크포워드 OOS"),
                 "검증4b. 워크포워드 누적 성과",
                 "**이 숫자가 실전 기대치다.** 고정 최적 밴드 성과와 비교하라.")

    # ---------------- 5. 비용·세금 ----------------
    if run("cost"):
        print("\n[5] 비용/세금 영향")
        emit("5_cost_tax", tax_impact(prices, cash, strat, ex, warmup_from=eval_start),
             "검증5. 가정별 성과 붕괴",
             "양도세 22%는 잦은 실현매매를 직격한다. 세후 CAGR을 매수보유와 비교하라.")

    # ---------------- 6. 벤치마크 (변동성 정규화) ----------------
    if run("bench"):
        print("\n[6] 벤치마크 공정비교")
        emit("6a_vol_matched", vol_matched_comparison({"전략(최적밴드)": base_r, **bench},
                                                      cash_eval, 0.10),
             "검증6a. 변동성 10% 정규화 비교",
             "복잡한 15파라미터 시스템이 'QQQ 200MA 단순필터'를 얼마나 이기는가? "
             "차이가 작으면 나머지 파라미터는 장식이다.")
        emit("6b_bootstrap", bootstrap_ci(base_r, 2000).reset_index()
             .rename(columns={"index": "metric"}),
             "검증6b. 블록 부트스트랩 신뢰구간",
             "관측된 MDD가 p5~p95 어디에 있는지 확인. 운이 좋았을 뿐일 수 있다.")

    # ---------------- 7. 실전 시뮬 ----------------
    if run("live"):
        print("\n[7] 전수 시뮬레이션 (세금·정수주 포함)")
        res = run_backtest(prices, cash, strat, ex, warmup_from=eval_start)
        stt = perf_stats(res.returns, cash, res.diagnostics)
        emit("7_live_sim", pd.DataFrame([stt]), "검증7. 실전 시뮬레이션 최종",
             f"연 매매 {res.diagnostics['trades_per_year']:.1f}회, "
             f"누적세금 ${res.diagnostics['total_tax']:,.0f}")
        res.equity.to_csv(OUT / "equity_curve.csv")
        if not res.trades.empty:
            res.trades.to_csv(OUT / "trades.csv")

    # ---------------- 출력 ----------------
    (OUT / "report.md").write_text("\n".join(NOTES), encoding="utf-8")
    try:
        with pd.ExcelWriter(OUT / "results.xlsx", engine="openpyxl") as w:
            for name, df in SHEETS.items():
                df.to_excel(w, sheet_name=name, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 엑셀 저장 실패({e}) - CSV는 정상")
        for name, df in SHEETS.items():
            df.to_csv(OUT / f"{name}.csv", index=False)

    print(f"\n완료 -> {OUT.resolve()}/report.md , results.xlsx")


if __name__ == "__main__":
    main()
