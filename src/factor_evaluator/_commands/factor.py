# -*- coding: utf-8 -*-
"""Single-factor command used by run_backtest.py."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


from factor_evaluator import (
    FactorEvaluationConfig,
    evaluate_factors,
)
from factor_evaluator.factor_report import write_factor_report
from factor_evaluator.config import load_frame
from factor_evaluator.data import normalize_panel


DEFAULT_RETURN_COLUMN = "return"


def shift_factor_to_next_trade_date(factor_frame: pl.DataFrame, trading_days_path: str | None = None) -> pl.DataFrame:
    """把因子的 date 从 T 映射到 T+1 交易日。

    因子是 T 日收盘后产生的信号，T+1 收盘才下单成交（T+1 日盘后），
    持仓收益为 T+1 收盘→T+2 收盘。forward_returns 的 date=T+1 行
    存的是 T+1 收盘→T+2 收盘的前向收益。
    用交易日历把因子 date 从 T 替换为 T+1，使其与 returns 在 T+1 日 join。
    如果某天在交易日历中没有下一个交易日（如末日），则丢弃该天的因子数据。
    """
    if not trading_days_path:
        raise ValueError("--trading-days is required for --signal-lag 1")
    path = trading_days_path
    days = sorted(set(
        line.strip().replace("-", "")
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ))
    next_day = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    factor_dates = set(factor_frame["date"].to_list())
    # 只保留交易日历中存在下一天的日期，没有下一天的直接丢弃
    mapping = {d: next_day[d] for d in factor_dates if d in next_day}
    dropped = factor_dates - set(mapping)
    if dropped:
        print(f"  ⚠ 以下因子日期无下一个交易日，已忽略: {sorted(dropped)}")
    return (
        factor_frame
        .filter(pl.col("date").is_in(list(mapping)))
        .with_columns(
            pl.col("date").replace_strict(mapping, default=pl.col("date"))
        )
    )


def neutralize_factor(
    factor_frame: pl.DataFrame,
    styles_path: str,
    factor_columns: list[str],
    style_columns: list[str],
    min_assets: int = 30,
) -> pl.DataFrame:
    """每日截面 OLS 回归，因子值对风格暴露取残差。"""
    styles = normalize_panel(load_frame(Path(styles_path).expanduser())).rename({"asset": "code"})
    missing = [c for c in style_columns if c not in styles.columns]
    if missing:
        raise ValueError(f"风格暴露文件缺少列: {missing}; 当前列: {styles.columns}")

    joined = factor_frame.join(
        styles.select("date", "code", *style_columns),
        on=["date", "code"],
        how="inner",
    )

    result_parts: list[pl.DataFrame] = []
    for day in joined.partition_by("date", maintain_order=True):
        clean = day.drop_nulls(factor_columns + style_columns)
        if clean.height < min_assets:
            continue
        x = np.column_stack([
            np.ones(clean.height),
            *[clean[s].to_numpy().astype(float) for s in style_columns],
        ])
        residual_dict: dict[str, np.ndarray] = {}
        for f in factor_columns:
            y = clean[f].to_numpy().astype(float)
            finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
            if finite.sum() < min_assets:
                residual_dict[f] = np.full(clean.height, np.nan)
                continue
            x_f = x[finite]
            y_f = y[finite]
            beta = np.linalg.lstsq(x_f, y_f, rcond=None)[0]
            residual = np.full(clean.height, np.nan)
            residual[finite] = y_f - x_f @ beta
            residual_dict[f] = residual
        result_parts.append(
            clean.select("date", "code").with_columns(
                pl.Series(f, residual_dict[f]) for f in factor_columns
            )
        )

    if not result_parts:
        raise ValueError("中性化后无有效数据，检查 min_assets 和 join 覆盖率")
    return pl.concat(result_parts).sort(["date", "code"])


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="单因子回测入口（基于 Factor-Evaluator）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--factor", required=True, type=str,
        help="因子值文件路径 (parquet/csv/ipc/feather)，长表: date, code, <factor_column>",
    )
    parser.add_argument(
        "--factor-columns", default=None, type=str,
        help="因子列名，逗号分隔（如 alpha,beta）；不传则自动取第三列起所有非 date/code 列",
    )
    parser.add_argument(
        "--forward-returns", required=True, type=str,
        help="前向收益文件，date 对应持仓形成日期",
    )
    parser.add_argument(
        "--return-column", default=DEFAULT_RETURN_COLUMN, type=str,
        help=f"使用的收益率列名 (默认: {DEFAULT_RETURN_COLUMN}；可选 ret_open/ret_close)",
    )
    parser.add_argument(
        "--benchmark", default=None, type=str,
        help="基准日收益文件 (date, benchmark_return)；不传则零基准",
    )
    parser.add_argument(
        "--universe", default=None, type=str,
        help="universe 文件 (date, code, eligible布尔列)；不传则用 returns 表当 universe",
    )
    parser.add_argument(
        "--styles", default=None, type=str,
        help="Barra 风格因子暴露文件 (date, code, <style列>); 设为空字符串跳过风格分析",
    )
    parser.add_argument(
        "--neutralize", action="store_true",
        help="回测前对因子做风格中性化（截面回归取残差）",
    )
    parser.add_argument(
        "--neutralize-styles", default="size,mid_cap,liquidity,residual_volatility", type=str,
        help="中性化剥离的风格列，逗号分隔 (默认: size,mid_cap,liquidity,residual_volatility)",
    )
    parser.add_argument("--start", default=None, type=str, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, type=str, help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--buckets", default=30, type=int,
        help="分桶数 (默认 30)",
    )
    parser.add_argument(
        "--cost-bps", default=8.0, type=float,
        help="单边交易费率 (bps)，日扣费 = 2 * turnover * cost_bps/10000 (默认 8)",
    )
    parser.add_argument(
        "--power", default=2.0, type=float,
        help="topk_power 组合加权的幂指数 (默认 2.0；=1.0 即全市场 rank 线性加权)",
    )
    parser.add_argument(
        "--min-assets", default=20, type=int,
        help="单日最少资产数，少于此数则跳过该日 (默认 20)",
    )
    parser.add_argument(
        "--annualization", default=252, type=int,
        help="年化天数 (默认 252)",
    )
    parser.add_argument(
        "--trading-days", default=None, type=str,
        help="交易日历文件，每行一个 YYYYMMDD；仅信号顺延时需要",
    )
    parser.add_argument(
        "--no-shift", action="store_true",
        help="兼容旧调用：不顺延信号日期",
    )
    parser.add_argument(
        "--output", default=None, type=str,
        help="输出目录，逐因子写入日序列 Parquet/CSV、汇总 JSON 和 SVG",
    )
    parser.add_argument(
        "--json", default=None, type=str,
        help="输出 JSON 文件路径 (evaluate_factors 稳定契约)",
    )
    parser.add_argument(
        "--plot", default=None, type=str,
        help="输出 SVG 报告图路径",
    )
    parser.add_argument("--signal-lag", type=int, choices=(0, 1), default=0,
                        help="信号顺延交易日数，默认 0；1 表示 T→T+1")
    parser.add_argument("--limit-labels", help="涨跌停标记长表")
    parser.add_argument("--topk", type=float, default=1.0, help="PowerRank 选股比例")
    args = parser.parse_args(argv)
    if args.no_shift and args.signal_lag:
        parser.error("--no-shift cannot be combined with --signal-lag 1")
    if args.signal_lag and not args.trading_days:
        parser.error("--signal-lag 1 requires --trading-days")
    if args.neutralize and not args.styles:
        parser.error("--neutralize requires --styles")

    # ---- 1. 识别因子列 ----
    factor_path = Path(args.factor).expanduser()
    factor_frame = load_frame(factor_path)
    raw_cols = factor_frame.columns
    asset_aliases = ("asset", "code", "header", "Code")
    asset_col = next((c for c in asset_aliases if c in raw_cols), None)
    if asset_col is None:
        raise ValueError(f"因子文件必须包含 {asset_aliases} 之一; 当前列: {raw_cols}")
    if "date" not in raw_cols and "Date" not in raw_cols:
        raise ValueError(f"因子文件必须包含 date 列; 当前列: {raw_cols}")
    if args.factor_columns:
        factor_columns = [c.strip() for c in args.factor_columns.split(",") if c.strip()]
    else:
        factor_columns = [c for c in raw_cols if c not in {"date", "Date", asset_col}]
    if not factor_columns:
        raise ValueError(f"未找到因子列; 文件列: {raw_cols}")

    # ---- 2. 读入因子数据并 shift T→T+1 ----
    # 兼容首字母大写的列名 Date/Code
    rename_map = {}
    if "Date" in factor_frame.columns and "date" not in factor_frame.columns:
        rename_map["Date"] = "date"
    if "Code" in factor_frame.columns and "code" not in factor_frame.columns:
        rename_map["Code"] = "code"
    if rename_map:
        factor_frame = factor_frame.rename(rename_map)
        # 同步更新 asset_col 引用
        if "code" in rename_map.values():
            asset_col = "code"
    factor_frame = normalize_panel(factor_frame).rename({"asset": "code"})
    if args.signal_lag:
        factor_frame = shift_factor_to_next_trade_date(factor_frame, args.trading_days)
        print(f"  因子日期已 shift T→T+1 交易日")
    else:
        print("  因子日期不做 shift")

    # ---- 2b. 可选：风格中性化 ----
    if args.neutralize:
        neutralize_cols = [c.strip() for c in args.neutralize_styles.split(",") if c.strip()]
        print(f"  对因子做风格中性化，剥离: {neutralize_cols}")
        factor_frame = neutralize_factor(
            factor_frame,
            args.styles,
            factor_columns,
            neutralize_cols,
            min_assets=max(args.min_assets, 30),
        )
        print(f"  中性化完成，剩余 {factor_frame.height:,} 行")

    # ---- 3. 公共配置 ----
    eval_config = FactorEvaluationConfig(
        start_date=args.start,
        end_date=args.end,
        min_assets=args.min_assets,
        buckets=args.buckets,
        annualization=args.annualization,
        cost_bps=args.cost_bps,
        power=args.power,
        topk=args.topk,
    )

    # ---- 3b. 加载前向收益并选取指定列 ----
    returns_frame = load_frame(Path(args.forward_returns).expanduser())

    # evaluate_factors 期望 forward_returns 有 return 列
    if args.return_column and args.return_column in returns_frame.columns:
        if "return" in returns_frame.columns and args.return_column != "return":
            returns_frame = returns_frame.drop("return")
        returns_frame = returns_frame.rename({args.return_column: "return"})
    else:
        raise ValueError(f"Return column not found: {args.return_column}")
    print(f"  前向收益列:     {args.return_column} -> return")

    print("=" * 70)
    print("单因子回测")
    print("=" * 70)
    print(f"  因子文件:       {args.factor}")
    print(f"  因子列:         {factor_columns}")
    print(f"  前向收益:       {args.forward_returns} ({args.return_column})")
    print(f"  基准:           {args.benchmark or '<零基准>'}")
    print(f"  universe:       {args.universe or '<用 returns 表>'}")
    print(f"  风格暴露:       {args.styles or '<无>'}")
    print(f"  中性化:         {'是 (' + args.neutralize_styles + ')' if args.neutralize else '否'}")
    print(f"  日期范围:       {args.start or '<全部>'} ~ {args.end or '<全部>'}")
    print(f"  分桶数:         {args.buckets}")
    print(f"  费率:           {args.cost_bps} bps (单边)")
    print()

    out_dir = Path(args.output).expanduser() if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for factor in factor_columns:
        if any(char in factor for char in '/\\\\') or factor in {".", ".."}:
            raise ValueError(f"Factor name is not a safe output filename: {factor}")
        print("#" * 70)
        print(f"# 因子: {factor}")
        print("#" * 70)

        result = evaluate_factors(
            factors=factor_frame,
            forward_returns=returns_frame,
            factor_names=[factor],
            benchmark=args.benchmark,
            universe=args.universe,
            styles=args.styles or None,
            limit_labels=args.limit_labels,
            config=eval_config,
        )

        def result_path(value: str) -> Path:
            path = Path(value).expanduser()
            if len(factor_columns) > 1:
                path = path.with_name(f"{path.stem}_{factor}{path.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        if args.json:
            result.write_json(result_path(args.json))
        if args.plot:
            write_factor_report(result.backtest_result, result_path(args.plot))

        # ---- 控制台摘要 ----
        intrinsic = result.intrinsic[factor]
        stability = result.stability[factor]
        dq = result.data_quality[factor]

        print("-" * 70)
        print("数据质量")
        print("-" * 70)
        print(f"  期望观测数:     {dq['expected_observations']:,}")
        print(f"  有效观测数:     {dq['valid_observations']:,}")
        print(f"  缺失比例:       {dq['missing_ratio']*100:.2f}%")
        print(f"  日均覆盖率:     {dq['mean_daily_coverage']*100:.2f}%")
        print(f"  唯一值比例:     {dq['unique_value_ratio']*100:.2f}%")
        if dq["warnings"]:
            for w in dq["warnings"]:
                print(f"  ⚠ {w}")

        print()
        print("-" * 70)
        print("IC 指标")
        print("-" * 70)
        print(f"  Pearson IC:     {intrinsic['pearson_ic']:.4f}")
        print(f"  Pearson ICIR:   {intrinsic['pearson_icir']:.4f}")
        print(f"  Rank IC:        {intrinsic['rank_ic']:.4f}")
        print(f"  Rank ICIR:      {intrinsic['rank_icir']:.4f}")
        print(f"  样本周期数:     {intrinsic['n_periods']}")

        print()
        print("-" * 70)
        print("IC 稳定性")
        print("-" * 70)
        print(f"  Pearson IC std:        {stability['pearson_ic_std']:.4f}")
        print(f"  Rank IC std:           {stability['rank_ic_std']:.4f}")
        print(f"  Pearson IC 正比:       {stability['pearson_ic_positive_ratio']*100:.2f}%")
        print(f"  Rank IC 正比:          {stability['rank_ic_positive_ratio']*100:.2f}%")
        if stability["factor_autocorrelation"]:
            acf1 = stability["factor_autocorrelation"][0]
            print(f"  因子自相关 (lag={acf1['lag']}):    {acf1['autocorrelation']:.4f}")

        # ---- 回测指标 ----
        bt = result.backtest_result
        ls_summary = bt.tables.get("01_longshort_summary")
        bucket_returns = bt.tables.get("06_bucket_return")
        daily = bt.daily.get(factor)

        print()
        print("-" * 70)
        print("回测指标（equal_weight 策略）")
        print("-" * 70)
        if ls_summary is not None and not ls_summary.is_empty():
            ew = ls_summary.filter(
                (pl.col("strategy") == "equal_weight") if "strategy" in ls_summary.columns else pl.lit(True)
            )
            if not ew.is_empty():
                for row in ew.iter_rows(named=True):
                    pnl = row.get("pnl_type", "?")
                    ar = row.get("annual_return", float("nan"))
                    ar_net = row.get("annual_return_net", float("nan"))
                    sh = row.get("sharpe", float("nan"))
                    sh_net = row.get("sharpe_net", float("nan"))
                    md = row.get("max_drawdown", float("nan"))
                    md_net = row.get("max_drawdown_net", float("nan"))
                    tv = row.get("avg_turnover", float("nan"))
                    ac = row.get("avg_cost", float("nan"))
                    def _pct(v):
                        return f"{v*100:>7.2f}%" if v == v else "     nan"
                    def _num(v):
                        return f"{v:>6.2f}" if v == v else "   nan"
                    print(f"  [{pnl}] 年化: {_pct(ar)} (净{_pct(ar_net)})  "
                          f"Sharpe: {_num(sh)} (净{_num(sh_net)})  "
                          f"回撤: {_pct(md)} (净{_pct(md_net)})  "
                          f"换手: {_pct(tv)}  日均成本: {_pct(ac)}")

        if bucket_returns is not None and not bucket_returns.is_empty():
            print()
            print("-" * 70)
            print("分桶年化超额收益")
            print("-" * 70)
            for row in bucket_returns.iter_rows(named=True):
                b = row.get("bucket", "?")
                r = row.get("annual_excess_return", float("nan"))
                r_s = f"{r*100:>7.2f}%" if r == r else "     nan"
                print(f"  桶 {b}: {r_s}")

        style_corr = bt.tables.get("09_style_correlation")
        if style_corr is not None and not style_corr.is_empty():
            print()
            print("-" * 70)
            print("风格因子相关性")
            print("-" * 70)
            for row in style_corr.iter_rows(named=True):
                style = row.get("style", "?")
                fret_corr = row.get("fret_corr", float("nan"))
                lbret_corr = row.get("lbret_corr", float("nan"))
                fc = f"{fret_corr:>7.4f}" if fret_corr == fret_corr else "   nan"
                lc = f"{lbret_corr:>7.4f}" if lbret_corr == lbret_corr else "   nan"
                print(f"  {style:30s}  因子收益相关: {fc}  多头收益相关: {lc}")

        # ---- 输出文件（按因子命名） ----
        if out_dir:
            # daily parquet + csv
            if daily is not None and not daily.is_empty():
                daily_parquet = out_dir / f"{factor}_daily.parquet"
                daily.write_parquet(daily_parquet, compression="zstd")
                print(f"\n  已写入: {daily_parquet}")
                daily_csv = out_dir / f"{factor}_daily.csv"
                daily.write_csv(daily_csv)
                print(f"  已写入: {daily_csv}")

            # summary json
            summary: dict = {}
            for name, table in bt.tables.items():
                if table.is_empty() or "factor" not in table.columns:
                    summary[name] = table.to_dicts() if not table.is_empty() else []
                else:
                    subset = table.filter(pl.col("factor") == factor)
                    summary[name] = subset.to_dicts() if not subset.is_empty() else []
            summary["metadata"] = bt.metadata
            json_path = out_dir / f"{factor}_summary.json"
            json_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"  已写入: {json_path}")

            # SVG
            svg_path = out_dir / f"{factor}.svg"
            try:
                write_factor_report(bt, svg_path)
                print(f"  已写入: {svg_path}")
            except Exception as e:
                print(f"  ⚠ 报告图生成失败: {e}")

        print()

    print("=" * 70)
    print("完成")
    print("=" * 70)
