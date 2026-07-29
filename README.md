# Factor Evaluator

同时支持单因子研究与可交易组合策略（包括指数增强）的 Python 回测框架。

## 输入协议

所有截面输入均为长表：

| 字段 | 约定 |
|---|---|
| `date` | 必须是 `YYYYMMDD` 格式的字符串 |
| `code` 或 `header` | 6 位代码，或带交易所后缀的 9 位代码，如 `000001.SZ` |
| 其余列 | 因子名称及数值；收益表使用 `return`，权重表使用 `weight` |

框架内部将 `code`/`header` 统一为 `asset`。非法日期、非法代码以及重复的 `date + code` 键会直接报错。

`return` 表示在该日期形成持仓后获得的收益。如果因子在 T 日收盘可交易，T 日键上的收益必须对应 T 日收盘到下一交易时点，框架不会暗中再次 shift。

## 数据路径配置

复制 `config/backtest.example.toml` 为 `config/backtest.toml`，替换所有 `REPLACE_ME`。配置已经为以下数据预留路径：

- 因子值、股票前瞻收益、股票池和可交易状态
- 风格/行业暴露、风格/行业收益、因子协方差和特异风险
- 分时成交价、成交量、公司行为、证券主数据和交易日历
- 策略目标权重
- ZeroIndex、沪深300、中证500、中证1000和中证2000的收益与成分权重

仍含 `REPLACE_ME` 的必需路径会被拒绝执行。

```bash
python examples/run_from_config.py
```

## 单因子输出

- `factor_metrics_summary.xlsx`：14 类指标表
- `{factor}_base.parquet`：LB、LS、IC、因子收益和 PowerRank 日序列
- `result_all.parquet`：多因子合并日序列
- SVG 4×4 综合报告：累计收益、年度收益、PowerRank、因子收益、分档收益、IC decay、覆盖率、分布、自相关、风格相关和四类分档风格暴露

## 组合输出

每个基准生成：

- 持仓、交易、日度 PnL 和分时交易损益 parquet
- 累计图：`holding_alpha`、各成交时段交易损益、费后 `alpha`，以及下方面板的分时换手曲线
- ROI、VOL、Sharpe、Calmar、最大回撤及日期、AnnualTVR、分时平均换手和 PF
- 年度超额收益柱状图
- 五曲线归因图：总超额、风格、行业、费前纯 Alpha、费后纯 Alpha
- 逐风格贡献 CSV、完整 `intraday_summary.json`、`factor_config.json` 和 `DONE.ok`

## 安装与验证

```bash
pip install -e ".[dev]"
pytest -q
python examples/run_synthetic.py
```

合成数据完整报告生成在 `synthetic_report/`。

