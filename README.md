# Factor Evaluator

单因子研究与组合回测工具。唯一可执行入口为根目录的 `run_backtest.py`；计算、报告和数据适配均在 `src/factor_evaluator/` 内部实现。

## 安装

需要 Python 3.11 或更新版本。

```bash
pip install -e .
python run_backtest.py --help
```

仅使用已有标准数据时，无需数据库依赖。Wind 数据适配另行安装 `pip install -e ".[wind]"`，并在系统安装对应的 SQL Server ODBC 驱动。

## 配置运行

复制 `config/backtest.example.toml` 为自己的配置，填入实际路径。相对路径相对于配置文件所在目录解析。必需路径不能留空或包含 `REPLACE_ME`；可选数据不用时设为空字符串。

```bash
python run_backtest.py --config config/backtest.toml --task factor --benchmark Zhishu500
python run_backtest.py --config config/backtest.toml --task portfolio --benchmark Zhishu500
```

`--task` 支持 `factor`、`portfolio`、`both`，默认 `factor`。组合任务读取配置的目标权重文件，不要求因子值文件，也不会自动联网取数。

## 直接传入因子文件

```bash
python run_backtest.py factor --factor data/factors.parquet --forward-returns data/returns.parquet --output reports
```

这里只描述正式入口的调用方式，不提供独立的示例程序。旧的 `python run_backtest.py --factor ...` 写法继续可用。完整选项见 `python run_backtest.py factor --help`。

- 可选 `--benchmark`、`--universe`、`--styles`、`--limit-labels`；不指定基准时使用零基准。
- `--return-column` 选择已有收益列，默认 `return`；`--factor-columns` 用逗号指定因子列。
- 默认不移动信号日期。需要 T→T+1 时显式使用 `--signal-lag 1 --trading-days 日历文件`。这与旧定制脚本默认顺延的行为不同。
- `--neutralize --styles 暴露文件` 开启每日 OLS 风格中性化，`--neutralize-styles` 指定风格列。
- `--output` 输出逐因子的日序列、汇总 JSON 和 SVG。`--json` 输出稳定评估 JSON，`--plot` 指定 SVG 路径；多因子会追加因子名，避免覆盖。

## 数据协议

| 数据 | 必需字段 |
|---|---|
| 因子长表 | `date`、证券代码、一个或多个因子值列 |
| 股票前瞻收益 | `date`、证券代码、`return` |
| 基准收益 | `date`、`benchmark_return` |
| 目标/基准权重 | `date`、证券代码、`weight` |
| 股票池 | `date`、证券代码、`eligible` |
| 涨跌停标记 | `date`、证券代码、`limit_label`，取 -1/0/1 |
| 风格/行业暴露 | `date`、证券代码、各暴露列 |

证券代码列支持 `code`、`header`、`asset`，代码使用六位数字或带交易所后缀的 A 股代码。日期为 YYYYMMDD 字符串。支持 CSV、Parquet、IPC、Feather、Arrow 文件；拒绝重复的日期与证券键。

T 日键上的 `return` 必须是 T 形成持仓后的收益。框架不会暗中 shift 收益。配置中的 `tradability` 可用于传入涨跌停归因标记；该归因不等于涨跌停成交约束。

## 结果说明

完整指标定义、输出文件、图例、原始设计参数的对应关系及实现边界见 [回测结果说明](docs/backtest-results.md)。配置模式输出 Excel、日序列 Parquet 和 SVG；组合模式输出持仓、交易、损益、归因图和汇总 JSON。

`example/` 暂时保留原设计参考图与结果。它不是可执行示例，也不作为当前实现的验收结果。

## 原有数据适配与策略流程

原 `tools/` 已移入内部模块；通过同一个入口选择命令，命令后加 `--help` 查看参数。

| 命令 | 用途及适用范围 |
|---|---|
| `prepare-data` | 将指定目录的 Wind/RQ 数据转换为标准表；依赖原字段约定，需显式指定风险暴露目录或使用 `--no-risk` |
| `fetch-wind` | 从配置的 Wind SQL Server 数据库拉取指数收益与权重；需要 wind 可选依赖 |
| `risk-model` | 从标准收益与暴露表估计风险数据；仍使用既定风格分类与默认市值字段 |
| `build-targets` | 根据因子与基准权重构建指数增强目标权重 |
| `small-account` | 保留小资金指数加股票的策略配方，不是通用资产配置接口 |
| `pipeline` | 保留原 Wind/RQ 完整流程；显式提供因子文件、数据目录和风险目录 |

这些命令保留现有数据适配行为，不是任意数据源的自动识别器。标准数据回测直接使用配置模式即可。

## 开发验证

```bash
pip install -e ".[dev]"
pytest -q
```

`tests/` 是内部验证，保留必要的测试数据构造；仓库不再提供 `examples/` 可执行示例。编辑器配置、安装产生的 egg-info、构建物和本地运行结果不应提交。

