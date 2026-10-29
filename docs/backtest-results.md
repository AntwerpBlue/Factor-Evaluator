# 回测结果说明

本文保留原始《因子测试接口使用说明》中的结果解释，并按当前仓库实现校正接口与计算口径。原始 Word 和 socket 客户端已移除；`example/` 保留原设计的参考图和结果，不能据此推断当前版本支持所有旧参数。

## 输入与时间对齐

截面输入是长表：`date` 为 YYYYMMDD 字符串，`code`、`header` 或 `asset` 为证券代码，其他列为因子值。收益表使用 `return`，目标权重使用 `weight`，涨跌停标记使用 `limit_label`（涨停 1、跌停 -1、未触及 0）。当前代码按六位数字证券代码处理，适用于 A 股数据约定。

日期 T 对应在 T 形成持仓后获得的前瞻收益，例如 T 收盘到下一交易日收盘。标准配置模式和直接因子模式默认都不移动日期。若信号需要推迟一个交易日使用，直接因子模式显式传入 `--signal-lag 1 --trading-days 日历文件`；无下一交易日的信号会被丢弃。收益数据必须事先符合时间约定，入口不会重新计算收益。

未提供基准时，直接因子模式使用零基准，此时 LB 和 PowerRank LB 是绝对收益；配置模式使用指定基准的收益文件。股票池用于覆盖率分母；当前单因子核心没有据此过滤全部因子持仓，如需限定交易股票池，应先在输入中完成筛选。

## 单因子文件

配置模式的结果写入 `[output].root` 下的 `single_factor_folder`：

| 文件 | 内容 |
|---|---|
| `factor_metrics_summary.xlsx` | 各类指标工作表；关闭相应分析或缺少风格输入时，部分表不会生成 |
| `{factor}_base.parquet` | 因子的 LB、LS、RankIC、回归收益、PowerRank、换手、成本及涨跌停贡献日序列 |
| `result_all.parquet` | 带 `factor` 列的合并日序列 |
| `factor.svg` | 4×4 报告画布；部分面板在缺少相应数据时为空 |

直接因子模式沿用本地入口的逐因子输出：`{factor}_daily.parquet`、`{factor}_daily.csv`、`{factor}_summary.json`、`{factor}.svg`。`--json` 另外输出 `factor-evaluation/v1` 结构，`--plot` 另外指定 SVG 路径；多因子时自动在这些指定文件名中追加因子名。它不自动输出 Excel；需要完整工作簿时使用配置模式。

## 单因子指标口径

设每日资产前瞻收益为 r，基准收益为 b，年化周期数为 A（单因子默认 252）。因子从低到高分桶，最大编号为多头桶，最小编号为空头桶；相同信号取平均排名，可能出现空桶。

| 指标 | 当前实现 |
|---|---|
| LB | 最高桶等权收益减基准收益 |
| LS | 最高桶等权收益减最低桶等权收益 |
| PowerRank LB | 用排名幂次权重计算多头收益，再减基准；`topk` 控制选取的头部比例，`power` 控制集中程度 |
| 因子回归收益 | 每日截面带截距的一元回归斜率，即中心化因子与中心化收益的内积除以因子平方和；不是多因子中性化后的纯 Alpha |
| RankIC / 日序列 `ic` | 每日因子与前瞻收益的 Spearman 相关系数；`factor-evaluation/v1` 还给出 Pearson IC |
| ICIR | 日度 IC 均值除以样本标准差，未乘年化平方根 |
| IC decay | 因子与后续第 lag 个观测收益的 RankIC 均值；lag=1 是当前输入的前瞻收益，不是累计 lag 日收益 |
| 年化收益 | 日收益均值 × A，非复合年化收益 |
| 波动率 / Sharpe | 日收益样本标准差 × √A；Sharpe 为年化收益除以该波动率，没有额外减无风险利率 |
| 累计收益 / 年度收益 | 日收益累加曲线 / 对应年份的收益求和 |
| 单因子最大回撤 | 基于逐日复利净值计算，与累计收益面板的简单累加口径不同 |
| 换手 `*_tvr` | 0.5 × 权重绝对变化之和；单因子按相邻目标权重计算，首日从零持仓起算 |
| 交易成本 `*_cost` | 2 × 单边换手 × `cost_bps` / 10000；`*_net` 为毛收益减成本 |

PowerRank 使用分位排名 q=(rank−1)/(N−1)：先计算 `max((q−(1−topk))/topk, 0)^power`，再归一化为权重。原设计默认 `topk=1.0`、`power=2.0`；`topk=0.5` 表示只取头部一半。

注意：当前 LS 毛收益使用“多头收益减空头收益”，而其换手和涨跌停归因使用多头 +0.5、空头 -0.5 的权重。两者存在尺度差异，阅读归因和费后表现时需据此解释。本次文档迁移没有改变该计算实现。

### 涨跌停贡献与图例

LB、LS 和 PowerRank 分别按自己的权重，对 `limit_label != 0` 的持仓计算加权前瞻收益贡献。标记齐全且没有涨跌停持仓时为 0；未传标记，或任一非零权重持仓缺少标记时为未知，图例显示 `n/a`。这不是涨跌停无法成交的撮合模拟，也不代表剔除涨跌停后的替代回测。

收益曲线图例依次显示：年化收益、Sharpe、年化涨跌停贡献、双边平均换手（2 × 平均单边换手）。正费率时增加费后曲线。元数据记录标记是否提供、覆盖率及已知标记中的涨跌停比例。

### 基础特征与风格

| 工作表 | 内容 |
|---|---|
| `01_longshort_summary` | LB、LS、PowerRank 的毛/净收益、Sharpe、成本、回撤、换手及涨跌停贡献 |
| `02_lb_by_year` | 每年 LB 和 PowerRank LB 的收益求和 |
| `03_coverage` | 每日非空因子数及唯一值数相对股票池数量的比例，再取日均值；没有股票池时使用收益表资产数 |
| `04_distribution` | 有限因子值的均值、标准差、偏度、超额峰度、极值、10%/90% 分位数及计数 |
| `05_factor_return_summary` | 因子回归收益的年化收益、波动率和 Sharpe |
| `06_bucket_return` | 各桶等权超额收益的年化均值 |
| `07_ic_summary` | RankIC 均值和 ICIR |
| `08_ic_decay` | 各 lag 的 RankIC 衰减 |
| `09_style_correlation` | 因子与风格的截面相关、回归收益相关、多头序列相关 |
| `10_autocorrelation` | 同一证券因子与滞后值的 Pearson 相关，合并所有匹配观测计算；lag 按证券观测行移动 |
| `11_...` 起的风格暴露表 | 各桶的平均风格暴露；有更多风格时可能超过 14 张表 |

报告的四个风格分桶面板优先展示 `size`、`mid_cap`、`liquidity`、`residual_volatility`。风格分析仅在提供对应暴露数据时可用。`lbret_corr` 的当前实现涉及 PowerRank 权重加权的风格暴露序列，不应直接当成另一个完整风格策略的收益相关性。

稳定 JSON 接口的 `data_quality`、`intrinsic`、`stability`、`correlations`、`daily_metrics` 和 `metadata` 适用于机器读取。其中数据质量按有效、有限观测统计，不能与 Excel 覆盖率的非空计数直接混用。

## 组合结果

配置模式的 `portfolio` 任务读取已经准备好的目标权重，不自动获取 Wind 数据或优化目标。文件写入 `portfolio_folder`：

- `{benchmark}_intraday_port.parquet`、`_trade.parquet`：持仓和交易权重。
- `{benchmark}_intraday_pnl_daily.parquet`、`_trading_pnl_by_time.parquet`：日损益和按时段分配的交易损益。
- `{benchmark}_intraday_alpha_cum.png`、`_alpha_yearly_sum.png`：累计及年度超额收益。
- `{test_name}_{benchmark}_decomp.csv`、`_decomp.png`：存在归因结果时输出贡献明细与总超额、风格、行业、费前/费后纯 Alpha 五条曲线。
- `intraday_summary.json`、`factor_config.json`、`DONE.ok`：指标、运行参数和完成标记。

`holding_alpha = 持仓收益 − 基准收益`；`trading_pnl` 当前主要是交易费用的负值；`alpha = holding_alpha + trading_pnl`。时段交易损益按 `slot_weights` 分摊，不是使用逐笔成交价模拟的盘中执行结果。`pf_by_time` 是配置记录，不能当成模型实测的执行效率。

ROI、VOL、Sharpe 分别采用算术年化收益、年化波动率及其比值；组合默认 A=240，可通过配置修改。AnnualTVR 是各时段平均单边换手之和 × A。组合最大回撤按累计简单收益曲线的峰谷差计算；Calmar 为 ROI / |最大回撤|。这与单因子回撤的复利口径不同。

风格和行业归因使用传入的因子收益，或在具备暴露等输入时按当前实现估计；归因依赖数据完整性。报告中的 `max_deviation`、`max_holding`、`target_volatility` 等记录不能单凭图标题解释为执行回测一定强制实施了约束，目标权重构建与回测属于不同步骤。

## 原始设计与当前版本的对应关系

| 原设计 | 当前处理 |
|---|---|
| socket 发送至 meta1，远端返回报告 | 当前本地 Python 计算，不再保留服务器连接客户端 |
| `TRADE_DT` / `WINDCODE` 输入别名 | 标准入口使用 `date` 和 `code` / `header` / `asset`；Wind 字段由可选适配流程转换 |
| `start_date`、`end_date` | 配置模式的单因子区间，直接模式对应 `--start`、`--end` |
| `out_of_sample_start`、`SINGLE_INTERVAL=False` | 配置类保留字段，但当前没有按这两个参数生成完整样本内外对照；需分别指定日期区间运行 |
| `EXPORT_SVG` / `EXPORT_EXCEL` / `EXPORT_PARQUET` | 配置模式统一导出完整报告；直接模式使用 `--output`、`--json`、`--plot` |
| `RET_TYPE=ret/pureret`，开盘价/VWAP 版本 | 当前由用户提供前瞻收益表；直接模式可用 `--return-column` 选择已有列，不会自动生成纯净收益 |
| `TEST_BASIC_FEATURES` / `TEST_BASIC_RETURN` / `TEST_STYLE_FEATURES` | 配置中的同名小写选项控制对应指标表计算 |
| `TOPK_K` / `TOPK_POWER` | 配置的 `topk` / `power`，或直接模式的 `--topk` / `--power` |
| `report_root`、Test/Compare 结果目录 | 配置中的 `[output]` 或直接模式 `--output`；不再按旧服务器目录命名 |

原设计参考图仍保留在 [`example/`](../example/)，当前输入协议和用法以 [README](../README.md) 为准。
