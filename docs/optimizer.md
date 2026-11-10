# 组合优化器

唯一脚本入口仍为 `run_backtest.py build-targets`。新增单次决策模式，直接用当前信号、真实持仓和决策前状态计算目标权重；不假设 T+1 开盘、不执行订单。Python 保留 `optimize()` / `optimize_result()`，新增 `optimize_signal()`。

## 单次调用协议

`IndexEnhancementStrategy.optimize_signal` 的关键字参数：

| 参数 | 内容 |
|---|---|
| `signal` | `{证券代码: 因子值}`，或 `{"observed_at": 时间, "values": {代码: 因子值}}`；裸字典表示调用方确认信号已可用 |
| `decision_time` | 任意决策时刻，ISO 时间或 datetime；无时区按上海时间，有时区转换为上海时间 |
| `benchmark_weights` | 当前基准权重，内部归一到 1；明确配置 `zero_benchmark=True` 才使用零基准 |
| `current_weights` | 决策时点实际股票权重，不重新归一化；未列出的资产持仓为零，剩余可为现金 |
| `risk_data` | 已载入历史数据的 `RiskMatrixProvider` |
| `trade_status` | Polars 表，或 `{代码: 状态记录}`；每条必须含 `observed_at` |
| `calibration_state` | `CalibrationState`；默认 `calibrated_rank` 必须提供 |
| `asset_constraints` | 可选 `AssetConstraints`，与交易状态约束取交集 |
| `alpha_transform` | 默认 `calibrated_rank`，也支持 `none / zscore / rank_zscore` |
| `unknown_status_policy` | 默认 `reject`；调用方显式指定 `freeze` 才可冻结未知状态资产继续求解 |

信号及校准状态的可用时间不得晚于决策时间；交易状态严格选 `observed_at < decision_time` 的最新记录。允许早盘、盘中或收盘后信号，规则与信号来源时刻无关。直接调用 `predict()` 不接收时间；时间截止由适配层负责。

输出 `OptimizationResult.weights` 为 `{代码: 目标权重}`，`diagnostics` 为计算诊断。优化器不会训练校准模型、选择参数或输出回测评价。

## 历史收益校准

`RankReturnCalibrator.initial_state()` 创建冷启动状态，`predict(ranks, state)` 只读取状态并返回每股预期收益。`ranks` 为 [0,1] 分位排名；高层入口先在信号横截面内做平均并列排名 `(rank - 0.5) / N`，高因子值对应高排名。因子方向由调用者确定。

默认 20 桶，正态均值先验中心为 `1.2bp × (2q - 1)`，q 是桶中心；`prior_std=0.0002`、`observation_std=0.02` 均为收益小数单位。这只是冷启动先验，历史样本会更新桶均值。`halflife_days=63` 按样本可用时间的日历天间隔遗忘历史充分统计量，保留先验精度。新样本到来时计算：

- `counts`、`sums` 乘 `2 ** (-elapsed_days / halflife_days)`，再加入该桶样本。
- 后验精度 `1/prior_std² + counts/observation_std²`。
- 后验均值 `(prior_mean/prior_std² + sums/observation_std²) / precision`。
- 相邻桶以固定 `smoothing=0.25` 做局部加权平均；中间桶左右各占 0.25，端点缺失邻居的权重留给自身。此参数不拟合、不自动选择。

`update(state, realized_samples)` 返回新状态，原状态不变。由外部程序在对应收益实现且可用后显式调用。每行字段：

| 字段 | 定义 |
|---|---|
| `rank` | 该信号产生时保存的分位排名，不用收益实现时的新排名 |
| `realized_at` | 该收益周期已完整实现的时间 |
| `available_at` | 实际取得该收益的时间，可省略为 realized_at；不得早于 realized_at |
| `raw_return` | `basis="raw"` 使用的已实现收益 |
| `barra_neutral_return` | `basis="barra_neutral"` 使用的已实现、外部完成 Barra 中性化的收益 |

两种口径各维护独立状态，优化器不执行收益中性化。样本按可用时间更新，不能早于原状态的更新时间；调用方负责避免重复提交样本。`horizon` 默认 `1d`，用于声明收益周期；调用方负责提供同周期样本和风险输入。`model_id` 标识一套校准参数，`state_id=model_id:version` 标识顺序更新版本，不计算哈希。分叉历史应使用不同 model_id。使用 `to_dict()` / `from_dict()` 保存、读取 JSON。

`predict(..., return_uncertainty=True)` 返回 `mu` 与 `uncertainty` 两个字典；uncertainty 是平滑后均值估计的标准差，不是股票收益波动率。预测时不读取价格、不读取未来收益、不推进状态，也不按当前时间自动衰减状态。

`none` 表示已输入预期收益；`zscore / rank_zscore` 保留变换、alpha_scale、alpha_clip、alpha_return_scale 的原顺序。不设 `alpha_return_scale` 的标准化分数只能用于兼容目标。`calibrated_rank` 直接使用校准收益，不再叠加这些缩放和截断。

## Barra asof 与统一暴露

`RiskMatrixProvider.snapshot(decision_time, assets)` 在 X、Ω、D 都存在的历史快照日期中，选择严格早于交易日期的最新日期。例如 11 月 10 日 10:00 或 14:30 都只能选 11 月 9 日或更早的数据；周一可回溯到周五。没有可用历史快照直接报错。数据的日期表示 Vendor 快照所属交易日，调用方应提供当时已发布的版本。

- X：`date/code/因子列...`，保留历史 Barra 原始数值，不按当前股票池重新标准化。
- Ω：长表 `date/factor_i/factor_j/covariance`；默认提供 Vendor 数据，也可直接替换为外部估计好的同因子 Ω，日期仍须与 X、D 一致。
- D：`date/code/specific_var`，或 `specific_risk`（标准差，读取后平方）。

风险保持 `Σ = X Ω Xᵀ + D`，不新增统计因子。高层入口以及将 RiskMatrixProvider 传入 build_targets 的路径，均直接用这份 X 计算风格和行业约束。行业须使用与 Ω 一致的列名；CLI 分类行业列会展开为 `industry:名称`。历史风格仅约束、未使用风险模型的兼容路径也选前一日暴露，默认不标准化。

原有个股数据填补行为保留：缺失暴露填零，缺失特异方差使用当天中位数或 specific_var_floor。不新增数据一致性扫描；调用方负责提供所需资产和因子覆盖。这些填补不代表真实风险已被完整测量。

## 优化目标和参数

单次高层入口默认 `objective_mode="te_cost"`：

`max muᵀ(w-b) - buy_rate * sum(max(w-p,0)) - sell_rate * sum(max(p-w,0))`

约束为满仓、个股上下界、交易权限、命名暴露上下界，以及 `activeᵀ Σ active × risk_annualization <= tracking_error²`。可选增加换手、单股交易量、成本/Alpha 门槛。提供真实当前持仓 p；费用以占组合净资产的收益小数表示。

| 参数 | 定义 / 默认 |
|---|---|
| `tracking_error` | 必须显式给出年化 TE 预算，例如 0.05；不自动选预算 |
| `risk_annualization` | 一年内的协方差周期数，日频 252，已年化协方差用 1 |
| `buy_cost_bps / sell_cost_bps` | 买入/卖出每单位成交金额的预计总费率，各默认 8bp；调用者按佣金、税费、预期滑点设定 |
| `max_weight / max_active_weight` | 默认 0.03 / 0.01，分别为个股权重和相对基准偏离上限 |
| `exposure_bounds` | `{名称: (下限, 上限)}`，独立控制风格、行业，`(0,0)` 表示严格中性 |
| `max_turnover` | 单边换手 `0.5 * sum(abs(w-p))` 上限；现金入场时仍采用此定义 |
| `max_trade_weight` | 单股 `abs(w-p)` 上限 |
| `turnover_cost_alpha_fraction` | 可选 `成本 <= fraction * muᵀ(w-p)`，默认关闭 |
| `solver_time_limit` | Clarabel 单次求解上限，默认 30 秒；不含数据加载和模型构建 |

买卖费用直接进入目标，按真实方向分开计算，既不是 turnover_penalty，也不是只在诊断里扣费。费率为调用方估计，不模拟真实成交。默认目标不使用 risk_aversion、turnover_penalty 或 active_exposure_penalty。

采用 [Clarabel 稀疏锥优化](https://clarabel.org/stable/python/getting_started_py/)。FactorCovarianceMatrix 输入只构建因子矩阵、对角特异风险和稀疏约束，不展开股票数量平方的协方差。兼容显式传入的稠密协方差。未在本次开发中运行 4,000 股性能基准。

显式 `objective_mode="legacy"` 保留风险厌恶型目标和原求解路径。低层 optimize / optimize_result 和旧逐日 CLI 默认仍为 legacy；旧 turnover_penalty 是平滑绝对值惩罚，不能当费率。旧统一暴露软惩罚只在 legacy 生效。统一硬带宽 `active_exposure_limit` 仍保留，与命名边界取交集。

## 交易权限和股票池

| 最新状态 | 对目标权重的约束 |
|---|---|
| `status=normal` | 允许买卖 |
| `status=limit_up` 或 `limit_label=1` | 禁买，w ≤ p |
| `status=limit_down` 或 `limit_label=-1` | 禁卖，w ≥ p |
| `status=suspended`、`tradable=false` 或 `frozen=true` | 冻结，w = p |
| `can_buy=false` / `can_sell=false` | 分别禁买 / 禁卖 |

也接受 `limit_label=0`、`tradable=true` 或完整的 can_buy/can_sell 权限声明为已知状态。记录可同时包含限制，限制取交集。缺失记录、缺失 observed_at 或仅有决策后记录均属于未知状态，默认拒绝求解。

`AssetConstraints` 原有字段保留。`allowed_assets` 排除的股票若可卖则目标为零；已持有但禁卖或冻结则保留原权重，且不允许再增持。其他个股界限、主动权重限制仍取交集，不会自动放宽。例如冻结持仓超过 max_weight 时仍不可行。

## 脚本适配

`run_backtest.py build-targets` 增加 `--decision-time` 后启用单次模式。关键参数：

- `--factor-values`：信号表含 code、因子列及 signal_time；无逐行时间时用 `--signal-time`。`--factor-column` 可指定因子列。每股只取截止决策时间的最新信号。
- `--previous-weights`：必须提供真实持仓快照 code/weight；可加 date 或 observed_at，选择截止决策时点的最新整份快照，不跨快照补回已清仓股票。
- `--benchmark-weights`、`--universe`、`--asset-constraints`：同样按最新整份快照读取；无时间列表示调用方已提供当前快照。非 ZeroIndex 默认只允许基准成分股，`--allow-off-benchmark` 可放开。
- `--trade-status`：含 code/observed_at 及状态列；默认回落到 --tradability 指定文件。按每股最新决策前记录生成权限。
- `--calibration-state`：外部保存的 CalibrationState JSON；默认使用 calibrated_rank，也可显式选择其他 --alpha-transform。
- `--risk-exposures / --factor-covariance / --specific-risk`：历史 Barra 文件；此模式始终使用风险输入。可加 --industry-exposures。
- `--tracking-error`、`--buy-cost-bps`、`--sell-cost-bps`、`--objective-mode` 对应上述参数。
- `--exposure-bounds`：TOML 的 `[exposure_bounds]` 表，键与 Barra 列名一致，值为 `[下限, 上限]`。
- `--unknown-status-policy freeze`：调用方明确选择的保守处理规则；默认 reject。

输出 Parquet 含 date/decision_time/code/weight，同名 .diagnostics.json 是该次诊断对象。CSV 和 Parquet 均可输入，无需固定目录组织。

不加 --decision-time 保留逐日 build_targets 行为，默认 zscore / legacy。该兼容路径使用调用者已按日对齐的约束表，未列出记录不附加交易限制，不具备盘中状态截止规则；盘中或实用持仓输出应使用单次模式。逐日初始持仓来自 --previous-weights，后续仍支持 --asset-returns 漂移；API 可用 calibration_states_by_date 提供逐日已可用状态。输出 date/code/weight 和逐日诊断数组。

## 诊断与输出检查

| 字段 | 含义 |
|---|---|
| `predicted_returns` | 每股预期收益，未提供信号的资产按零收益参与 |
| `alpha_calibration_state_id` | 所用状态版本；未校准时为空 |
| `barra_asof_date` | 实际采用的历史快照日期 |
| `predicted_te` | 年化主动风险标准差 |
| `factor_risk / specific_risk` | 年化主动**方差贡献**，两者相加等于 predicted_te²；不是两项波动率 |
| `active_exposures` | 高层接口为 `{因子名: Xᵀ(w-b)}`；低层兼容列表，并另给 named_active_exposures |
| `expected_cost` | 买入和卖出预计费用之和，占净资产比例 |
| `turnover` | 计划单边换手 |
| `constraint_slacks` | 权重、换手、TE、命名暴露和可选成本门槛的约束余量 |
| `solver_status` | 求解状态；te_cost 接受 Solved / AlmostSolved 后仍检查输出可行性 |

仅检查产出的持仓是否有限、是否满足权重和约束；违反约束则报错，不返回不可用持仓。不做显著性检验、回测评价、自动调参、文件哈希或输入一致性扫描。总目标权重固定为 1，现金保留目标、订单执行、自动放宽约束不在本接口内。
