# 组合优化器

通过唯一入口 `run_backtest.py build-targets` 生成目标权重，无需运行回测或成交模拟。原有目标权重输出仍为 `date/code/weight`，同目录新增同名 `.diagnostics.json`，按日期记录求解结果。

## Alpha 与风险单位

- 已有预测收益：使用 `--alpha-transform none`；收益以小数表示，例如 0.001 为 10 bps。
- 因子分数：使用 `zscore` 或 `rank_zscore`，通过 `--alpha-return-scale` 指定每单位变换后分数对应的预期收益。顺序为变换、`alpha-scale`、截断、`alpha-return-scale`。比例由调用者提供；不会用本次回测的未来收益拟合。
- 不提供收益比例时仍可做分数优化，但输出会标为 `alpha_units=score`，预期 Alpha 数值不可当作收益率。成本/Alpha 门槛不能用于这种模式。
- `--turnover-cost-alpha-fraction` 默认关闭；显式启用时约束交易成本不超过预期收益改善的指定比例。
- Alpha 和协方差使用相同预测周期，风险惩罚作用于该周期方差。`--tracking-error` 和诊断中的风险均为**年化**值；`--risk-annualization` 是一年内的协方差周期数，日频默认 252，已年化协方差用 1。年化跟踪误差为 `sqrt(active_weight' Sigma active_weight * risk_annualization)`。

**兼容性变化：**旧版 tracking-error 直接与输入协方差同周期比较。迁移旧配置时需要换算为年化值，或设置 risk-annualization 为 1 保持原数值语义。pipeline 使用 252；旧目标缓存需使用已有的 `--refresh-cache` 重新生成。

## 独立生成持仓

`--previous-weights` 接受 `date/code/weight` 表，使用最后一日快照作为第一次优化前的持仓。文件应由调用者提供为当前已漂移到调仓时点的权重。多日期运行后续使用上次目标；提供 `--asset-returns` 时按已有逻辑漂移。未提供初始持仓则以基准权重为起点，ZeroIndex 以等权为起点，因此首期换手是相对该起点计算。

`--max-daily-turnover` 是单边换手 `0.5 * sum(abs(w - previous))` 上限；`--max-trade-weight` 是单个资产权重改变量上限。交易费用按买卖合计权重变化乘以 bps 计算。换手与成本门槛直接进入求解，不再在求解后缩放或回退到旧持仓。

API 保留 `IndexEnhancementStrategy.optimize(...) -> dict`；`optimize_result(...) -> OptimizationResult` 返回权重和诊断。这两个单次调用接收**已经准备好的、每周期预测收益**；变换和标定配置只用于 `build_targets`。`build_targets(..., initial_weights=...)` 返回权重表，逐日诊断保存在 `strategy.diagnostics`。

## 输出持仓检查

仅对求解结果做一次检查：有限权重、权重和、上下界、换手、年化跟踪误差、暴露限制及成本门槛。违反约束会报错，不输出该次权重。不扫描输入数据的一致性、不计算文件哈希、不做协方差半正定检验。

诊断包含求解器与停止状态、约束余量、权重总和、总绝对权重、最大权重/主动权重/交易量、换手、预期 Alpha、收益改善、费用、年化波动率/跟踪误差和主动暴露。暴露列表顺序与输入矩阵列一致。`constraints_satisfied` 仅表示权重满足约束；投影梯度的 `line_search_stopped` 或 `iteration_limit` 不表示已证明最优。

启用风险模型或硬跟踪误差时，缺少该日风险矩阵直接报错；启用暴露控制时缺少该日暴露矩阵也直接报错。未要求风险模型时仍允许单位矩阵正则化，诊断标记为 `identity_regularizer`，预测波动率与跟踪误差留空。股票级风险数据的原有填补逻辑不变。

## 行业与风格边界

`--exposure-bounds` 读取 TOML 的 `[exposure_bounds]` 表。键为暴露名称，值为相对基准的 `[下限, 上限]`，可不对称，`[0.0, 0.0]` 表示严格中性。风格名称与风格表列名一致，CLI 沿用横截面标准化风格；行业名称加 `industry:` 前缀，例如 `"industry:Bank"`。行业文件可为 `industry` 分类列或各行业的 0/1 列，行业列不标准化，所以 0.02 表示相对基准 2 个百分点的权重偏离。

原 `--style-exposure-limit` 在提供的暴露矩阵上设置统一带宽；同时使用命名边界时取交集。混合风格与行业时，建议只使用命名边界，以分别控制各自单位。软暴露惩罚作用于整个传入矩阵。诊断新增 `named_active_exposures` 和逐暴露上下限余量。

API 使用 `IndexEnhancementConfig(exposure_bounds={名称: (下限, 上限)})`，并向 `optimize` / `optimize_result` 传入 `exposure_names`，顺序与 `exposure_matrix` 列一致。API 直接使用提供的暴露数值，不自动标准化。批量 `build_targets` 也接受 `exposure_names`，或使用暴露 provider 的 `exposure_columns`。

## 股票池、个股边界与持仓权限

`--universe` 文件中的 `eligible=true` 行构成当日允许持有的股票池；池外和当日未列出的股票权重必须为零。非 ZeroIndex 默认还要求在基准成分股内，`--allow-off-benchmark` 可取消该项限制。禁止持有不会通过删除基准成分股并重新归一化基准来实现，基准本身保持不变。因子文件不再因股票池过滤而丢失整天的调仓记录。

`--asset-constraints` 接受带 `date/code` 的 CSV 或 Parquet；以下列均可选，空值表示该项不附加限制：

| 列 | 对目标权重的限制 |
|---|---|
| `min_weight` / `max_weight` | 个股权重下限 / 上限 |
| `fixed_weight` | 指定固定目标权重 |
| `frozen=true` | 保持调仓前权重 |
| `can_buy=false` | 目标权重不高于调仓前权重 |
| `can_sell=false` | 目标权重不低于调仓前权重 |

`--tradability` 中 `tradable=false` 现在表示冻结持仓，不再只是过滤 Alpha。该表也可使用 `frozen/can_buy/can_sell`。这些是目标权重上的限制，不执行成交模拟。冻结或禁买禁卖需要提供真实当前持仓，首次运行用 `--previous-weights`；后续使用上一期持仓及已有收益漂移逻辑。未列出的日期/股票不附加交易权限限制；文件不自动向后填充。股票池文件则必须提供每个优化日期的允许集合。

所有限制与全局最大权重、主动权重和交易上限取交集，不覆盖或自动放宽。例如某股冻结在 4%，而全局上限为 3%，会报告不可行；某股同时被排除出股票池并冻结在非零权重，也会失败。

API 的 `AssetConstraints` 提供 `weight_bounds`、`allowed_assets`、`frozen`、`no_buy`、`no_sell`，通过 `asset_constraints` 传给单次优化；`build_targets` 使用 `asset_constraints_by_date`（日期字典或日期回调）。证券代码键与 Alpha/基准保持一致；约束中提到的股票会进入优化资产集合，即使没有 Alpha，仍按零 Alpha 参与。

当前仍为总权重等于 1 的组合优化。基准成分股最低持仓比例、可配置净/总敞口，以及自动放宽约束尚未加入。
