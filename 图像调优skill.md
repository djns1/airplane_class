# 图像调优方法论总结

> 本文是对 `label-guided-isp-tuning` skill（车载相机 ISP 自动调优）的逆向整理。
> 读它的目的不是"会用这个 skill"，而是**抽出它的设计骨架，迁移到任何"有参考基准 + 有一堆旋钮"的问题上**。
>
> 文中标注约定：
>
> - ✅ **默认生效**：不配任何开关，代码就这么跑
> - 🔧 **可选**：需要显式开环境变量/参数
> - 📄 **设计表述**：文档里写的方法论，未直接落到默认执行路径

---

## 0. 一句话概括

> 有张标准答案（label），拿尺子量差距（指标），拧旋钮（参数），
> **先大后小地试（步长公式）**，尺子不过就打回（judge），实在拧不动就换方法；
> 旁边请个人眼顾问说"哪儿看着别扭"（VLM），但他没权判分；
> 干完把经验带数字地写回知识库，下次直接查。

核心是三个东西：**指标化验收** + **有公式的搜索** + **会沉淀的知识库**。

---

## 1. 整体闭环

```
三输入（工程 / 输入数据 / 参考基准）
   ↓
prep：自动勘察、配对、基线、生成目标宪章
   ↓
profile6：六维差异画像 ─────────────┐   ← 硬门①：不跑不许写参
   ↓                                │
知识检索（经验 + 指南 + 专家）──────┤   ← 硬门②
   ↓                                │
plan.json（目标 / 维度 / 步骤）──────┘   ← 硬门③：无 plan 禁止 apply
   ↓
[循环] probe（探活性）→ diagnose → apply --verify → run → judge
   ↓
checkpoint（唯一出口判定）
   ↓
达标 → retrospective.post_tune → 经验回写知识库
```

**合法出口只有几种**，其余一律打回：全部 gate 通过 / 用户喊停 / 预算耗尽 / 超时取 best / 人工暂停。
"平台期"**不是**出口——平台期只允许换道、换族、换算法、回访。

---

## 2. 指标体系（三层，别混着看）

### 第 1 层：硬门（判"过没过"，裁判只看这层）

| 维度 | 判据                                               | 逻辑                         |
| ---- | -------------------------------------------------- | ---------------------------- |
| 亮度 | `Y_PSNR > 34` **或** `Y_hist_sim > 0.96` | **OR**（两条路通一条） |
| 细节 | `RGB_PSNR > 33`（原图，禁止下采样）              | 单条                         |
| 颜色 | 感知门**且** 饱和度门                        | **AND**                |

- **感知门**：`D_color ≤ 4.0`，其中
  `D_color = 0.45·ΔE00_mean + 0.30·ΔE00_p95 + 0.15·block_p90 + 0.10·neutral_dE`
  另外 `ΔE00 mean≤3 / p95≤6 / p99≤10`、`block_max≤10`、色相误差 `≤5°`、中性区 `≤3`
- **饱和度门**：`D_sat≤3.0`、`|ΔC|mean≤2.0`、`|ΔC|p95≤5.0`、`C_ratio∈[0.90,1.10]`、过/欠饱和比 `≤0.20`、`neutral_C≤3.0`
- **H/S 直方图相似度已降级为"护栏"**（不是门）：抹平颜色也能刷高直方图相似度，所以它只做回归保护，绝对阈值 `0.90` 或相对回归 `−0.01`

> 这里有个重要设计：**OR 门取"更近的那条分支"算差距，AND 门取"最差的那一项"算差距**。
> 直接决定了后面 `gap` 怎么算（见 §3.2）。

### 第 2 层：诊断特征（判"往哪使劲"，不判过没过）

| 维度   | 代表特征                                                                                                                                        | 一句话用途                           |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| 亮度   | `Y p5/p50/p95`、`dark_band_delta`、`bright_band_delta`、`clip_high_delta`、`clip_low_delta`                                           | 高光还有余量吗？欠曝在暗部还是整体？ |
| 对比度 | `Y_std_ratio`、`lc_ratio_mean`（局部对比度）、`tone_delta_curve`(16 段映射) RMS                                                           | 是"发灰"还是"反差过大"               |
| 色相   | `ΔE00 mean/p95/p99`、`ΔE76`、`hue_err_deg`、`dHue/dC/dE` 分量、`neutral_dE`、`global_fit.reduction_pct`、颜色簇 top3、8×8 热点块 | 全局色偏还是局部色相错？             |
| 饱和度 | `S_mean/std_ratio`、`dC_abs_mean`、`C_ratio_median`、`D_sat`、`over/under_ratio`、`neutral_C`、`by_luma / by_hue`、`wb(da,db)`  | 是整体欠饱和还是某亮度段/某色相段    |
| 噪声   | `noise_luma_bands`（分频带噪声比）、`impulse_idx`、`flat_a_bias`                                                                          | 灰噪还是彩噪？麻点？                 |
| 细节   | `acutance_ratio`、`texture_ratio`（中频纹理，防抹平）、`rgb_psnr`、`falsecolor_ratio`                                                   | 是软了还是被抹平了                   |

**关键理念**：主指标（均值类）对"少数像素的不可逆破坏"是钝感的。所以每个主指标都配了**盲区护栏**：

| 盲区                           | 补的护栏                                                                  |
| ------------------------------ | ------------------------------------------------------------------------- |
| 单向破坏性误差（裁剪、过饱和） | `clip_high/low_delta`、`high_sat_delta`（全分辨率算，下采样会平均掉） |
| 分布形状（双峰、量化条带）     | `y_hist_sim`                                                            |
| 中频纹理被抹平                 | `texture_ratio`                                                         |
| 空间局部化异常（边角过曝）     | `tile_dY_max`（8×8 分块）                                              |
| 场景安全色（刹车灯/交通灯）    | `red_dHue`                                                              |

### 第 3 层：冲刺目标（自己给自己定的"优秀线"）

从"三张图实际调成功后的状态"反推标定，比门更狠：

```
psnr_floor ≥ 38         J ≤ 0.85            dE_mean ≤ 1.0
p95_dE ≤ 3.5            tile_max_dE ≤ 3.2   tdc_rms ≤ 2.5
lc_ratio ∈ [0.95,1.05]  acutance_ratio ∈ [0.95,1.05]
noise_luma_flat ≤ 1.1   texture_ratio ∈ [0.85,1.15]
```

> **门 = 及格，这张表 = 90 分。** 先奔及格，有余力再奔这张表。
> 这张表最值钱的是它的备注：每个数值都写了"从哪张图的哪个成功态标的"，改阈值必须留痕。

---

## 3. 调整公式（本文重点）

### 3.1 总览：步长是五层叠出来的，不是拍脑袋

```
                    ┌── ① 基础格 grid（该旋钮最小可动一格）
                    │
step = f( ② 反馈倍数  │   accept ×2 / reject ÷2 / neutral ×1
         ③ 裕度夹紧   │   贴门限 → ≤2；快到了 → ≤3；撞护栏 → ÷2^hits
         ④ 硬顶       │   MAX_TRUST_STEP = 8（禁止 ≥10 盲步）
         ⑤ 外部档位   ┘   LLM small/mid/large = 0.5×/1×/2×；人工 step_abs 可越过硬顶
       )
```

**不是"大力出奇迹"**——预算是死天花板（单模块 100 次、单图 1000 次渲染），穷举两小时就爆了。
唯一的"大力"是**探针那一把**（故意打到区间端点），目的是花 1 次省掉后面十几轮。

---

### 3.2 公式①：还差多少 —— `gap`（亏欠比）

定义：`gap ≥ 1` 表示没过门（1.1 = 差 10%），`gap = 0` 表示已过。

```python
# ✅ 默认生效（planner._gate_gap）

# 亮度：OR 门 → 取"更近的、能救回来的那条分支"
gap_brightness = max(1.0, min(
        34.0 / y_psnr,          # 走 PSNR 分支还差多少
        0.96 / y_hist_sim       # 走直方图分支还差多少
))

# 颜色：AND 门 → 取最差的一项（木桶原理）
ratio_max(v, lim) = v/lim   if v > lim else 1.0     # 越大越差的指标
ratio_min(v, lim) = lim/v   if v < lim else 1.0     # 越小越差的指标

ratios = [
    ratio_max(ΔE00_mean, 3.0), ratio_max(ΔE00_p95, 6.0), ratio_max(ΔE00_p99, 10.0),
    ratio_max(D_color, 4.0),   ratio_max(block_max, 10.0),
    ratio_max(hue_err_deg, 5.0), ratio_max(neutral_dE, 3.0),
    ratio_max(D_sat, 3.0),     ratio_max(dC_abs_mean, 2.0), ratio_max(dC_p95, 5.0),
    ratio_max(over_ratio, 0.20), ratio_max(under_ratio, 0.20), ratio_max(neutral_C, 3.0),
    ratio_min(h_sim, 0.90),    ratio_min(s_sim, 0.90),
    ratio_min(C_ratio_median, 0.90), ratio_max(C_ratio_median, 1.10),
]
gap_color = max([1.0] + ratios)

# 细节：同构，用 rgb_psnr
gap_detail = max(1.0, 33.0 / rgb_psnr)
```

**为什么这么定义**：`gap` 不是给"看"的，是给"算"的——它要作为下一步"该走多远"的输入。
用比值而不是差值，是因为不同指标的绝对尺度不可比（0.1 的相似度和 4 dB 的 PSNR 没法直接相加）。

---

### 3.3 公式②：基础格 —— 该旋钮最小能转的一格

每个参数族在配置里写死 `[lo, hi, grid]`：

```json
["brightness", "drc_contrast",    ["drc", 4, 15, 2]]
["brightness", "drc_brightness",  ["drc", 0, 255, 16]]
["detail",     "bnr_coarse",      ["bnr", 40, 200, 22]]
["color",      "ccm_hue_row3",    ["ccm", 0, 40, 2]]
```

`grid` 就是基础步幅，**不能再小**——小了会被硬件量化/阈值吞掉。

> **血泪史**：AWB 的 `static_wb` 用 ±5 试，毫无反应，被判"死旋钮"；实测要 **±40** 才可见。
> demosaic 的 `desat_uv_gain` 有量化台阶：24→63 一点不变，**64 突变**。
> 这类"步长过小导致的误判"专门有一张清单，是这套方法存在的理由之一。

---

### 3.4 公式③：活性探针 —— 唯一的"大力"

新族第一次碰，**不打小步，直接跳到离当前值最远的区间端点**：

```python
# ✅ 默认生效（liveness.probe_value）
if cur is None:                       probe = hi          # 没当前值，取上端
elif abs(hi - cur) >= abs(cur - lo):  probe = hi          # 取距离更远的那端
else:                                 probe = lo

# 一轮判死
if render_md5_unchanged or affected <= 0.002:
    search_guard.exhaust(family, ttl=permanent)   # 会话级永久拉黑
else:
    enter_normal_search()                          # 活的，继续
```

**收益**：复盘数据显示，死族在全网格扫描下每会话要浪费 **15~21 轮**才凑够"判定死亡"的样本；
探针一轮就能判定，省下的预算全给活族。

**风险控制**：探针用的是"区间端点"，端点必须在**已验证的工作范围**内（不是硬件极限），
且探针写参必须能被 best 快照正确还原（历史上有过"探针写参泄漏到 live xml"的 bug，专门加了状态卫生检查）。

---

### 3.5 公式④：自适应剂量 —— 按学到的影响率反算（可选）

这是"有理论依据"的那一层，来自**参数→特征效应库**（离线从历史轨迹拟合出的斜率）：

```python
# 🔧 可选：PARAM_EFFECTS_MODE=on 才真用；默认 shadow 只记录不改决策
# param_effects.suggest()

span = hi - lo
cap  = max(4 * grid, 0.25 * span)      # 安全上限：不超过量程 25%，且至少 4 格
base = snap( min(gap / |slope|, cap), grid, lo )

snap(v, grid, lo) = lo + round((v - lo) / grid) * grid    # 对齐到网格
clamp(v, lo, hi)  = max(lo, min(hi, v))

tiers = { "small": 0.5×base, "mid": base, "large": 2×base }
```

含义直白：**"要补上 `gap` 这么多的特征差距，按实测斜率 `slope`（每单位参数能换来多少特征变化）算，需要动多少参数"**。

文档层的等价表述（更早的方法论）：

```
G = clamp(max(G_metric, G_feature), 0, 1)
step = coarse_step * clamp(G / 0.30, 0.15, 1.0)

G > 0.20 → coarse 1.0×
0.05~0.20 → medium 0.4×
G < 0.05 → fine 0.15×（但不小于量化单位）
```

> ⚠️ **默认跑的时候这一层不生效**（`shadow` 模式只写日志）。
> 开箱实际生效的是"基础格 + 反馈倍数 + 信任域夹紧"，也就是 §3.3 / §3.6 / §3.7。

---

### 3.6 公式⑤：反馈倍数 —— 对了加速，错了回退或掉头

```python
# ✅ 默认生效（planner.choose_candidate）

if 上次 verdict == "accept" or "provisional-accept":
    step = min(step * 2, hi - lo)            # 加速往前走

elif 上次 verdict == "reject" and 上次 delta < 0:
    direction = 反方向，step 不变              # 走反了 → 掉头

elif 上次 verdict == "reject":
    direction = 同方向，step = max(1, step // 2)   # 走过头了 → 减半

elif 上次 verdict == "neutral":
    direction = 同方向，step 不变              # ⚠️ 故意不减半
```

**为什么 neutral 不减半**（这是踩过坑的）：
减半 → 变化更小 → 更测不出变化 → 再减半，形成**假反馈死循环**。
正确做法是"零响应"由另外两个机制接管：`search_memory` 连续 4 轮零响应判死，或 `liveness` 探针直接拉黑。

---

### 3.7 公式⑥：信任域夹紧 —— 越贴门限，越不许大步

```python
# ✅ 默认生效（planner._trust_step + _dim_margin）

MAX_TRUST_STEP = 8              # 环境变量可改

# ── 裕度：0 = 贴着门限/护栏；1 = 离门限很远，可以大步 ──
if dim == "brightness":
    margin = clamp( (y_psnr - 30.0) / 4.0, 0, 1 )        # 或用 (y_hist_sim - 0.86)/0.10
elif dim == "color":
    margin = min( clamp((h_sim - 0.90)/0.10, 0, 1),
                  clamp((s_sim - 0.90)/0.10, 0, 1) )

# ── 夹紧（四道闸） ──
if margin < 0.25:  s = min(s, 2)                    # 贴门限，最多 2 格
elif margin < 0.60: s = min(s, 3)                   # 快到了，最多 3 格

if 上次 reject and 上次 delta < 0:  s = max(1, s // 2)

hits = 同族连续撞护栏次数
if hits:  s = max(1, round(s / 2 ** min(hits, 3)))  # 撞 1 次半，撞 3 次 1/8

s = max(1, min(s, MAX_TRUST_STEP))                  # 硬顶 8
s = max(s, min(grid, MAX_TRUST_STEP))               # 下限 = 原生网格（否则慢扫空转）
```

**下限的意义**：网格是 0.03 的参数，如果被夹到 0.01，就变成"60 个点慢扫"，比大步试更浪费预算。
所以"收敛"不能收敛到低于该族原生精度。

---

### 3.8 完整伪代码（把五层串起来）

```python
def choose_next_step(dim, family, gate_state, memory, llm_plan, session):
    lo, hi, grid = spec(family)                       # ① 基础格
    gap  = gate_gap(dim, gate_state)                  # ② 还差多少
    cur  = current_value(session, family)

    # ③ 新族首探：端值，一轮定生死
    probe = liveness.probe_value(session, dim, family, lo, hi, cur)
    if probe is not None:
        return {"value": probe, "why": "liveness-probe"}

    step = grid
    # ④ 反馈倍数
    v, d, last_dir = memory.last(family)
    if   v in ("accept", "provisional-accept"): step = min(step * 2, hi - lo)
    elif v == "reject":                        step = max(1, step // 2)

    # ⑤ 信任域夹紧
    step = trust_step(dim, gate_state, family, step, session, native=grid)

    # ⑥ 外部档位（LLM 只能给档，不能给数）
    if llm_plan.get("step"):      # "small"|"mid"|"large"
        step = round(step * {"small": 0.5, "mid": 1.0, "large": 2.0}[llm_plan["step"]])

    # ⑦ 人工覆盖（唯一能越过硬顶的通道）
    if llm_plan.get("step_abs"):
        step = clamp(int(llm_plan["step_abs"]), 1, hi - lo)
    else:
        step = clamp(step, 1, MAX_TRUST_STEP)

    # ⑧ 落到网格上，跳过已试过的值
    value = snap(cur + sign * step, grid, lo)
    while value in tried:  value += sign * step
    return {"value": clamp(value, lo, hi), "step": step, "gap": gap}
```

---

### 3.9 数值例子

**例 A：ISP 的 `drc_contrast`**（`lo=4, hi=15, grid=2`，当前 4）

| 轮次 | 触发       | 计算过程                                                                          | 结果 step          |
| ---- | ---------- | --------------------------------------------------------------------------------- | ------------------ |
| 1    | 新族首探   | 当前 4，到上端 11 更远                                                            | 直接打**15** |
| 2    | 探针有响应 | 基础格 2 → margin：`y_psnr 29.6` → `(29.6−30)/4 < 0` → margin=0 → 夹到 2 | **2**        |
| 3    | accept     | `min(2×2, 11) = 4`；夹紧后仍 4                                                 | **4**        |
| 4    | accept     | `min(4×2, 11) = 8`；硬顶 8                                                     | **8**        |
| 5    | accept     | `min(8×2, 11) = 11` → 硬顶 8                                                  | **8**        |
| 6    | 到顶       | 落值超 hi → 换族                                                                 | —                 |

**例 B：换成感知的 `score_thr`**（`lo=0.05, hi=0.60, grid=0.05`）

同一套公式，只是把"格"换成 0.05：首探打 0.60 → 基格 0.05 → accept 0.10 → 0.20 → 0.40（硬顶 8 格 = 0.40）。
**代码一行都不用改**，只改 spec 文件。

---

### 3.10 平台期逃逸：网格走不动时换算法（不是乱试）

连续 N 批无 accept（默认 6~10）触发"死胡同协议"，这时**禁止收尾**，只允许换道：

**① BO（GP + EI）** —— 参数 ≥3 个且预算 ≥15 次时启用，替代坐标下降

```
μ(x), σ(x) ← GaussianProcessRegressor 拟合已试过的 (参数, 分数)
z = (μ(x) − f_best − ξ) / σ(x)
EI(x) = (μ(x) − f_best − ξ)·Φ(z) + σ(x)·φ(z)     ← 选 EI 最大的点
```

是**模型推断**，不是随机搜索。

**② active_gn（活跃集 + Levenberg–Marquardt 反演）** —— 模型无关

```
1) 分层测活性：先按组（默认 4 个元素一组）× (+Δ) 测组是否活，组内再单测
2) 只在活跃集上解：  Δ = −(JᵀJ + λI)⁻¹ JᵀR      （J = 雅可比，R = 残差，λ 阻尼）
3) 沿方向线搜索，trust = 0.03（相对步长）
```

含义：**"要补掉这些残差 R，参数需要动多少"** —— 是解析反推，不是试错。

**③ 死胡同取证**：宣称"不可达"必须交五类证据之一，缺项直接驳回

| 证据类型     | 含义                                                               |
| ------------ | ------------------------------------------------------------------ |
| `type0`    | 测量链路本身先验证过（防"stale png 误判 14dB 天花板"这类测量 bug） |
| `coverage` | 参数轴已覆盖足够密度                                               |
| `ceiling`  | 单调曲线/天花板已量化（如 `Y_sim ≤ 0.9741`）                    |
| `redline`  | 被红线/护栏挡住（需登记例外并授权）                                |
| `reverse`  | 反向探索也做过，双向都劣化                                         |

---

### 3.11 ⚠️ 哪些公式"默认真的在跑"

这张表很重要，别把设计文档当成运行时行为：

| 层                           | 开关               | 默认        | 说明                             |
| ---------------------------- | ------------------ | ----------- | -------------------------------- |
| 基础格 `grid`              | 无                 | ✅ 生效     | 来自 `family_search_spec.json` |
| 探针端值                     | `LIVENESS_PROBE` | ✅ 生效(=1) | 新族首轮打端点                   |
| 反馈倍数（×2 / ÷2 / 翻转） | 无                 | ✅ 生效     | `planner.choose_candidate`     |
| 信任域夹紧 + 硬顶 8          | `MAX_TRUST_STEP` | ✅ 生效(=8) | `_trust_step`                  |
| 裕度分段（2 / 3 格）         | 无                 | ✅ 生效     | `_dim_margin`                  |
| LLM 三档 0.5/1/2×           | 无                 | ✅ 生效     | 但 LLM 只给档位                  |
| 斜率反算 `gap/               | slope              | `           | `PARAM_EFFECTS_MODE`           |
| BO 的 EI                     | `TOPK` / 预算    | 🔧 条件触发 | ≥3 参数且预算够                 |
| active_gn LM 反演            | 手动 campaign      | 🔧 离线工具 | 不是主循环默认                   |
| 文档的 `gap/0.30` 缩放     | —                 | 📄 设计表述 | 与斜率反算同源，未直接落地       |
| 重复评估确认                 | `CONFIRM_ACCEPT` | ✅ 生效(=2) | 防指标噪声误判                   |

---

## 4. 知识库设计（怎么快速从人的经验和文档里找到方法）

### 4.1 三层结构，各管一件事

| 层                | 目录                                          | 内容                                                              | 回答什么问题                                   |
| ----------------- | --------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| **L1 经验** | `agent_tuning_experience/`                  | 386 条结构化条目（`experience.json`）+ 速查 + 全量目录 + 方法论 | **"这情况该怎么动？踩过什么坑？"**       |
| **L2 指南** | `tuning_guide/<平台>/{reference,playbooks}` | 每模块一篇（参数路径/量程/方向/副作用）+ 平台 playbook            | **"这个参数的路径、范围、方向是什么？"** |
| **L3 专家** | `human_tuning_experience/`                  | 策略总纲、常调参数表、模块标注汇编                                | **"先调什么后调什么？权衡怎么做？"**     |

三层是**互补**关系，不是备份关系：
L2 给"静态事实"（参数在哪、量程多少），L1 给"动态经验"（这么动有效/无效、有什么副作用），L3 给"顺序与取舍"。

### 4.2 能"快速找到"的四个前提（这是设计精髓）

大多数人做知识库失败，不是内容不够，是**找不准**。这个库用四件套解决：

#### ① 症状词表：把"五花八门的说法"归一化

```json
{ "symptom": "欠曝", "dimension": "亮度", "want": "up",
  "aliases": ["偏暗", "太暗", "亮度不足", "看不清", "亮度低", "欠曝", "暗"] }
```

13 类规范症状，每类带 `aliases` + `dimension` + `want`（方向）。

匹配规则：**别名子串匹配，长别名优先**（`_alias_map` 按长度倒序，避免"偏亮"被"亮"抢先命中）。
找不到时回退扫 `title + rule` 正文。

> 这一步是关键：用户说"发灰"、文档写"对比度不足"、代码里叫 `lc_ratio`——
> 没有归一化，检索就永远对不上。

#### ② 清单索引 manifest：机器可读的目录

```json
{ "path": "tuning_guide/cpipe1952/reference/14_DRC.md",
  "title": "DRC调优指南", "platform": "cpipe1952", "type": "reference",
  "module": "DRC", "keywords": ["drc","local_mixing","对比度","暗部"], "sha256": "..." }
```

`build_manifest.py` 扫描全目录自动生成，字段全部可从文件本身推断：

- `platform`：按路径片段（`/cpipe1952/`、`/labelisp/`、`/shared/`）
- `type`：按文件名关键词（`playbook` / `methodology` / `调试策略总纲` / `常调参数` / `汇编` …）
- `module`：按文件名里的模块 token 匹配（`14_DRC.md` → `DRC`）
- `title`：抓第一个 `# ` 标题行
- `keywords`：从路径和标题抽英文标识符 + 中文词组（≤40 个）

**好处**：检索不用打开文件就能先筛，且 `sha256` 能检测内容漂移。

#### ③ 路由表 ROUTING.md：症状 → 模块 → 文件（人/AI 的第一跳）

```markdown
| 症状 | 首选模块 | 次选 | 指南文件 |
| 整体偏暗/偏亮 | PreTm(13) | DRC(14)→Dehaze(16) | 13_PreTm.md |
| 对比度不足/发灰 | DRC(14) | GTM→Dehaze | 14_DRC.md |
| 灰白颗粒噪声(luma) | BNR(02) | Sharpen(17)→DPC(01) | 02_BayerNR.md |
| 彩色噪点/chroma | 3DNR(21) | AntiFalseColor(05)→BNR | 21_3DNR.md |
```

这是"**先缩小到哪里**"，然后才进检索。"症状→模块"这一跳由人总结（最稳），"模块→参数"由检索完成。

#### ④ 检索打分：命中之后怎么排序

```python
# ✅ 默认生效（tools/kb_query.py）

# 硬过滤（不满足直接排除）
platform  必须 ∈ {query.platform, generic}
module    必须匹配
symptom   规范症状集合必须与条目的 symptoms_canonical 有交集
query     token 必须至少一个出现在 title/rule/evidence/conditions/tags

# 打分
score =
    15 × |规范症状 ∩ 条目症状|          # 症状命中，权重最高
  +  5                              # 模块命中
  +  severity{critical:6, high:4, medium:2, low:1}
  +  transferability{high:3, medium:2, low:1}
  +  1  if status == adopted
  +  5  if platform == query.platform
  + 25 × (query token 命中 title 的个数)
  + 12 × (query token 命中 rule 的个数)
  +  3 × (query token 命中 evidence 的个数)
  + 15  if 所有 query token 都在 title 里
if 症状命中: score ×= 1.3            # 症状是正确的第一信号
```

排序取 top-k → 输出时还做**正文片段抽取**：打开命中的文档，把包含 token 的行抽 6 条（每条截 180 字）贴出来，
这样 Agent 不用再"打开文件找重点"。

### 4.3 给 LLM 的"知识简报"：四路召回 + 可选改写

日常循环里 LLM 不是自己搜，而是拿到一份**预先组装好的简报**（`kb_brief.py`）：

```
A. 确定性多路召回（无 LLM）
   ① 症状路：六维特征 → 规范症状词（确定性映射，不是 LLM 判断）
             例：饱和度 D_sat ≥ 2.0 或 过/欠饱和比 > 0.15 → "饱和度过高/不足"
   ② 维度路：brightness → "欠曝 亮度"，color → "偏色 色相 饱和"
   ③ 模块路：当前正在调的模块
   ④ 场景/族路：scene + iso_band + 候选族名
   → 合并去重 → top-5 候选

B. 可选一次 LLM 改写（JSON 协议，最多 2 轮）
   只能"选/归纳/剔除无关"，不能改经验内容；引用 ID 必须真实存在

C. 输出 top-3 简报（含 why_relevant）

D. 兜底：任何异常/超时 → 直接返回确定性结果，绝不阻塞调优
```

模式 `KB_AGENT_MODE = off | shadow(默认) | on`：
**影子模式只记录"LLM 的选择"和"确定性结果"的差异**，不影响决策——先观察它准不准，再决定要不要放权。

### 4.4 一条经验长什么样（写入规范）

```json
{
  "id": "E175",
  "title": "单模块退化≠单参数:模块内其它参数可补偿残差",
  "rule": "退化只动一个参数时, 恢复不必只调该参数: 同一模块的其它常调参数可继续补偿残差
           (如 drc strength 收敛后, auto_curve.contrast x0.9 / local_mixing_ace_bright x0.85
           再 +1.3~2.0dB)。顺序: 退化参数先归位, 再用模块内补偿参数收尾。",
  "evidence": "drc 3图: strength单独 31.05/29.18/34.31 -> 加 auto_curve.contrast+ace_bright 后
               32.33/31.23/36.36 (+1.28/+2.05/+2.05dB)",
  "platform": "cpipe1952",
  "module": "DRC",
  "symptoms_canonical": ["欠曝", "发灰"],
  "dimensions": ["亮度", "对比度"],
  "severity": "high",
  "transferability": "high",
  "status": "adopted",
  "outcome": "success",
  "conditions": { "scene": "day", "stage": "drc-converged" }
}
```

**必填字段**：`title / rule / severity / platform / module / symptoms_canonical`
**必须有**：`evidence`（哪张图、什么数值变化）

四条硬规矩：

1. **只写通用规律，不写"这张图调好了"** —— 单图单场景的特例**不许**标 `transferability=high`
2. **写入只能走 API**（`knowledge.api.add()`），自动分配 ID、去重、规范化、重建索引
3. **写入后必须重建** `kb_manifest.json` / `experience_catalog.md`，且 `validate_kb.py` 0 errors
4. **AI 复盘默认只产 draft**，进 `review_queue.jsonl` 等人审，不直接改库

### 4.5 让知识库不腐烂的机制

| 机制                     | 作用                                                            |
| ------------------------ | --------------------------------------------------------------- |
| `validate_kb.py`       | 一致性校验（必填字段/ID 唯一/平台合法/交叉引用），必须 0 errors |
| `build_manifest.py`    | 重建索引（文档 sha256 + 关键词），检出内容漂移                  |
| `build_catalog.py`     | 重生成全量目录（按平台→模块），给人看的                        |
| `CHANGELOG.md`         | **每次行为变更的交接记录**，接手先读最近 5 条             |
| `experience_log.jsonl` | 追加式写入日志，可回溯"谁什么时候加了什么"                      |
| `merge_manifest.json`  | 多来源合并的来源与策略（本地/server1/server2）                  |
| 证据优先级               | `human > vlm > label`；同主题冲突以本地为准                   |
| 跨会话族账本             | 某族在多个会话里都零响应 → 全局降权，防"单图过拟合"            |

> **这一整套的核心思想**：知识库不是"文档堆"，而是**带索引、带打分、带来源、带证据、可校验、可审计**的结构化资产。

---

## 5. 泛化：任何"有标签 + 有旋钮"的问题都能套

### 5.1 抽象模板（六件套）

只要你的问题满足下面 6 条，就能照搬：

| # | 要件                               | ISP 里是什么                 | 你的领域里是什么                        |
| - | ---------------------------------- | ---------------------------- | --------------------------------------- |
| 1 | **有参考基准**               | label 出图                   | GT 标注 / 基线模型输出 / 人工调好的配置 |
| 2 | **有量化指标**               | Y-PSNR、ΔE00、acutance      | mAP、recall、IoU、延迟、重投影误差      |
| 3 | **有大量旋钮**               | 9 个白名单模块，几十个参数族 | 阈值、外参、位宽、融合权重、增强强度    |
| 4 | **每次试都要花钱**           | 一次 ISP ≈ 1 分钟           | 一次评测几分钟 / 一次训练几小时         |
| 5 | **旋钮有死有活、有量化台阶** | 一半参数 inert               | 某层量化不敏感、某阈值有台阶            |
| 6 | **"好"的定义可以代码化**     | 三类 gate                    | AP 不降 + 延迟达标 + 小目标不退化       |

**判定口诀**：*有标准答案、要试很多次、每次试都贵、好≠单一分数* → 就是这套方法的目标问题。

### 5.2 迁移映射（视觉感知相关）

| 领域                          | 参考基准      | 硬门（及格）                            | 盲区护栏                                   | 参数族 `[lo,hi,grid]`                                    |
| ----------------------------- | ------------- | --------------------------------------- | ------------------------------------------ | ---------------------------------------------------------- |
| **检测后处理**          | GT 标注       | mAP 不降 / recall↑                     | 小目标 AP、FP/frame、框抖动、per-class P/R | `score_thr[0.05,0.60,0.05]`、`nms_iou[0.35,0.75,0.05]` |
| **多目标跟踪**          | MOT GT        | MOTA / IDF1 不降                        | ID switch、碎片化、轨迹断裂                | `max_age`、`hit_thresh`、`iou_gate`、`kalman_q/r`  |
| **低照度/降噪前处理**   | 下游任务指标  | 夜间 recall↑**且** 白天 mAP 不降 | 纹理抹平、伪彩、振铃                       | 增强强度、gamma、去噪 level                                |
| **量化/剪枝部署**       | FP32 输出     | mAP 掉≤1%**且** 延迟≤33ms       | per-layer 敏感度、小目标掉点               | 位宽、跳层、算子选择、batch                                |
| **3D 感知 / occupancy** | 点云 GT       | 深度 RMSE、Occ IoU 不降                 | δ<1.25、Chamfer、远处稀疏区               | voxel size、depth bin、BEV 分辨率                          |
| **多传感器融合**        | 融合后 GT     | 融合 mAP 不降                           | 投影一致性、时序抖动                       | 外参微调、δt、融合权重                                    |
| **相机外参校准**        | 车道线/消失点 | 重投影误差 ≤ x px                      | 横向偏差、pitch 漂移、远处放大             | pitch/roll/yaw/height/focal                                |
| **自动标注流水线**      | 人工 GT 子集  | 一致率 ≥95%                            | 漏标率、定位误差、类别混淆                 | VLM 阈值、tile 重叠、NMS                                   |

### 5.3 例：检测后处理阈值调优（最像 ISP 的一个）

**目标文件**

```json
{
  "goal": "night recall up without daytime regression",
  "gates": {
    "day":   { "mAP50_95": ">= baseline", "per_class_drop": "<= 0.5" },
    "night": { "recall": ">= baseline + 0.03" },
    "cost":  { "latency_ms": "<= 35", "fp_per_frame": "<= baseline * 1.2" }
  },
  "guardrails": {
    "small_obj_ap":    ">= baseline - 0.02",
    "box_jitter_std":  "<= 1.5 px",
    "id_switch":       "<= baseline"
  }
}
```

**参数族**

```json
["night_score_thr",   ["postproc", 0.05, 0.60, 0.05]]
["nms_iou",           ["postproc", 0.35, 0.75, 0.05]]
["small_obj_scale",   ["postproc", 0.80, 1.20, 0.10]]
```

**坑与 ISP 完全同构**

| ISP 的坑                                   | 感知的同款坑                                              |
| ------------------------------------------ | --------------------------------------------------------- |
| AWB 拧 ±5 没反应 → 误判死旋钮            | `score_thr` 0.50→0.51，AP 一动不动 → 误判"阈值不敏感" |
| demosaic 量化台阶（24→63 不变，64 突变）  | 某些阈值在 AP 上是阶梯状，网格太细全是重复点              |
| `auto_curve.contrast` 只能动活跃 BV slot | per-class 阈值只对某些类有效，全局改会伤其他类            |
| 单参数不够要模块内补偿                     | 光调 `score_thr` 不行，得配 `nms_iou` 一起动          |

### 5.4 移植时：改 3 个，保 5 个

| 模块                                | 动作               | 说明                                                                       |
| ----------------------------------- | ------------------ | -------------------------------------------------------------------------- |
| `evaluation`                      | **重写**     | 换指标、换 gate、换"六维画像"（改成白天/夜间/小目标/远处/遮挡/时序）       |
| `knowledge`                       | **重灌**     | 症状词表换成领域的（漏检/误检/框抖/ID switch…），L2 指南换成你的 playbook |
| `execution`                       | **换执行器** | `apply --verify` + `run-isp` → `apply config` + `run_infer_eval`  |
| `judge`                           | **保留**     | 只吃数字的裁判，与领域无关                                                 |
| `planner`（步长链）/ `liveness` | **保留**     | 只改 spec 文件（`[lo,hi,grid]`）                                         |
| `stall_doctor`（死胡同协议）      | **保留**     | 五类取证换成你的取证方式                                                   |
| `retrospective`                   | **保留**     | 轨迹分析、经验沉淀、改进 backlog 全通用                                    |
| `testing`                         | **保留**     | 契约测试/插拔测试骨架通用                                                  |

### 5.5 感知域额外要小心的三个坑

| 坑                          | 表现                                          | 对应机制                                                              |
| --------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| **指标噪声大**        | 小验证集上 AP 抖 ±1~2%，以为调好了其实是噪声 | `CONFIRM_ACCEPT=2`（accept 要重复评估确认）+ 自动回滚               |
| **单次评估贵**        | 一次评测几分钟，试 20 次就一下午              | 预算 +`TOPK` 并行候选 + **小 proxy**（子集评测/短训先筛）     |
| **过拟合到某个 clip** | A 场景调好的阈值，B 场景崩                    | `transferability` 分级 + 跨会话族账本降权 + "单场景特例不许标 high" |

---

## 6. 从零搭一个的最小步骤

1. **定参考基准**：谁能当"标准答案"？（GT / 基线输出 / 人工配置）
2. **定指标三层**：
   - 硬门（判过没过，**明确 OR 还是 AND**）
   - 诊断特征（判往哪使劲，每维至少 3 个特征，别用单一标量）
   - 盲区护栏（主指标看不见的破坏，至少补 3 条）
3. **定症状词表**：13 类左右规范症状 + aliases + `want` 方向 + 对应维度
4. **定参数族规格**：`[lo, hi, grid]`，`grid` 必须实测（能被硬件量化吞掉的值不能当格）
5. **定探针方式**：怎么 1 次判定"这旋钮是活的"（极值 / 开关 0-1 / 换方向参数名）
6. **搭最小闭环**：`量 → 查 → 写 plan → 试 → 判 → 记账`，先跑通单旋钮
7. **加裁判**：独立的、只吃数字的 `judge`（不许 LLM 自由裁量）
8. **加沉淀**：每个 accept/reject 都写 trace → 定期复盘成条目 → 必须带 evidence
9. **加预算与白名单**：不给上限的自动搜索一定会烧穿

---

## 7. 附录

### 7.1 命令速查

```bash
# 环境自检
python modules/execution/scripts/check_env.py --check-vl

# 三输入接入（自动勘察/配对/基线/填宪章）
python modules/execution/scripts/cpipe_prep.py init --base-dir <工程> --raw <RAW> --label <LABEL>

# 六维差异 / 颜色报告
python modules/evaluation/cli.py profile6     --image <CUR> --label <LABEL> --out p6.json
python modules/evaluation/cli.py color-report --image <CUR> --label <LABEL>

# 知识检索（三源合并）
python modules/knowledge/cli.py query --platform cpipe1952 --module DRC --symptom 欠曝 --topk 8

# 写参必须 verify；ISP 只能经它跑
python modules/execution/scripts/cpipe_bo.py apply   --params '<JSON>' --verify
python modules/execution/scripts/cpipe_bo.py run-isp --base-dir ... --raw ... --out ...

# 出口判定 + 裁判复核
python modules/evaluation/cli.py checkpoint --session <S> --scope brightness,color,detail
python modules/judge/cli.py review --session <S> --claim done

# 复盘沉淀
python modules/retrospective/cli.py analyze  --session <S> --out retro.json
python modules/retrospective/cli.py sediment --report retro.json

# 知识库维护
python knowledge/tools/validate_kb.py
python knowledge/tools/build_manifest.py
```

### 7.2 关键文件索引

| 文件                                            | 内容                                                             |
| ----------------------------------------------- | ---------------------------------------------------------------- |
| `SKILL.md`                                    | 编排层：模块地图、硬红线、路由、主循环                           |
| `MODULES.md`                                  | 7 模块职责与插拔规则                                             |
| `CHECKLIST.md`                                | **验收标准（当需求文档读）**                               |
| `PORTING.md`                                  | 换机清单 + 全部环境变量表                                        |
| `knowledge/ROUTING.md`                        | 症状 → 模块 → 文件路由表                                       |
| `knowledge/symptom_vocabulary.json`           | 规范症状词表（13 类 + 别名）                                     |
| `knowledge/kb_manifest.json`                  | 文档机器索引（path/platform/type/module/keywords/sha256）        |
| `.../reference/param_steps.json`              | 每参数建议步长（liveness/coarse/fine/quantum）                   |
| `.../reference/metric_targets.json`           | **冲刺目标 + 五个盲区的护栏设计**                          |
| `family_search_spec.json`                     | 参数族搜索空间 `[dim, family, [module,lo,hi,grid]]`            |
| `modules/execution/planner.py`                | 步长链（`_gate_gap` / `_trust_step` / `choose_candidate`） |
| `modules/execution/liveness.py`               | 活性探针                                                         |
| `modules/execution/param_effects.py`          | 斜率反算剂量（可选）                                             |
| `modules/evaluation/scripts/profile6.py`      | 六维画像实现                                                     |
| `modules/evaluation/scripts/color_metrics.py` | ΔE00 / D_color / J_v3                                           |
| `modules/judge/api.py`                        | 裁判（纯指标）                                                   |
| `knowledge/tools/kb_query.py`                 | 检索打分实现                                                     |

### 7.3 一句话记住每件事

- **门**：OR 还是 AND，决定差距怎么算
- **gap**：不是一个分数，是"还差百分之几"的比值
- **格**：硬件最小可动单位，比它小的动作等于没做
- **探针**：唯一的"大力"，花 1 次省 15 轮
- **步长**：格 × 反馈倍数，再被裕度和硬顶夹住
- **neutral 不减半**：减半会掉进假反馈死循环
- **裁判**：只吃数字，不听解释
- **VLM**：只说"哪儿别扭"，不判分
- **平台期**：不是出口，只允许换道
- **经验**：必须带证据和可迁移性，否则是噪音
