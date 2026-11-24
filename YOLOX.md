# YOLOX 交通灯 / 交通牌检测工程总结

> **工程路径**：`\\100.102.76.225\cameraall_disk1\n50063324\yolox`  
> **总结范围**：仓库定位、网络结构、相对官方 YOLOX 的改动、训练/评测/部署流程、设计动机、长尾问题、初级算法工程师学习指南。  
> **阅读提示**：每个模块尽量同时给“专业描述”和“通俗解释”；带“⚠️”的是容易误解或需要特别注意的点。  
> **一句话定位**：这不是官方 YOLOX 的简单魔改，而是把 YOLOX 拆进 **UVPSandBox 多任务框架**，用 **AscendNet + 轻量 FPN + 可共享的解耦 Head** 做交通灯/交通牌的四个任务联合检测。

---

## 目录

1. [先看结论](#1-先看结论)
2. [工程概览](#2-工程概览)
3. [官方 YOLOX 基线](#3-官方-yolox-基线)
4. [整体网络结构](#4-整体网络结构)
5. [逐模块详解](#5-逐模块详解)
6. [训练流程](#6-训练流程)
7. [评测与指标](#7-评测与指标)
8. [相对官方 YOLOX 的改动清单](#8-相对官方-yolox-的改动清单)
9. [设计动机：问题 → 策略 → 落点 → 收益 → 代价](#9-设计动机问题--策略--落点--收益--代价)
10. [长尾专题](#10-长尾专题)
11. [初级感知算法工程师学习指南](#11-初级感知算法工程师学习指南)
12. [术语表](#12-术语表)
13. [代码与文档索引](#13-代码与文档索引)
14. [一页纸总结](#14-一页纸总结)
15. [附录：常见误解](#15-附录常见误解)

---

| 字段 | 含义 | 模型里怎么用 | 类型 |
|---|---|---|---|
| TrafficLight_v01 | 结果格式/版本名 | 不参与计算 | 元数据 |
| size | 原图宽高（3840×2160） | 坐标缩放/还原 | 元数据 |
| list | 目标列表 | 每个元素一个交通灯 | 元数据 |
| outbox | 整个灯箱/灯组外框 | 外框检测任务 | 检测目标 |
| inboxList | 灯箱内每个灯球/箭头/数字 | 内框检测任务 | 检测目标 |
| bndbox | 框坐标 xmin/ymin/xmax/ymax | 4 通道回归 | 回归 |
| color | 颜色编码 | 分类通道 | 分类 |
| shape | 内灯形状/功能 | 分类通道 | 分类 |
| countdown_in_one | 个位数字 | 分类通道 | 分类 |
| countdown_in_ten | 十位数字 | 分类通道 | 分类 |
| receiver | 控制对象（机动车/行人） | 外框分类 | 分类 |
| direction | 朝向（正面/侧面/背面） | 外框分类 | 分类 |
| occluded | 遮挡程度 | 外框分类 | 分类 |
| truncated | 是否被图像边界截断 | 当前训练脚本未用 | 元数据 |
| bulb_num | 灯球数量 | 过滤/合并用 | 元数据 |

交通灯的属性天然是多标签的：一个灯可能同时是“红灯”和“左转箭头”，也可能同时是“绿灯”和“右转箭头”。如果用 Softmax，模型会被迫在这些属性之间做“互斥选择”，无法表达“红灯 + 左转箭头”这种组合，语义上就错了。BCE 允许每个属性独立激活，才与真实世界的交通灯语义对齐。---所以交通灯这种就要使用BCE

## 1. 先看结论

### 1.1 这个工程到底是什么

- 仓库根目录叫 `yolox`，但代码主体是华为 **UVPSandBox 4.1** 多任务感知沙盒。
- YOLOX 只是其中的一个 task：`tasks/YOLOXTraffic`。
- 当前 1080p 主训练配置在：

  ```text
  tasks/YOLOXTraffic/uvpsandbox/configs/train_traffic_ascendnet_fpn_ncasd/
  ```

- 四个任务一起训练：

  | 任务 | 配置文件 | 类别数 | 说明 |
  |---|---|---|---|
  | `tl_outer` | `tl_outer_...yaml` | 12 | 交通灯外框：颜色、功能、朝向、遮挡 |
  | `tl_inner` | `tl_inner_...yaml` | 38 | 交通灯内框：颜色、形状、个位、十位 |
  | `tl_pose` | `tl_pose_...yaml` | 12 + 2 通道角度 | 交通灯姿态/朝向 |
  | `ts` | `ts_aux_...yaml` | 69 | 交通牌分类检测 |

### 1.2 三句话概括改动

1. **骨干重写**：官方 `CSPDarknet` → `AscendNet`（3 层 conv stem + 16 个分组残差块，ReLU，无 SPP）。
2. **增加颈部**：官方 Megvii 版 YOLOX 没有 neck，这里加了轻量 FPN（1×1 横向卷积 + 上采样相加 + 3×3 平滑）。
3. **头部解耦并共享**：公共卷积 + 共享回归分支 + 任务私有分类分支；分类从“平铺类别”改成“多标签 + 属性分组使用方式”，并额外挂 depth / pose / aux 分支。

### 1.3 三个重要提醒

> ⚠️ **提醒 1：本项目没有“双框 / QFL / amodal box”。**  
> 你之前看的 Mono3D 总结里有 visible box + full box 双框联合监督、QFL 质量分、按类回归、Car/Truck/Tram 合并 NMS，这些**不是**这个 YOLOX 工程的做法。本项目的遮挡处理是“把遮挡做成外框的一个属性标签”，不是回归完整框。

> ⚠️ **提醒 2：项目没有给逐项消融实验。**  
> `eval/benchmark.md` 只有最终指标，没有“去掉 FPN 掉多少、换回 CSPDarknet 掉多少”的对照。因此本文说“为什么能提升”时，讲的是**结构机制上的合理解释**，不是实验证明。面试/汇报时不要把“机制合理”说成“我验证过涨点”。

> ⚠️ **提醒 3：配置名里的 `insl3` 不是网络层。**  
> 代码里没有叫 `insl3` 的模块，它属于内部模板/部署版本命名。真正决定结构的是 YAML 里的 `net_parts`，看结构请直接看四个任务 yaml 的 `network.net_parts`。

---

## 2. 工程概览

### 2.1 仓库地图

| 路径 | 作用 | 初学者是否需要看 |
|---|---|---|
| `launch/train_test.py` | 训练/评测入口，初始化分布式、加载配置 | 知道即可 |
| `launch/reform.py` | 模型转换/导出入口（ONNX 等） | 暂时不用 |
| `uvpsandbox/framework.py` | 多任务框架：建网络、建 dataloader、DDP、loss 反传、保存 | 只看“子网共享”部分 |
| `uvpsandbox/module_container.py` | 按拓扑执行子网、按 `net_name` 存取权重 | 暂时不用 |
| `uvpsandbox/weighter.py` | 多任务 loss 平衡（ls / gradnorm / imtl 等） | 知道即可 |
| `infrastructure/base_module.py` | 子网基类：参数初始化、权重衰减、冻结 BN、DDP 配置 | 暂时不用 |
| `components/model/backbone/ascendnet.py` | **AscendNet 骨干** | 必看 |
| `components/model/decoder/fpn.py` | **FPN 颈部** | 必看 |
| `tasks/YOLOXTraffic/models/heads/head_common.py` | **解耦 Head**（公共卷积/提取分支/预测头/深度头/属性头） | 必看 |
| `tasks/YOLOXTraffic/models/backbones/csp_darknet.py` | 官方风格 CSPDarknet（当前主配置未用） | 对比时看 |
| `tasks/YOLOXTraffic/loss_infer/vision_pv_tl.py` | 交通灯检测 loss、SimOTA 目标分配 | 必看 |
| `tasks/YOLOXTraffic/loss_infer/vision_pv_tl_dep.py` | 带深度的交通灯 loss | 选看 |
| `tasks/YOLOXTraffic/loss_infer/vision_pv_tl_pose.py` | 姿态/角度 loss | 选看 |
| `tasks/YOLOXTraffic/dataset/pv_dataset.py` | 数据加载、`height_used` 裁剪、dim4 | 必看 |
| `tasks/YOLOXTraffic/uvpsandbox/configs/` | 各版本训练配置 | 必看 |
| `pv_eval_yamls/config_ascend/` | 推理/评测配置 | 选看 |
| `configs/tasks/common.yaml` | 通用训练超参 | 选看 |
| `configs/deploy/export_onnx.yaml` | ONNX 导出配置 | 暂时不用 |
| `eval/benchmark.md` | 评测数据集与最终指标 | 必看 |
| `docs/TensorRt_deployment.md`、`docs/MDC_deployment.md` | TensorRT / MDC 部署说明 | 暂时不用 |
| `pretrained/ft.pth` | 四任务微调 checkpoint | 可用于验证结构 |
| `eval/Iter-5000_StepShot_narrow_0415.pth` | 另一个迭代的 checkpoint | 选看 |

### 2.2 数据概览（按 README 记录）

| 数据 | 规模/说明 |
|---|---|
| 交通灯基础数据 | 上海、深圳采集，约 260,650 帧 |
| 交通灯 corner case | 逆光、数字灯、右转/掉头、高速、假数字、分离倒计时、特殊灯型、黑灯、绿色横排等，多批共约十几万帧 |
| `add_20230202` | 29,747 帧，含北京、广州、恶劣天气、重庆黄闪、右转自行车、东陆路、灯笼误检等 |
| 交通灯姿态 | 侧视相机采集，约 285,166 帧 |
| 交通牌检测 | 101,175 帧 |
| 交通牌 `detection_v01` | 47,528 帧 |
| 交通牌假数据 | 27,084 帧 |
| 交通牌 `pv_new` | 6,009 帧 |
| 交通牌仿真 | 30,977 帧 |
| 交通牌 ETC | 23,894 帧 |

> **通俗解释**：数据不是“一个数据集”，而是“基础数据 + 一堆 corner case 补丁”。交通灯项目最贵的不是模型，而是这些针对坏场景反复补的数据。

### 2.3 标注字段与属性

标注侧（LabelJson / `tools/create_traffic_light_*.py`）包含多属性字段，例如：

- 外框：颜色（黑/红/黄/绿）、接收对象（机动车/行人）、朝向（正面/侧面/背面）、遮挡（无遮挡/部分遮挡）；
- 内框：颜色、形状（圆/直行/左转/右转/掉头/数字/非机动车等）、个位数字、十位数字、倒计时；
- 还有 `isValid`、`blinking`、`receiver` 等业务字段。

模型侧不是把这些字段硬塞进一个类别，而是组织成 **多标签分类 + 分组使用**。具体映射见 `tools/create_traffic_light_outer.py`、`tools/create_traffic_light_inner.py`。

---

## 3. 官方 YOLOX 基线

在看改动前，先回顾官方 Megvii YOLOX 的基本盘。

### 3.1 官方结构

```text
输入 640×640
   │
CSPDarknet
   │  Focus stem（space-to-depth + 1×1）
   │  BaseConv + CSPLayer × 4 个 stage
   │  P5 带 SPPBottleneck
   │  激活 SiLU
   ▼
P3 / P4 / P5（stride 8/16/32）
   │
YOLOXHead（每个 level）
   │  1×1 stem
   │  cls 分支：2 个 3×3
   │  reg 分支：2 个 3×3
   │  3 个 1×1：bbox(4) / obj(1) / cls(N)
   ▼
Decode + NMS
```

### 3.2 官方 YOLOX 的关键设计

| 设计 | 作用 |
|---|---|
| Anchor-Free | 不预设 anchor，每个格子直接回归框，减少超参和计算 |
| Decoupled Head | 分类和回归用不同卷积分支，缓解“分类要平移不变、回归要位置敏感”的冲突 |
| SimOTA | 动态正样本分配：按代价给每个 GT 选 top-k 候选，并动态决定正样本数 |
| IoU Loss | 框回归用 IoU 系列损失，直接优化重叠度 |
| BCE | 分类和 objectness 都用二分类交叉熵 |
| 无 NMS 之外的复杂后处理 | 解码 + 置信度筛选 + NMS |

> ⚠️ **关于 neck**：Megvii 官方实现里，backbone 的 P3/P4/P5 是**直接**进入 `YOLOXHead` 的，没有独立 neck；mmdetection 复现版才提供 `YOLOXPAFPN`。这个仓库里两种代码都有，但当前 1080p 主配置用的是自己写的轻量 FPN。

> **通俗解释**：官方 YOLOX 像“一个熟练工从看片到画框一条龙”；这个项目把熟练工拆成“眼睛（backbone）→ 参谋（neck）→ 判读员（head）”，方便多任务协作和部署。

---

## 4. 整体网络结构

### 4.1 结构图

```text
输入：704×1600（BGR，/255）
   │
   ▼
AscendNet 骨干
   ├── stem：3×3 s2 → 3×3 s1 → 3×3 s2，输出 /4、64 通道
   ├── stage1：2 个 AscendBlock，/4、64 通道（不输出给 FPN）
   ├── stage2：3 个 AscendBlock，/8、128 通道  → C3
   ├── stage3：3 个 AscendBlock，/16、256 通道 → C4
   └── stage4：8 个 AscendBlock，/32、512 通道 → C5
   │
   ▼
轻量 FPN
   ├── FPNLateral：1×1 + BN + ReLU 统一到 128；自顶向下上采样相加
   └── FPNConv：每层 3×3 + BN + ReLU 平滑
   │
   ▼
三个 128 通道特征（/8、/16、/32）
   │
   ▼
Head（按任务组合）
   ├── 共享公共卷积 head_tl_com（1 个 3×3 / level）
   ├── 任务私有 cls 分支（2 个 3×3）
   ├── 共享 reg 分支（2 个 3×3）
   ├── 1×1 预测头：bbox(4) / obj(1) / cls(N)
   ├── 深度头：1×1 → 1 通道
   └── 姿态头：1×1 → 2 通道（sin/cos 角度）
   │
   ▼
Decode + NMS（按任务）
```

### 4.2 张量形状（以 704×1600 输入为例）

| 阶段 | 分辨率 | 通道 | stride | 说明 |
|---|---|---|---|---|
| 输入 | 704×1600 | 3 | 1 | BGR，/255 |
| stem | 176×400 | 64 | 4 | 3 层 conv |
| stage1 | 176×400 | 64 | 4 | 2 个 block，不输出 |
| C3 | 88×200 | 128 | 8 | 小目标主要看这层 |
| C4 | 44×100 | 256 | 16 | 中等目标 |
| C5 | 22×50 | 512 | 32 | 大目标/语义 |
| FPN 输出 | 88×200 / 44×100 / 22×50 | 128 | 8/16/32 | 三层统一 128 |
| Head 输出 | 同上 | `4 + 1 + N` | 同上 | 每个格子一组预测 |

> **通俗解释**：stride 就是“格子有多粗”。stride 8 表示原图每 8×8 像素对应特征图上一个格子；格子越细，越容易看到远处的小灯；格子越粗，每个格子“看到”的原图范围越大，越懂语义。

### 4.3 四个任务的共享关系

共享规则：**同名 `net_name` 且 `specs` 完全一致**，框架就会复用同一个子网；`uvpsandbox/framework.py` 会断言共享配置一致。

| 子网 | 共享范围 | 作用 |
|---|---|---|
| `backbone` | 四任务全共享 | 通用特征提取 |
| `fpnlat` | 四任务全共享 | 多尺度融合 |
| `fpnconv_c3/c4/c5` | 四任务全共享 | 每层平滑 |
| `head_tl_com` | 外框 / 内框 / 姿态 | 红绿灯公共卷积 |
| `head_tl_extract_reg` | 外框 / 内框 / 姿态 | 共享回归特征 |
| `head_tl_outer_extract_cls` | 外框 / 姿态 | 外框分类特征 |
| `head_tl_outer_predict` | 外框 / 姿态 | 外框预测头 |
| `head_tl_inner_extract_cls` | 内框私有 | 内框分类特征 |
| `head_tl_inner_predict` | 内框私有 | 内框预测头 |
| `head_tl_outer_depth_predict` | 外框私有 | 深度预测 |
| `head_tl_pose` | 姿态私有 | 角度预测 |
| `head_ts_conv` + `head_ts` | 交通牌私有 | 交通牌检测 |

> **通俗解释**：四个任务一起上课，**基础课（主干、FPN、回归画框）一起上**；**专业课（颜色/形状/数字/姿态/交通牌分类）分开上**。这样省老师（参数），但基础课老师讲偏了，四个学生一起偏。

### 4.4 checkpoint 证据

`pretrained/ft.pth` 的 state_dict key 可以直接验证结构：

```text
backbone.layers.0 ~ backbone.layers.15     # 16 个 AscendBlock
backbone.stem.0 / 3 / 6                    # 3 层 stem 卷积
fpnlat.lateral_convs.0/1/2                 # 3 层横向卷积
fpnconv_c3 / fpnconv_c4 / fpnconv_c5       # 每层 3×3 平滑
head_tl_com                                # 公共卷积
head_tl_extract_reg                        # 共享回归分支
head_tl_outer_extract_cls / inner_extract_cls
head_tl_outer_predict / inner_predict
head_tl_outer_depth_predict                # 深度头
head_tl_pose                               # 姿态头
head_ts_conv / head_ts                     # 交通牌头
```

> 这比只看代码更可靠：代码里有很多“可选但没启用”的模块，checkpoint 的 key 才是实际训练结构。

---

## 5. 逐模块详解

### 5.1 Backbone：CSPDarknet → AscendNet

#### 5.1.1 它是什么（专业）

Backbone 负责把输入图像映射成多尺度特征图。它不输出类别和框，只输出“通用隐式特征”。

AscendNet 的结构：

- **stem**：`Conv3×3 s2` → `Conv3×3 s1` → `Conv3×3 s2`，输出 `/4`、64 通道；
- **主体**：16 个 `AscendBlock`，每个 block 是：
  - `3×3 groups=2` 卷积 → BN → ReLU；
  - `1×1` 卷积 → BN；
  - 残差相加 → ReLU；
  - 中间通道 = `out_channel × exp_ratio / 2`；
- **stage 配置**用字符串编码：
  - `widths: 11-211-121-11111121`
  - `exp_ratios: 11-211-211-22222222`
  - `-` 表示 stride 2 下采样，数字表示通道倍率；
- **stage 划分**：2 / 3 / 3 / 8 个 block；
- **输出**：C3 `/8` 128、C4 `/16` 256、C5 `/32` 512；
- **激活**：ReLU；
- **没有 SPP**；
- **BN**：`momentum=0.03, eps=0.001`。

#### 5.1.2 它是什么（通俗）

Backbone 就是网络的“眼睛”。

它把一张大图压缩成三张“特征地图”：

- 第一张（C3）格子最细，适合看远处的小灯；
- 第二张（C4）中等；
- 第三张（C5）格子最粗，但每个格子“看到的范围”最大，最懂“这是灯还是广告牌”。

每个 AscendBlock 可以理解成：

> 先用 3×3 分组卷积看看周围长什么样 → 用 1×1 卷积把通道信息整合一下 → 再加上“原来的自己”（残差），保证不学坏。

#### 5.1.3 与官方对比

| 项目 | 官方 CSPDarknet | 本项目 AscendNet |
|---|---|---|
| Stem | Focus（space-to-depth + 1×1），/2 | 3 层普通卷积，/4 |
| 主体 | BaseConv + CSPLayer（split-transform-merge） | 16 个 AscendBlock（分组 3×3 + 1×1 + 残差） |
| 激活 | SiLU | ReLU |
| P5 处理 | SPPBottleneck + CSP | 无 SPP，靠 8 个 block 堆感受野 |
| 输出 | P3/P4/P5 | C3/C4/C5 |
| 通道（YOLOX-s 宽度） | 128/256/512 | 128/256/512 |
| 预训练权重 | 官方 COCO 权重 | 不兼容，需重新训练/加载本项目 checkpoint |

#### 5.1.4 为什么这么改

从结构选择看，主要动机是：

1. **部署友好**：分组卷积 + ReLU 比标准卷积 + SiLU 更便宜，适合大输入和车载芯片。
2. **感受野**：P5 堆 8 个 block，让深层“看得更远”，对远处红绿灯/交通牌的上下文有帮助。
3. **保留浅层**：C3 保留 `/8` 高分辨率，给 FPN 和 head 提供小目标细节。
4. **控制参数量**：16 个轻量 block，比官方 CSP 堆叠更适合 704×1600 大图。

#### 5.1.5 代价与注意

- 和官方 YOLOX 预训练权重**不兼容**，不能直接加载官方 CSPDarknet 权重；
- 没有 SPP，少了一个多尺度池化分支，可能损失一点大感受野/多尺度鲁棒性；
- ReLU 比 SiLU 便宜，但可能掉点；
- **stage1 在 `/4` 上算了 2 个 block，但没有输出给 FPN**，这部分计算是“白算”的；
- 项目没有给“换回 CSPDarknet 掉多少”的消融，不能量化收益。

> **通俗解释**：这就像把“专业单反”换成“手机多摄”：更轻、更快、更适合装进车里，但极限画质不一定比单反强。它不是“全面升级”，而是“面向部署的取舍”。

> **代码位置**：`components/model/backbone/ascendnet.py`；当前配置见四个任务 yaml 的 `net_parts` 第一项。

---

### 5.2 Neck：官方没有，这里加了一个轻量 FPN

#### 5.2.1 它是什么（专业）

Neck 负责多尺度特征融合。这个项目的 FPN 分两步：

1. **FPNLateral**
   - 每层 `1×1 Conv + BN + ReLU`，把 128/256/512 统一到 128；
   - 自顶向下：`laterals[i-1] = laterals[i-1] + upsample(laterals[i])`；
   - **只有 top-down，没有 bottom-up，也没有 CSP 融合**。
2. **FPNConv**
   - 每个 level 再接 `3×3 Conv + BN + ReLU` 做平滑；
   - 输出三个 128 通道特征。

配置里 `fpnlat`、`fpnconv_c3/c4/c5` 是四个独立 `net_parts`，不是一个大模块，方便导出/部署时拆子图。

#### 5.2.2 它是什么（通俗）

Neck 是“三层地图互相抄作业”。

- C3 细节多，但看不懂语义；
- C5 语义强，但位置粗；
- FPN 把 C5 放大后加到 C4，再把 C4 放大后加到 C3；
- 这样 C3 既“看得细”，又“懂语义”。

`1×1` 卷积像“把三层地图翻译成同一种语言”，上采样像“把粗地图放大”，`3×3` 卷积像“把放大后的马赛克磨平”。

#### 5.2.3 为什么加

- 704×1600 的长条图里，红绿灯/交通牌经常很小；
- 小目标的信息主要在 C3（`/8`），但 C3 语义弱；
- FPN 让 C3 同时拿到 C4/C5 的语义，提升小目标判别；
- 统一 128 通道，方便后面的 head 共享。

#### 5.2.4 代价

- 额外计算和参数；
- 只有 top-down，没有 bottom-up，理论上不如 PAFPN/BiFPN；
- 代码里还保留了 `YOLOXPAFPN`、`BiFPNUp/BiFPNDown`、`DyHead` 等可选 neck；
- `pv_eval_yamls/config_ascend/` 里有些评估配置用的是 BiFPN，不是训练主配置的 FPN，**看指标时要确认 yaml 对应哪套结构**。

> **通俗解释**：FPN 不是“必须”，但在“小目标 + 多尺度”场景几乎是标配。它像给近视的判读员配了一副“既能看近又能看远”的眼镜。

> **代码位置**：`components/model/decoder/fpn.py`；配置见 `net_parts` 里的 `FPNLateralAug`、`FPNConvAug`。

---

### 5.3 Head：从“一个整体”拆成“共享 + 私有”

#### 5.3.1 它是什么（专业）

Head 在每个特征图格子上输出预测。官方 `YOLOXHead` 是一个整体；本项目拆成三类模块：

- `YOLOXHeadCom`：每层 1 个**共享** 3×3 公共卷积，输出 128 维公共特征；
- `YOLOXHeadExtractConvs`：2 个 3×3 的提取分支，分 cls 和 reg；
- `YOLOXHeadPredictor`：3 个 1×1，输出 `cls / reg / obj`；
  - reg 是 **class-agnostic** 的 4 通道（所有类别共用一套框回归）；
  - obj bias 初始化为 -4.5；
- 激活 ReLU，通道 128。

共享关系见 4.3 节。

#### 5.3.2 它是什么（通俗）

Head 是“每个格子上的判读员”。

- 先由“公共老师”讲一遍基础课（`head_tl_com`）；
- 然后“分类老师”判断这是什么（cls 分支）；
- “画框老师”判断四条边在哪（reg 分支）；
- 最后三个 1×1 像“填写答题卡”：框、有没有目标、类别。

拆开的好处是：

- 分类和画框可以各学各的；
- 多个任务可以共用“公共老师”和“画框老师”；
- 每个任务只保留自己的“分类老师”。

#### 5.3.3 与官方对比

| 项目 | 官方 YOLOXHead | 本项目 |
|---|---|---|
| 组织 | 一个整体模块 | 拆成 Com / Extract / Predictor 子网 |
| 每层卷积 | 1×1 stem + 2×3×3 cls + 2×3×3 reg | 1 个共享 3×3 + 2×3×3 cls + 2×3×3 reg |
| 激活 | SiLU | ReLU |
| 参数共享 | 无（一任务一模型） | 公共卷积/回归分支跨任务共享 |
| 额外分支 | 无 | depth / pose / aux / keypoint |
| 回归通道 | 4（class-agnostic） | 4（class-agnostic） |

#### 5.3.4 为什么可能提升

- 共享公共卷积和回归分支，减少参数，提升多任务训练效率；
- 私有分类分支保留任务差异，降低“颜色/形状/数字/姿态”互相干扰；
- 参数效率高，部署更小；
- 公共特征被多个任务同时监督，可能学到更通用的表征。

#### 5.3.5 代价

- 共享分支的梯度会混在一起，一个任务数据差会拖累其他任务；
- 需要 loss 权重、采样比例来平衡；
- 共享层一旦学偏，所有任务一起偏；
- 拆分后结构复杂，调试和导出更麻烦；
- 没有消融，无法量化“共享”到底贡献多少。

> **通俗解释**：多任务的核心不是“多写几个 loss”，而是“**哪些层共享、哪些层私有**”。共享越深，任务冲突越大；共享越浅，参数越省但任务越独立。

> **代码位置**：`tasks/YOLOXTraffic/models/heads/head_common.py`；配置见 `net_parts` 里的 `Head` 项。

---

### 5.4 分类与标签：从“12 选 1”到“多标签 + 分组使用”

#### 5.4.1 标签语义（专业）

外框 12 类：

```text
back, red, yellow, green, black, veh, non_veh, front, side, reversre, no_occupied, part_occupied
```

内框 38 类：

```text
red_inner, yellow_inner, green_inner, other_inner,
circle, straight, left, right, uturn, num, non_veh, left_uturn, other_shape,
no_num_in_one, vague_in_one, 0~9_in_one,
no_num_in_ten, vague_in_ten, 0~9_in_ten, mbai_in_ten
```

一个框可以同时有多个标签，例如：

- 外框：“红 + 机动车 + 正面 + 无遮挡”；
- 内框：“红 + 左转 + 个位 3 + 十位无”。

#### 5.4.2 实现方式（重要，容易误解）

> ⚠️ **模型输出仍然是扁平的 12/38 个分类通道，不是 4 个独立分类头。**

具体来说：

1. 分类头输出 `num_classes` 个通道（外框 12、内框 38、交通牌 69）；
2. loss 用 `CrossEntropyLoss(use_sigmoid=True)`，即**逐通道 BCE**；
3. 一个框的多个标签通过 `box2multi_labels` 聚合，再把多个 one-hot 相加，得到 multi-hot 目标；
4. 分类目标还会乘上 `pos_ious`（IoU 软标签），这一点和官方 YOLOX 一致；
5. 配置里的 `attrs` 只影响两件事：
   - SimOTA 分配时只用 `attrs[0]`（第一组，外框是颜色 5 类）的分数算分类代价；
   - 分类 loss 最后除以 `len(attrs)` 做归一化；
6. 所以 `attrs` 更像“分组使用/统计约定”，不是“网络里真的分了 4 个 softmax”。

外框 `attrs: [5,3,2,2]`、内框 `attrs: [4,9,12,13]` 是按通道顺序切组；第一组分别是颜色 5 类、颜色 4 类。

> **通俗解释**：不是让模型做“12 选 1”的选择题，而是让它做 12 道“是/不是”的判断题；一个框可以同时勾选多个“是”。`attrs` 只是告诉训练器“第一组是颜色，分配正样本时先看颜色”。

#### 5.4.3 为什么这么设计

- 交通灯天然是多属性：颜色、功能、朝向、遮挡是正交的；
- 硬做成 12 类互斥，会把“红-机动车-正面”和“红-行人-侧面”当成完全不同的类，样本被切碎，长尾更严重；
- 多标签更符合真实语义，也能保留“部分可见”的信息；
- 分组后每组类别少，学习难度下降；
- 可以对数字灯、形状单独加权，缓解长尾。

#### 5.4.4 代价

- 标注和评估更复杂，要按属性算 AP/F1；
- 组间不平衡，某些组样本很少；
- SimOTA 只用第一组，可能忽略其他属性的难例；
- 多标签求和可能让某些组合过强；
- 不是“4 个独立分类头”，所以不能简单说“每个属性独立建模”。

> **代码位置**：`loss_infer/vision_pv_tl.py` 的 `_get_target_single`、`compute_loss`；`dataset/pv_dataset.py` 的标签加载。

---

### 5.5 额外分支：depth / pose / aux / keypoint

这些是官方 YOLOX 没有的，属于“检测头挂属性分支”。

| 分支 | 输入 | 输出 | 损失 | 作用 | 代价 |
|---|---|---|---|---|---|
| 深度 `YOLOXHeadDepthPredictor` | 共享 reg 特征 | 1 通道深度 | SmoothL1，只对 GT depth ≥ 1 | 给红绿灯估距 | 深度 GT 难得、误差大 |
| 姿态 `YOLOXHeadAttrPredictor` | 共享 reg 特征 | 2 通道 sin/cos | `atan2` 解角度后 SmoothL1，`attr_weight=10` | 估朝向/角度 | 角度周期性、标注一致性 |
| 交通牌 aux | 交通牌 cls 特征 | 61 类辅助分类 | 分类 loss | 训练时多一个监督 | 只在训练用，导出要去掉 |
| 关键点 `YOLOXKeypointHead` | 回归特征 | 2 通道关键点 | 关键点 loss | 停车杆任务 | 非主线 |

> ⚠️ 姿态头不是“2 类分类”。配置里 `NUM_CLASSES_POSE=2` 只是输出通道数；loss 里用 `atan2(y, x)` 把它解释成角度，所以它是**角度回归**。

> **通俗解释**：检测头像“主科老师”，这些分支像“附加作业”：深度、朝向、交通牌分类。附加作业能帮主科，但每加一项都要有标准答案（GT）和评分权重，不然会互相拖累。

---

### 5.6 输入与数据增强

#### 5.6.1 输入尺寸

- 配置里 `IMAGE_SCALE: [1024, 1600]` 是增强/填充画布；
- 数据加载最后 `height_used: 704`，会 `img[:704]` 裁掉下部；
- 实际送网是 **704×1600**；
- 部署文档里的推理配置也是 `704x1600`。

为什么：

- 宽图保留更多横向场景；
- 704 高度控制算力；
- 裁掉下部是因为交通灯主要在上方。

代价：

- 裁掉的目标会丢标签；
- 有些评估 yaml 没写 `height_used`，会按 1024×1600 评估，**训练/评估尺寸要对齐**。

#### 5.6.2 预处理

- `Resize(keep_ratio=True)` + `Pad`：等比缩放后填充，即 letterbox 式，不变形；
- `Normalize(mean=0, std=255, to_rgb=False)`：BGR + /255，和官方 YOLOX 一致；
- 数据加载把 `uint8` 转 `float32`。

#### 5.6.3 数据增强

| 增强 | 配置 | 作用 | 代价 |
|---|---|---|---|
| Mosaic | 4 图拼接 | 增加小目标和上下文 | 可能拼出不真实场景 |
| MixUp | 两图混合 | 提升泛化 | 目标边界可能模糊 |
| HSV | 随机色调/饱和度/亮度 | 光照鲁棒 | 过度会失真 |
| RandomAffine | 缩放 0.7~1.5，平移 border | 尺度/位置鲁棒 | 可能切掉目标 |
| RandomFlip | `flip_ratio: 0.0` | **不翻转** | 交通灯左右转箭头有方向语义，翻转会改标签 |

> **通俗解释**：数据增强就是“把一张图变着花样给模型看”。Mosaic 像把四张照片剪贴在一起，MixUp 像把两张照片叠影；不翻转是因为“左转箭头”翻过来就变成“右转箭头”，标签会错。

#### 5.6.4 长尾处理

- 内框配置有 `rebalanced_dataset`，对左转/右转/掉头等少数类过采样；
- 内框 loss 开启 `cls_weighted`，对形状/个位/十位乘 10~20 权重；
- 外框主要靠多标签和数据补充，没有单独的类别加权。

> **通俗解释**：少数类样本少，就多给它看几遍（过采样），或者它错了就罚得更重（加权）。

---

### 5.7 损失函数

#### 5.7.1 损失组成

交通灯检测（外框/内框）的核心损失：

| 损失 | 类型 | 作用 | 权重/说明 |
|---|---|---|---|
| `loss_cls` | BCE（sigmoid） | 分类/多标签 | 默认 1.0，最后除以 `len(attrs)` |
| `loss_obj` | BCE（sigmoid） | 有没有目标 | 默认 1.0 |
| `loss_bbox` | IoU Loss（square） | 框回归 | 默认 5.0 |
| `loss_dep` | SmoothL1 | 深度回归 | 只对 depth ≥ 1 |
| `loss_attr` | SmoothL1 | 姿态/角度 | `attr_weight=10` |

#### 5.7.2 SimOTA 目标分配

关键参数：

- `center_radius=2.5`；
- `candidate_topk=10`；
- `iou_weight=3.0`；
- `cls_weight=1.0`；
- 分类代价只用 `attrs[0]`（颜色组）的 sigmoid × objectness；
- 动态 k：每个 GT 选 top-k IoU 候选，按代价排序，动态决定正样本数。

> **通俗解释**：SimOTA 像“给每个目标挑学生”：先圈出目标中心附近的格子，再按“框 overlap 好不好 + 类别像不像”打分，最后每个目标挑几个最合适的格子当正样本。挑几个不是固定的，而是根据目标大小/重叠动态决定。

#### 5.7.3 训练稳定性

- bbox 解码时 `wh` clamp 到 `[-5, 10]`，防止 `exp` 爆炸；
- loss 前检查 NaN，若 NaN 则返回全 0 loss；
- 这些是版本记录里明确修过的训练稳定性问题。

> **通俗解释**：框的宽高是用 `exp` 算的，数值一大就会“爆炸”；clamp 像给油门加限位。NaN 保护像保险丝，检测到异常就跳过这一步，避免整个训练崩掉。

> **代码位置**：`loss_infer/vision_pv_tl.py`、`vision_pv_tl_dep.py`、`vision_pv_tl_pose.py`。

---

### 5.8 多任务框架

#### 5.8.1 它是什么（专业）

- 网络由 YAML 的 `network.net_parts` 描述成 DAG：
  - `name`：类名；
  - `net_name`：实例名/权重名；
  - `input`：依赖谁、取第几个输出；
  - `output`：给下游的名字；
  - `specs`：结构参数。
- `uvpsandbox/framework.py` + `module_container.py` 按拓扑执行；
- 同名子网只建一份，跨任务共享；
- 共享子网的 DDP 进程组覆盖所有使用它的任务 rank；
- 任务私有子网只在各自 rank 组内同步；
- `common_ft.yaml` 里 `gpus: [[0..7]]`、`task_ranks: [2,2,2,2]`，即 4 任务 × 2 卡；
- loss 侧 `weighter: ls`，`task_weights: [1.0, 1.0, 0.5, 1.0]`。

#### 5.8.2 它是什么（通俗）

四个任务一起训练，但不是一个网络从头到尾共用：

- 主干、FPN、回归分支：四个任务共用；
- 分类头、深度头、姿态头、交通牌头：各自私有；
- 每个任务分 2 张卡，共享部分跨 8 张卡同步；
- 最终 loss 是四个任务 loss 的加权和（姿态权重 0.5）。

> **通俗解释**：像四个学生共用语文/数学老师（主干、FPN、画框），但各自有专业课老师（分类头）。期末考试总分按权重相加，姿态这科权重低一点。

#### 5.8.3 代价

- 任务冲突：一个任务数据差会拖累共享层；
- 调试复杂：一个 loss 不降，可能是别的任务影响；
- checkpoint 要合并；
- 需要 loss 权重、采样比例、梯度监控。

> **代码位置**：`uvpsandbox/framework.py` 的 `_check_sharing_nets`、`_build_customized`；`uvpsandbox/weighter.py`。

---

### 5.9 部署相关

- `convert_d`：打开后 head 不再返回拼接张量，而是返回扁平的 cls/obj/bbox（以及 attr/depth）多输出；
- `torch.ops.nnx.custom_decode_bbox`：自定义解码算子；
- `launch/reform.py` + `configs/deploy/export_onnx.yaml`：导出 ONNX；
- `docs/TensorRt_deployment.md`、`docs/MDC_deployment.md`：TensorRT / MDC（昇腾）部署；
- 推理输入 704×1600；
- 评估 yaml 把 backbone / neck / head 拆成独立子图，便于分块导出。

> **通俗解释**：训练时模型输出“一个大拼盘”，部署芯片可能只吃“分开的小碟子”；`convert_d` 就是把拼盘改成分碟，方便芯片消化。

> ⚠️ 部署输出格式和训练输出格式不完全一致，后处理要按导出配置对齐。

---

## 6. 训练流程

### 6.1 两阶段训练（按 README）

| 阶段 | 配置目录 | 数据增强 | 目的 |
|---|---|---|---|
| 第一阶段 | `train_traffic_ascendnet_fpn_ncasd/` | 有 Mosaic、MixUp | 在大数据上充分学习，提升泛化 |
| 第二阶段 | `ft_traffic_iter5k/` | README 描述为去掉 Mosaic/MixUp | 微调，让模型适应真实分布 |

> ⚠️ 第二阶段具体关掉哪些增强，以 `ft_traffic_iter5k/*.yaml` 的 `pipeline` 为准。

### 6.2 关键超参（`train_traffic_ascendnet_fpn_ncasd/common_ft.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| `gpus` | `[[0,1,2,3,4,5,6,7]]` | 8 卡 |
| `task_ranks` | `[2,2,2,2]` | 4 任务 × 2 卡 |
| `train_with_eval` | `True` | 训练中评估 |
| `save_freq` | `5000` | 每 5k iter 存 checkpoint |
| `save_optim_for_resume` | `False` | 不存 optimizer，节省空间 |
| `adjust` | `True` | 按 checkpoint 调整网络宽度 |
| `load_path` | `ft_narrow/.../Iter-5000_StepShot.pth` | 从四任务微调模型继续训 |
| `weighter` | `ls` | 线性加权 |
| `task_weights` | `[1.0, 1.0, 0.5, 1.0]` | 姿态权重 0.5 |
| `optimizer` | `SGD` | lr 0.02，momentum 0.9，wd 0.0005 |
| `lr` | cosine 0→50k | 0.01 → 0 |
| `log_freq` | `100` | 日志频率 |

### 6.3 启动命令（README 示例）

第一阶段：

```bash
python train_test.py \
  --common_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/train_traffic_ascendnet_fpn/common_ft.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/train_traffic_ascendnet_fpn/ts_aux_1080p_ascendnet_fpn_insl3_public_full_halfbs.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/train_traffic_ascendnet_fpn/tl_outer_1080p_ascendnet_fpn_insl3_public_halfbs.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/train_traffic_ascendnet_fpn/tl_pose_1080p_ascendnet_fpn_insl3_public_full_halfbs.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/train_traffic_ascendnet_fpn/tl_inner_1080p_ascendnet_fpn_insl3_public_halfbs.yaml
```

> 注：仓库里实际目录名是 `train_traffic_ascendnet_fpn_ncasd`，README 里写的是 `train_traffic_ascendnet_fpn`，以实际目录为准。

第二阶段：

```bash
python train_test.py \
  --common_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/ft_traffic_iter5k/common_ft.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/ft_traffic_iter5k/ts_aux_1080p_ascendnet_fpn_insl3_public_full_halfbs.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/ft_traffic_iter5k/tl_outer_1080p_ascendnet_fpn_insl3_public_halfbs.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/ft_traffic_iter5k/tl_pose_1080p_ascendnet_fpn_insl3_public_full_halfbs.yaml \
  --task_cfg=tasks/YOLOXTraffic/uvpsandbox/configs/ft_traffic_iter5k/tl_inner_1080p_ascendnet_fpn_insl3_public_halfbs.yaml
```

### 6.4 数据与权重路径

- 训练数据路径在任务 yaml 的 `TRAINSET.ann_file` / `img_prefix`；
- 交通灯数据一般在 `/cache/traffic_light/`；
- 交通牌数据一般在 `/cache/traffic_sign/`；
- 姿态数据一般在 `/cache/traffic_light_pose/`；
- 预训练/微调权重在 `pretrained/ft.pth` 或 OBS 路径；
- 评估数据在 `VALSET`。

> **通俗解释**：这个工程不是“下载即跑”，数据和权重都在内部缓存/OBS 上；先确认路径，再谈训练。

---

## 7. 评测与指标

### 7.1 评测数据

`eval/benchmark.md` 把评测分成几类：

- **DTS 数据**：历史上出过问题的场景，例如上海世纪大道重叠灯、东陆路红灯、泰州灯笼误检；
- **城市数据**：上海、深圳、广州等，看泛化；
- **困难场景**：按颜色/功能/朝向/遮挡/内框形状统计；
- **倒计时数据**：个位/十位数字。

### 7.2 指标

- 外框：颜色、功能、朝向、遮挡的 AP；
- 内框：颜色、形状的 AP；
- 交通牌：分类 AP、召回、F1；
- 倒计时：每个数字的 AP。

### 7.3 DTS 数据结果（benchmark.md）

| 模型 | 外框颜色 | 外框功能 | 外框朝向 | 外框遮挡 | 内框颜色 | 内框形状 |
|---|---|---|---|---|---|---|
| TLS0206 | 91.10% | 91.13% | 90.76% | 87.69% | 86.48% | 97.11% |
| TLS0222 | 92.19% | 92.44% | 92.03% | 89.05% | 86.26% | 97.38% |
| TLS0302 | 92.02% | 91.92% | 91.16% | 88.43% | 87.88% | 96.88% |

> 原表表头写的是“Model (mAP)”，但按属性列展示；具体统计口径以 `benchmark.md` 为准。

### 7.4 重要提醒：没有逐项消融

> ⚠️ `benchmark.md` 只有最终指标，没有“去掉 FPN / 换回 CSPDarknet / 去掉共享头”的对照实验。  
> 所以这些数字能说明“这个模型最终效果”，但不能说明“每个模块各贡献多少”。  
> 面试/汇报时，正确说法是：“结构上它解决了 X，项目最终指标是 Y，但逐项贡献没有公开。”

---

## 8. 相对官方 YOLOX 的改动清单

### 8.1 改了什么

| # | 改动 | 官方 | 本项目 |
|---|---|---|---|
| 1 | Backbone | CSPDarknet（Focus + CSP + SPP + SiLU） | AscendNet（3 conv stem + 16 个分组残差块 + ReLU，无 SPP） |
| 2 | Neck | Megvii 版无 neck；mmdet 版有 PAFPN | 轻量 FPN（1×1 lateral + top-down add + 3×3 平滑） |
| 3 | Head 组织 | 单个 YOLOXHead，一任务一份 | 拆成 Com / Extract / Predictor，公共卷积/回归跨任务共享 |
| 4 | 分类语义 | 80 类平铺 BCE | 12/38/69 通道多标签 BCE + `attrs` 分组使用 |
| 5 | 额外分支 | 无 | depth / pose / TS aux / keypoint |
| 6 | 输入 | 640×640 | 704×1600（1024×1600 画布裁下部） |
| 7 | 训练组织 | 单任务 | 四任务联合训练 + 子网共享 + 模型并行 |
| 8 | 部署 | 单个 YOLOX ONNX | `convert_d` 多输出 + 子图拆分导出 |

### 8.2 没改什么

这些核心范式仍然保留：

- Anchor-Free；
- Decoupled Head 的思想（分类/回归分支分开）；
- SimOTA 动态正样本分配；
- IoU 系列框回归损失；
- 分类/objectness 用 BCE；
- obj bias 初始化 -4.5；
- BGR + /255；
- 不做水平翻转。

> **通俗解释**：它换的是“身体”（backbone/neck/head 组织），不是“灵魂”（anchor-free、SimOTA、decoupled head、IoU loss）。

---

## 9. 设计动机：问题 → 策略 → 落点 → 收益 → 代价

### 9.1 问题一：小目标漏检

| 项 | 内容 |
|---|---|
| 问题 | 远处红绿灯/交通牌很小，下采样后特征弱，正样本少 |
| 策略 | 704×1600 高分辨率 + C3 `/8` + FPN + 类别加权/重采样 |
| 落点 | 数据管线、backbone、neck、loss |
| 机制 | 分辨率保留更多像素；FPN 让浅层拿到语义；加权让长尾不被淹没 |
| 代价 | 算力大；裁掉下部目标；过采样可能过拟合 |
| 证据 | 机制合理；无逐项消融 |

### 9.2 问题二：遮挡

| 项 | 内容 |
|---|---|
| 问题 | 目标被路灯杆、前车、画面边缘挡住，可见范围小于真实范围 |
| 策略 | 把“遮挡”做成外框属性（无遮挡/部分遮挡）；多标签保留可见证据；不做双框 |
| 落点 | 标签体系、分类头 |
| 机制 | 遮挡不再和颜色/功能互斥；模型可以学“红 + 机动车 + 部分遮挡” |
| 代价 | 只判断“挡没挡”，不回归完整框；无法像 Mono3D 那样输出 amodal box |
| 证据 | 机制合理；无消融 |

> ⚠️ 这个项目**没有** visible box + full box 双框联合监督；它的遮挡策略是“属性标签”，不是“amodal 回归”。

### 9.3 问题三：长尾

| 项 | 内容 |
|---|---|
| 问题 | 黄灯、掉头、右转、部分遮挡等类别样本极少 |
| 策略 | 过采样（`rebalanced_dataset`）+ 类别加权（数字/形状 ×10~20）+ 多标签/分组 |
| 落点 | 数据采样、loss |
| 机制 | 少数类多看几遍、错了罚更重；分组降低每组类别数 |
| 代价 | 过采样可能过拟合；加权可能破坏分类/回归平衡 |
| 证据 | 机制合理；无消融 |

### 9.4 问题四：多任务冲突

| 项 | 内容 |
|---|---|
| 问题 | 四个任务数据分布不同，共享主干会互相拖累 |
| 策略 | 主干/FPN/回归共享，分类头私有；loss 权重 `[1,1,0.5,1]` |
| 落点 | 网络结构、loss |
| 机制 | 共享通用表征，私有头保留任务差异 |
| 代价 | 任务冲突、调试复杂、共享层学偏会一起偏 |
| 证据 | 机制合理；无消融 |

---

## 10. 长尾专题

### 10.1 什么是长尾

“长尾”指**类别样本数量极度不均衡**：少数类别样本特别多（“头”），大多数类别样本很少（“尾”），画成柱状图像一条长长的尾巴。

> **通俗解释**：不是“类别多”，而是“类别之间样本量差太多”。就像班里 10 个人，3 个人有 100 道练习题，7 个人只有 5 道。

### 10.2 本项目里的长尾

外框（困难场景数据，按帧统计）：

| 类别 | 样本量 | 位置 |
|---|---|---|
| 无遮挡 | 190,344 | 头 |
| 机动车控制 | 138,645 | 头 |
| 正面 | 104,516 | 头 |
| 黑灯 | 76,275 | 头 |
| 红灯 | 76,078 | 头 |
| 侧面 | 47,634 | 中 |
| 反向 | 46,813 | 中 |
| 绿灯 | 43,831 | 中 |
| 行人控制 | 60,318 | 中 |
| 黄灯 | 1,792 | 很尾 |
| 部分遮挡 | 8,619 | 尾 |

内框：

| 类别 | 样本量 | 位置 |
|---|---|---|
| 圆形 | 18,244 | 头 |
| 红灯 | 24,134 | 头 |
| 绿灯 | 19,117 | 头 |
| 数字 | 14,052 | 中 |
| 左转 | 6,156 | 中 |
| 自行车/行人 | 4,076 | 尾 |
| 右转 | 1,237 | 很尾 |
| 掉头 | 183 | 极尾 |
| 十位“0”/“百位” | 0 | 没有样本 |

> 注：以上是 `benchmark.md` 困难场景数据的帧数，不是全量训练集，但足以说明长尾程度。

### 10.3 长尾的危害

1. **模型偏心头类**：样本多的类别贡献梯度多，参数往头类调；
2. **尾类被当背景**：见得太少，分数低，被 `score_thr` 过滤，表现成漏检；
3. **尾类被头类吃掉**：黄灯少、红灯多，黄灯可能被判成红灯；
4. **整体指标被头类主导**：只看 mAP 看不出尾类掉点，要看 per-class AP。

### 10.4 本项目怎么处理

| 手段 | 位置 | 说明 |
|---|---|---|
| 过采样 | `rebalanced_dataset` | 对左转/右转/掉头等按 3~8 倍重复采样 |
| 类别加权 | `cls_weighted` | 数字/形状通道乘 10~20 |
| 多标签 | 分类 loss | 一个框可同时有多个属性，不互斥 |
| 属性分组 | `attrs` | 第一组颜色用于 SimOTA，loss 除以组数 |
| 数据补充 | 数据集 | 专门补黄灯、掉头、黑灯、数字灯等 corner case |

### 10.5 长尾怎么评估

- 不能只看整体 mAP；
- 要看 per-class AP、per-attribute AP、Recall；
- 特别关注尾类的漏检和误判；
- 训练时看每个类的 loss/正样本数，不要只看总 loss。

---

## 11. 初级感知算法工程师学习指南

### 11.1 三圈学习法：先抓 20%

**内圈（必须会，天天用）**

1. 检测三段式：backbone → neck → head → decode → NMS；能画出每层 stride 和通道数。
2. 分类和回归是两件事，可以共享特征，也可以分开。
3. 标签设计：多标签、软标签、类别权重、属性分组。
4. 输入分辨率/letterbox/裁剪对小目标的影响。
5. 训练稳定性：NaN、梯度、wh clamp、loss 权重。

**中圈（知道即可，能说出作用）**

- FPN / PAFPN / BiFPN 的区别；
- 共享头和私有头；
- 多任务 loss 权重；
- depth / pose / aux 分支；
- `convert_d` 是部署格式转换。

**外圈（暂时不用管，用到再查）**

- DDP process group、`net_parts` 拓扑；
- checkpoint 合并；
- ONNX 拆分导出；
- AMCT 剪枝；
- 自定义算子。

> 你不需要把外圈弄懂，先把内圈吃透，就已经能应付大多数感知算法岗的日常了。

### 11.2 必须掌握的 5 件事

1. **先问问题，再改结构**  
   小目标 → 分辨率 + C3 + FPN；遮挡 → 标签/属性；长尾 → 采样/加权；多任务 → 共享/私有。不要一上来就换 backbone。

2. **标签体系往往比模型更值钱**  
   把 12 类拆成多标签/属性，比换一个 backbone 更可能提升长尾和可解释性。

3. **共享是把双刃剑**  
   共享越多，参数越省，但任务冲突越大。共享层要配 loss 权重、采样比例、梯度监控。

4. **输入尺寸是“隐形的超参数”**  
   704×1600 还是 640×640，对小目标的影响可能比换 neck 还大。训练/评估/部署尺寸必须一致。

5. **没有消融，就没有“涨点”**  
   结构听起来高级 ≠ 有效。一次只改一个变量，固定数据/种子/训练轮数，看 per-class AP，不只看 mAP。

### 11.3 可以暂时跳过的

- `uvpsandbox/framework.py` 的 DDP 进程组细节；
- `module_container.py` 的拓扑执行；
- `reform_framework.py` 的 ONNX 导出；
- `slimming.py` 的剪枝；
- `components/model/` 里大量未启用的 backbone/neck（ResNet、DLA、EfficientNet、DyHead 等）。

> 这些是“平台工程”，不影响你理解检测模型本身。等你要部署/改多任务时再回来看。

### 11.4 读代码顺序

1. **四个任务 YAML 的 `net_parts`**：先看结构到底长什么样；
2. **`ascendnet.py`**：backbone 的 block 和 stage；
3. **`fpn.py`**：neck 的融合方式；
4. **`head_common.py`**：head 怎么拆、怎么共享；
5. **`vision_pv_tl.py`**：loss、SimOTA、多标签；
6. **`pv_dataset.py`**：输入尺寸、`height_used`、数据增强；
7. **`framework.py`** 只看“同名子网共享”那一段，其他先跳过。

### 11.5 四步学习计划

| 步骤 | 做什么 | 产出 |
|---|---|---|
| 第 1 步 | 画结构图：输入 → C3/C4/C5 → FPN → head → 输出，标出每层 stride/通道 | 一张自己能讲清楚的图 |
| 第 2 步 | 对照官方 YOLOX，列出 5 个差异（backbone/neck/head/分类/输入） | 一张对比表 |
| 第 3 步 | 读 loss：SimOTA、cls/obj/bbox、depth/attr，理解每一项在算什么 | 一页 loss 笔记 |
| 第 4 步 | 做小实验：推理一张图、改 `score_thr` 看结果；有算力就做 1k iter 的小消融 | 一组对比结果/结论 |

> 如果没算力，至少做“推理对比 + 结构推理”，不要编造训练结果。

### 11.6 面试/汇报模板

讲任何一个改动，都可以用这 6 句话：

1. **问题**：什么场景下效果差？（小目标/遮挡/长尾/多任务）
2. **原因**：为什么会差？（分辨率不够/特征弱/标签互斥/任务冲突）
3. **策略**：我选了什么？（FPN/多标签/共享头/加权）
4. **落点**：加在哪一层？（backbone/neck/head/loss/数据）
5. **代价**：多了多少参数/延迟？有什么副作用？
6. **证据**：有没有消融？指标是多少？没有就说没有。

> 这套模板比“我用了 YOLOX，加了 FPN，涨了点”强太多。

### 11.7 避坑清单

- 不要把 Mono3D 的双框/QFL/BiFPN/1700 万参数/23.6ms 套到这个 YOLOX 上；
- 不要看到“共享”就认为一定好，共享越深越容易互相拖累；
- 不要忽略训练/评估尺寸不一致（704 vs 1024）；
- 不要忽略部署约束（ReLU、分组卷积、无 SPP 可能是为 NPU 选的）；
- 不要只追求“结构高级”，数据、标签、输入分辨率往往更有效；
- 不要只报 mAP，要拆开看小目标、遮挡、长尾类别的 AP/Recall；
- 不要把 `attrs` 当成 4 个独立分类头，模型输出仍是扁平多通道 BCE。

### 11.8 自测问题

1. 为什么 YOLOX 用 anchor-free？和 anchor-based 相比省了什么？
2. Decoupled head 解决什么问题？为什么分类和回归要分开？
3. SimOTA 怎么给每个 GT 分配正样本？`center_radius` 和 `topk` 各控制什么？
4. 这个项目为什么把回归分支共享、分类分支私有？
5. 为什么输入是 704×1600，而不是 640×640？
6. 为什么这个项目没有 SPP、用 ReLU？可能是为了什么？
7. `attrs` 在 loss 里到底起什么作用？为什么说它不是 4 个分类头？
8. 多标签和普通多分类的区别是什么？
9. 长尾问题有哪些解法？这个项目用了哪几种？
10. 没有消融实验时，怎么评价一个结构改动？

---

## 12. 术语表

| 术语 | 英文/缩写 | 通俗解释 |
|---|---|---|
| 骨干 | Backbone | 网络的“眼睛”，把图片变成特征图 |
| 颈部 | Neck | 多尺度特征融合器 |
| 头部 | Head | 在每个格子上判读类别和框 |
| 特征金字塔 | FPN | 让浅层细节和深层语义融合 |
| 步长 | Stride | 下采样倍数，stride 8 = 每 8 像素一个格子 |
| 无锚框 | Anchor-Free | 不预设框，直接回归 |
| 正样本分配 | Assigner | 训练时决定哪些格子算正样本 |
| SimOTA | — | 动态正样本分配算法 |
| 二分类交叉熵 | BCE | 每个类别独立判断“是/不是” |
| 交并比 | IoU | 两个框重叠程度 |
| 平均精度 | AP / mAP | 检测精度指标 |
| 长尾 | Long-tail | 类别样本量极度不均衡 |
| 完整框 | Amodal Box | 包含被遮挡部分的完整目标框（本项目没有） |
| 质量焦点损失 | QFL | 把分类分数和框质量结合的损失（本项目没有） |
| 等比例填充 | Letterbox | 等比缩放 + 补边，不变形 |
| 部署转换 | convert_d | 把训练输出转成部署芯片格式 |
| 数据增强 | Augmentation | 把图片变着花样给模型看 |
| 过采样 | Rebalance | 少数类多重复几次 |

---

## 13. 代码与文档索引

| 想看什么 | 去哪里 |
|---|---|
| 网络结构 | 四个任务 yaml 的 `network.net_parts` |
| Backbone | `components/model/backbone/ascendnet.py` |
| Neck | `components/model/decoder/fpn.py` |
| Head | `tasks/YOLOXTraffic/models/heads/head_common.py` |
| 官方风格 CSPDarknet | `components/model/backbone/cspdarknet.py`、`tasks/YOLOXTraffic/models/backbones/csp_darknet.py` |
| 检测 loss / SimOTA | `tasks/YOLOXTraffic/loss_infer/vision_pv_tl.py`、`assigner.py` |
| 深度 loss | `tasks/YOLOXTraffic/loss_infer/vision_pv_tl_dep.py` |
| 姿态 loss | `tasks/YOLOXTraffic/loss_infer/vision_pv_tl_pose.py` |
| 数据加载/裁剪 | `tasks/YOLOXTraffic/dataset/pv_dataset.py` |
| 数据增强 | `tasks/YOLOXTraffic/dataset/transforms.py` |
| 标注转换 | `tools/create_traffic_light_outer.py`、`create_traffic_light_inner.py`、`create_traffic_sign.py` |
| 多任务框架 | `uvpsandbox/framework.py`、`module_container.py`、`weighter.py` |
| 训练入口 | `launch/train_test.py` |
| 导出入口 | `launch/reform.py`、`configs/deploy/export_onnx.yaml` |
| 评测指标 | `eval/benchmark.md`、`eval/detection_inference.py` |
| 部署文档 | `docs/TensorRt_deployment.md`、`docs/MDC_deployment.md` |

---

## 14. 一页纸总结

1. **定位**：UVPSandBox 多任务沙盒里的 YOLOX 交通灯/交通牌任务，不是官方 YOLOX 单任务仓库。
2. **结构**：AscendNet（16 个分组残差块）+ 轻量 FPN（1×1 + 上采样相加 + 3×3）+ 解耦共享 Head（公共卷积/回归共享，分类私有）。
3. **分类**：输出仍是 12/38/69 通道扁平 BCE；多标签 + `attrs` 分组使用，不是 4 个独立分类头。
4. **输入**：704×1600，BGR + /255，letterbox 式 Resize+Pad，不做水平翻转。
5. **多任务**：四个任务共享主干/FPN/回归分支，loss 权重 `[1,1,0.5,1]`，4 任务 × 2 卡。
6. **收益**：小目标靠分辨率 + C3 + FPN；遮挡靠属性标签；长尾靠过采样/加权；多任务靠共享/私有分工。
7. **诚实提醒**：没有逐项消融，所有“为什么提升”都是机制解释，不是实验证明；不要把 Mono3D 的双框/QFL 套过来。

---

## 15. 附录：常见误解

| 误解 | 事实 |
|---|---|
| “这个项目有 visible/full 双框” | ❌ 没有。遮挡是外框属性（无遮挡/部分遮挡），不是 amodal box 回归。 |
| “这个项目用了 QFL” | ❌ 没有。分类是 BCE，目标会乘 IoU 软标签，但不是 QFL。 |
| “这个项目按类回归，每类 4 个通道” | ❌ 回归是 class-agnostic 的 4 通道；Mono3D 才是按类回归。 |
| “这个项目把 Car/Truck/Tram 合并 NMS” | ❌ 没有。这是 Mono3D 的后处理；本项目是标准 NMS。 |
| “这个项目 1700 万参数、23.6ms” | ❌ 这是 Mono3D 的数据，不是这个 YOLOX 工程。 |
| “`attrs` 是 4 个独立分类头” | ❌ 模型输出仍是扁平 12/38 通道；`attrs` 只影响 SimOTA 第一组和 loss 归一化。 |
| “官方 YOLOX 一定有 PAFPN” | ❌ Megvii 官方版没有独立 neck；mmdet 版才有 YOLOXPAFPN。 |
| “共享头一定涨点” | ❌ 共享可能带来任务冲突，需要消融验证。 |
| “benchmark 的分数能证明每个模块的贡献” | ❌ 只有最终指标，没有逐项消融。 |
| “配置名里的 insl3 是一个网络层” | ❌ 代码里没有这个模块，属于模板/部署命名。 |

---

> **最后一句**：这个工程最值得学的不是“AscendNet 长什么样”，而是它**把问题拆成数据、标签、结构、损失、部署五层，并让每一层都为具体场景服务**。你不需要一次弄懂所有细节；先抓住“检测三段式 + 标签设计 + 输入尺寸 + 消融意识”这四件事，就已经超过大多数只会调包的人了。













## 可能需要去借鉴
按“对初级感知算法工程师的迁移价值”排序：
  - 固定数据/种子/iter，报告 per-class AP、latency、params。

  如果没算力：

  - 做推理侧实验：输入尺寸、score_thr、NMS IoU、裁剪策略；
  - 用公开交通灯数据集（LISA、DriveU、Bosch Small Traffic Lights）跑一遍流程；
  - 做一个离线小系统：检测 + 简单跟踪/去抖（SORT/ByteTrack 或手写 IoU 跟踪），看闪烁是否减少。

  产出：一页报告：问题、方案、指标、结论、下一步。

  ───
  五、哪些可以先跳过

  - UVPSandBox 的 DDP 进程组、net_parts 拓扑执行；
  - reform_framework.py 的 ONNX 拆图导出；
  - 自定义解码算子、Ascend/MDC 工具链；
  - BiFPN/DyHead/PAFPN 等可选 neck 的细节；
  - 剪枝、AMCT、量化感知训练。

  这些是“平台/部署工程”，等你要上车、要换芯片、要做多任务平台时再学。现在学容易陷进去，收益低。

  ───
  六、推荐的学习资源（按优先级）

  1. YOLOX 论文 + 官方代码：理解 anchor-free、decoupled head、SimOTA；
  2. mmdetection 的 YOLOX 配置：看工业级复现怎么组织；
  3. FCOS / SimOTA 原文：理解正样本分配；
  4. BCE / 多标签分类基础：理解为什么不是 softmax；
  5. FPN / BiFPN：理解多尺度融合；
  6. TensorRT / ONNX 基础：理解部署约束；
  7. SORT / ByteTrack：理解工业界为什么一定要跟踪；
  8. 高精地图/ROI 融合基础：理解系统级交通灯方案。

  不要一上来读 UVPSandBox 框架；先读“检测本身”，再读“系统”，最后读“平台”。

  ───
  七、一句话总结

  ▎ 这个项目最值得学的不是“AscendNet 长什么样”，而是它把交通灯拆成“检测 + 多属性 + 长尾 + 小目标 + 部署”的问题，并让每一层都为场景服务。
  ▎ 它的优势是“专科模型设计有想法”，劣势是“系统验证和工程化还不够”；你要学的是问题拆解和评测纪律，不是照抄结构。
  ▎ 工业界选型看的是场景、数据、芯片、系统，不是看谁的结构听起来更高级。
