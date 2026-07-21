# MASAC 轻量结构化 Critic 输入设计

## 1. 设计背景

当前 MASAC 在使用 `--critic-encoder mlp` 时，将全体智能体的原始观测和动作按固定顺序展平后输入集中式 Q 网络。默认 3 智能体场景中，单智能体观测为 312 维，Critic 动作为 6 维，因此联合输入为：

\[
D_Q=3\times(312+6)=954.
\]

当前结构可表示为：

\[
[o_1,a_1,o_2,a_2,o_3,a_3]
\rightarrow \operatorname{MLP}
\rightarrow Q_i.
\]

该结构主要存在以下问题：

1. 954 维输入主要由原始 LiDAR 距离组成，Critic 需要自行学习射线索引与空间方向之间的关系；
2. 智能体相对位置、相对速度和对应动作之间缺少显式绑定；
3. 对路径交叉、相向接近和让行关系缺少直接表征；
4. 固定顺序拼接不具备智能体排列不变性；
5. Actor 使用时序编码，而 MLP Critic 只选择当前有效观测；
6. 直接使用完整 Attention Critic 会增加 Q1、Q2、Target Q 和多个 focal critic 的重复计算成本。

因此，本设计采用轻量结构化 centralized critic state。在保持 Actor 局部观测不变的前提下，为 Critic 提供能够直接描述任务状态、控制作用和交互风险的低维输入。

## 2. 总体结构

对于 focal agent \(i\)，构造：

\[
x_i^Q=
\left[
x_i^{self},
x_i^{goal},
a_i^Q,
\{x_{ij}^{pair}\}_{j\ne i},
x_i^{obstacle}
\right].
\]

| 模块 | 作用 |
| --- | --- |
| 自身状态 \(x_i^{self}\) | 描述 focal agent 的位置、速度和任务阶段 |
| 目标状态 \(x_i^{goal}\) | 描述目标相对方向和剩余距离 |
| focal action \(a_i^Q\) | 描述当前策略产生的实际控制作用 |
| pair token \(x_{ij}^{pair}\) | 描述其他智能体的相对状态、动作和预测冲突风险 |
| 障碍摘要 \(x_i^{obstacle}\) | 描述不同方向的近期障碍距离和接近趋势 |

Actor 仍然只使用局部观测：

\[
a_i\sim\pi_i(o_i).
\]

Critic 在集中训练阶段使用结构化状态：

\[
Q_i=Q_i(x_i^Q).
\]

Critic 可以使用训练环境提供的归一化绝对位置和派生风险特征，但 Actor 不依赖这些特权信息，因此不破坏 CTDE 的分散执行条件。

## 3. 自身状态

建议定义：

\[
x_i^{self}=
\left[
\bar p_i,
\bar v_i,
s_i,
m_i^{reached}
\right].
\]

| 字段 | 维度 | 说明 |
| --- | ---: | --- |
| 归一化绝对位置 \(\bar p_i\) | 3 | 供 Critic 判断全局位置和边界关系 |
| 归一化速度 \(\bar v_i\) | 3 | 描述当前运动状态 |
| DMP phase \(s_i\) | 1 | 描述 DMP 执行阶段 |
| 到达标记 \(m_i^{reached}\) | 1 | 区分运动中和已冻结智能体 |

自身状态总维度为：

\[
D_{self}=8.
\]

### 3.1 位置归一化

\[
\bar p_i=
2\frac{p_i-p_{min}}{p_{max}-p_{min}}-1.
\]

采用 workspace 各轴分别归一化，使三个坐标分量均位于 \([-1,1]\)。绝对位置只供 Critic 使用。

### 3.2 速度归一化

\[
\bar v_i=
\operatorname{clip}
\left(
\frac{v_i}{v_{max}},-1,1
\right).
\]

如果各轴速度上下界不同，应逐轴使用相应的最大绝对值。

## 4. 目标相对状态

定义目标相对位移：

\[
r_i^g=g_i-p_i.
\]

建议输入：

\[
x_i^{goal}=
\left[
\bar r_i^g,
\bar d_i^g
\right],
\]

其中：

\[
\bar r_i^g=
\frac{g_i-p_i}{p_{max}-p_{min}},
\]

\[
\bar d_i^g=
\operatorname{clip}
\left(
\frac{\|g_i-p_i\|}{d_{workspace}},0,1
\right),
\qquad
d_{workspace}=\|p_{max}-p_{min}\|.
\]

目标状态包含 3 维相对位移和 1 维距离，总维度为：

\[
D_{goal}=4.
\]

相对位移同时保留方向和尺度，额外距离标量用于强化剩余任务进度。

## 5. 联合动作

当前 Critic 动作为实际加速度与 goal offset 的组合：

\[
a_j^Q=
\left[
a_j^{acc},
\Delta g_j
\right].
\]

归一化形式为：

\[
\bar a_j^{acc}=
\frac{a_j^{acc}}{a_{max}},
\qquad
\overline{\Delta g}_j=
\frac{\Delta g_j}{\Delta g_{max}}.
\]

如果采用固定向量 MLP，可以输入：

\[
x^{joint\_action}=[a_1^Q,a_2^Q,a_3^Q].
\]

更推荐将动作与实体绑定：

- focal agent 的动作放入 focal 输入；
- agent \(j\) 的动作放入对应的 pair token。

该设计能够保留“动作—实体—相对关系”的对应关系，避免集合聚合后无法判断某个动作属于哪个智能体。

## 6. 智能体两两相对状态

对于 focal agent \(i\) 和其他智能体 \(j\)，定义：

\[
r_{ij}=p_j-p_i,
\qquad
u_{ij}=v_j-v_i,
\qquad
d_{ij}=\|r_{ij}\|.
\]

### 6.1 双尺度相对位置

当前实现使用较小影响距离归一化并裁剪，远距离信息容易饱和。建议同时保留全局尺度和近场尺度。

全局尺度：

\[
\bar r_{ij}^{global}=
\frac{p_j-p_i}{p_{max}-p_{min}}.
\]

近场尺度：

\[
\bar d_{ij}^{near}=
\operatorname{clip}
\left(
\frac{d_{ij}}{d_{safe}},0,d_{near,max}
\right).
\]

全局尺度保留远距离关系，近场尺度强化安全距离附近的变化。

### 6.2 相对速度

\[
\bar u_{ij}=
\frac{v_j-v_i}{2v_{max}}.
\]

### 6.3 接近速度

定义单位相对方向：

\[
\hat r_{ij}=
\frac{r_{ij}}{\|r_{ij}\|+\epsilon}.
\]

接近速度为：

\[
v_{ij}^{close}=
-\hat r_{ij}^{\mathsf T}u_{ij}.
\]

- \(v_{ij}^{close}>0\)：两个智能体正在接近；
- \(v_{ij}^{close}<0\)：两个智能体正在远离；
- 数值越大，表示沿连线方向的接近速度越高。

该特征将相对位置与相对速度之间的点积关系显式提供给 Critic。

## 7. TTC 预计碰撞时间

TTC（Time to Collision）用于估计在恒定速度假设下，两个智能体多久以后会进入安全距离。

设安全距离为 \(d_{safe}\)，相对运动模型为：

\[
r_{ij}(t)=r_{ij}+u_{ij}t.
\]

求解：

\[
\|r_{ij}+u_{ij}t\|^2=d_{safe}^2.
\]

展开为：

\[
at^2+bt+c=0,
\]

其中：

\[
a=u_{ij}^{\mathsf T}u_{ij},
\quad
b=2r_{ij}^{\mathsf T}u_{ij},
\quad
c=r_{ij}^{\mathsf T}r_{ij}-d_{safe}^2.
\]

判别式为：

\[
\Delta=b^2-4ac.
\]

当 \(a>\epsilon\)、\(\Delta\ge0\)，且最小根非负时：

\[
TTC_{ij}=
\frac{-b-\sqrt{\Delta}}{2a}.
\]

特殊情况处理：

1. 当前已经进入安全距离时，设置 \(TTC_{ij}=0\)；
2. 相对速度过小、无实数根或碰撞根为负时，认为预测窗口内无碰撞；
3. 无有效碰撞预测时，将 TTC 设置为 \(T_{max}\)，并将有效标记设置为 0。

归一化：

\[
\overline{TTC}_{ij}=
\operatorname{clip}
\left(
\frac{TTC_{ij}}{T_{max}},0,1
\right).
\]

同时输入有效标记：

\[
m_{ij}^{TTC}\in\{0,1\}.
\]

不能只输入裁剪后的 TTC，因为“超过预测窗口”和“根据当前速度不会碰撞”都可能映射为 1。建议初始设置：

\[
T_{max}=2\sim4\ \mathrm{s}.
\]

## 8. CPA 最近接近距离

CPA（Closest Point of Approach）用于描述保持当前速度时，两个智能体将在什么时刻达到最小距离，以及该距离是否满足安全约束。

最近接近时刻：

\[
t_{ij}^{CPA}=
-\frac{r_{ij}^{\mathsf T}u_{ij}}
{\|u_{ij}\|^2+\epsilon}.
\]

裁剪到预测窗口：

\[
t_{ij}^{CPA}=
\operatorname{clip}(t_{ij}^{CPA},0,T_{max}).
\]

最近接近距离：

\[
d_{ij}^{CPA}=
\left\|
r_{ij}+u_{ij}t_{ij}^{CPA}
\right\|.
\]

CPA 安全余量：

\[
c_{ij}^{CPA}=d_{ij}^{CPA}-d_{safe}.
\]

- \(c_{ij}^{CPA}<0\)：预计进入不安全区域；
- \(c_{ij}^{CPA}=0\)：预计达到安全边界；
- \(c_{ij}^{CPA}>0\)：预计保持安全。

归一化形式为：

\[
\bar t_{ij}^{CPA}=
\frac{t_{ij}^{CPA}}{T_{max}},
\]

\[
\bar c_{ij}^{CPA}=
\operatorname{clip}
\left(
\frac{c_{ij}^{CPA}}{d_{sensing}},-1,1
\right).
\]

TTC 描述“是否以及多久以后进入安全距离”，CPA 描述“按照当前运动趋势，两条轨迹将接近到什么程度”。二者联合使用能够更稳定地表达路径交叉风险。

## 9. Pair Token

对每个其他智能体 \(j\)，建议构造：

\[
x_{ij}^{pair}=
\left[
\bar r_{ij}^{global},
\bar u_{ij},
\bar d_{ij}^{near},
\bar v_{ij}^{close},
\overline{TTC}_{ij},
m_{ij}^{TTC},
\bar t_{ij}^{CPA},
\bar c_{ij}^{CPA},
\bar a_j^{acc},
\overline{\Delta g}_j,
m_j^{reached}
\right].
\]

| 字段 | 维度 |
| --- | ---: |
| 相对位置 | 3 |
| 相对速度 | 3 |
| 近场距离 | 1 |
| 接近速度 | 1 |
| TTC | 1 |
| TTC 有效标记 | 1 |
| CPA 时刻 | 1 |
| CPA 安全余量 | 1 |
| 对方实际加速度 | 3 |
| 对方 goal offset | 3 |
| 对方到达标记 | 1 |

单个 pair token 为 19 维。当前 3 智能体场景中，每个 focal agent 对应两个 pair token，总维度为 38。

## 10. 最近障碍距离摘要

为降低计算成本，Actor 继续使用完整 LiDAR，Critic 使用低维障碍摘要，而不是将全体智能体的原始 LiDAR 数组再次展平。

### 10.1 目标对齐分区

当前点质量模型不显式建模姿态，可以构造目标对齐局部坐标系。

前向轴：

\[
e_f=
\frac{g_i-p_i}{\|g_i-p_i\|+\epsilon}.
\]

世界坐标上向轴：

\[
e_z=[0,0,1]^{\mathsf T}.
\]

侧向轴：

\[
e_l=
\frac{e_z\times e_f}
{\|e_z\times e_f\|+\epsilon}.
\]

局部上向轴：

\[
e_u=e_f\times e_l.
\]

当目标方向接近竖直方向时，应使用固定备用轴构造侧向方向。

将射线划分为前、后、左、右、上、下 6 个区域。每个区域统计：

\[
x_{i,k}^{sector}=
\left[
d_{min},
d_{softmin},
\dot d_{min},
ratio_{hit}
\right].
\]

距离变化率：

\[
\dot d_r=
\frac{d_{t,r}-d_{t-1,r}}{\Delta t}.
\]

- \(\dot d_r<0\)：对应方向的障碍正在接近；
- \(\dot d_r>0\)：对应方向的障碍正在远离。

6 个区域、每个区域 4 个统计量时：

\[
D_{obstacle}=6\times4=24.
\]

### 10.2 Soft-Min

直接使用最小距离容易受单条射线影响，可以增加平滑 soft-min：

\[
d_{softmin}=
\frac{\sum_r d_r\exp(-\beta d_r)}
{\sum_r\exp(-\beta d_r)+\epsilon}.
\]

参数 \(\beta\) 越大，结果越接近区域最小距离。

### 10.3 Top-K 危险射线备选

也可以选择距离最小的 \(K\) 条命中射线，每条射线表示为：

\[
x_{ik}^{ray}=
[d_{ik},\dot d_{ik},d_{x,k},d_{y,k},d_{z,k},hit_k].
\]

当 \(K=8\) 时，障碍摘要为 48 维。该方案保留危险射线方向，但 Top-K 排序可能在相邻时间步跳变，因此第一版优先使用分区 soft-min。

### 10.4 Critic 特权障碍物摘要

如果 LiDAR 摘要不足以支持准确 Q 值估计，可直接使用训练环境中最近 \(K\) 个障碍物：

\[
x_{ik}^{obs}=
[
type_k,
c_k-p_i,
size_k,
v_k-v_i,
clearance_{ik},
TTC_{ik}^{obs},
d_{CPA,ik}^{obs}
].
\]

障碍物类型可区分边界、地面长方体、空中静态球和动态球。建议仅在分区摘要效果不足时，再引入最近 \(K=4\) 个障碍物 token。

## 11. 第一版推荐结构

第一版采用固定长度输入：

\[
x_i^Q=
[
x_i^{self},
x_i^{goal},
a_i^Q,
x_{i1}^{pair},
x_{i2}^{pair},
x_i^{obstacle}
].
\]

| 模块 | 维度 |
| --- | ---: |
| 自身状态 | 8 |
| 目标状态 | 4 |
| focal action | 6 |
| 两个 pair token | \(2\times19=38\) |
| 6 区域障碍摘要 | 24 |
| 合计 | 80 |

推荐 MLP：

\[
80\rightarrow256\rightarrow256\rightarrow256\rightarrow1.
\]

相比当前：

\[
954\rightarrow256\rightarrow256\rightarrow1,
\]

即使增加一个隐藏层，第一层矩阵乘法规模和总参数量仍然显著降低。Q1 与 Q2 保持参数完全独立：

\[
Q_{i,1}=f_{\theta_1}(x_i^Q),
\qquad
Q_{i,2}=f_{\theta_2}(x_i^Q).
\]

## 12. 轻量关系网络扩展

如果固定长度 MLP 不能充分处理路径交叉和让行关系，可进一步采用单查询 Cross-Attention：

```text
focal token ─────────────── Query

pair token 1 ─┐
pair token 2 ─┼──────────── Key / Value
obstacle token┘

Query + Key/Value
        ↓
轻量 Cross-Attention
        ↓
interaction context
        ↓
Q MLP
```

当前只有两个其他智能体和少量障碍 token，复杂度近似为 \(O(N+K)\)，远低于在多个 Q 分支中重复编码 144 条 LiDAR 射线。

## 13. Replay Buffer 接口

Actor 观测不包含未裁剪绝对位置、完整 pairwise 状态和精确障碍物信息，因此结构化 Critic state 不能在采样后从 Actor 观测中完整恢复。

Replay Buffer 需要分别保存：

```text
actor_obs
critic_state
critic_action
reward
next_actor_obs
next_critic_state
done
```

单步数据流程为：

```text
当前环境状态
    ├─ 构造 actor_obs
    ├─ 构造 critic_state
    └─ Actor 生成策略动作

策略动作
    ↓ DMP 映射
实际加速度与 goal offset
    ↓
环境执行
    ├─ reward
    ├─ next_actor_obs
    └─ next_critic_state

写入 Replay Buffer
```

Critic 更新使用 \(Q_i(x_{i,t}^Q,a_t)\)，Target Q 使用 \(Q_i^{target}(x_{i,t+1}^Q,a_{t+1})\)。Actor 更新时：

- focal agent 的策略动作保留梯度；
- 其他智能体动作使用 replay action 或 detached policy action；
- critic state 不需要梯度；
- Actor 梯度通过 Critic 对 focal action 的偏导返回策略网络。

## 14. 与奖励设计的关系

结构化 Critic 只能提高当前 reward function 下的 Q 值估计精度，不能自动修正奖励目标。

当前课程成功指标要求：

\[
success=\bigwedge_i success_i.
\]

如果两个智能体到达后已经获得较高局部奖励，而第三个智能体失败时没有充分的 team failure cost，那么更准确的 Critic 仍可能将 `2/3` 到达轨迹估计为高价值。

因此应同步验证：

1. 增加全体完成后的 team success bonus；
2. 区分个体到达奖励和团队完成奖励；
3. 使碰撞代价与成功奖励保持合理比例；
4. 对 `2/3` 到达但整体超时或碰撞增加团队失败代价；
5. 分别记录 individual return 和 team return。

## 15. 训练诊断指标

修改 Critic 后应同步记录：

### 15.1 价值估计

- Q1、Q2 和 Target Q 的均值与标准差；
- Q1/Q2 disagreement；
- TD error 均值、标准差和高分位数；
- Critic loss；
- Q 值最大绝对值。

### 15.2 策略更新

- Actor loss；
- entropy 和 temperature \(\alpha\)；
- forcing term 与 goal offset 饱和率；
- 实际加速度裁剪率；
- Actor 与 Critic 梯度范数。

### 15.3 风险特征

- 最小智能体间距离；
- 最小 TTC；
- TTC 有效 pair 数量；
- 最小 CPA 安全余量；
- 各障碍分区最小距离；
- 障碍接近速度。

## 16. 分阶段实现建议

### 阶段 A：奖励与诊断基线

1. 保持现有 MLP Critic；
2. 补充 Q、TD error、Actor/Critic loss 和 entropy 监控；
3. 对齐个体奖励与全体成功指标；
4. 使用固定 100 个种子建立 checkpoint 基线。

### 阶段 B：结构化 MLP Critic

1. 增加独立 critic state；
2. 增加自身状态、目标状态和 focal action；
3. 增加 pair token、TTC 和 CPA；
4. 增加 6 区域 LiDAR 障碍摘要；
5. 将 Critic 输入由 954 维原始拼接替换为约 80 维结构化输入；
6. 保持 Actor、课程场景和其余超参数不变，执行单变量消融。

### 阶段 C：轻量关系聚合

1. 使用共享 MLP 编码 pair token；
2. 比较固定拼接、Mean-Max pooling 和单查询 Cross-Attention；
3. 必要时增加最近障碍物 token；
4. 比较 Q 稳定性、成功率和训练吞吐量。

### 阶段 D：算法对照

1. 在相同奖励和观测条件下建立 MAPPO 基线；
2. 比较达到目标成功率所需的环境步数；
3. 比较达到目标成功率所需的墙钟时间；
4. 比较训练曲线方差和后期策略退化程度。

## 17. 推荐消融实验

| 实验 | Critic 输入 | 实验目的 |
| --- | --- | --- |
| B0 | 当前 954 维 raw MLP | 基线 |
| B1 | 自身状态 + 目标状态 + 联合动作 | 验证基础结构化状态 |
| B2 | B1 + 相对位置/速度 | 验证显式交互关系 |
| B3 | B2 + TTC/CPA | 验证预测风险特征 |
| B4 | B3 + 6 区域障碍摘要 | 验证低维障碍信息 |
| B5 | B4 + pair Cross-Attention | 验证轻量关系 Attention |
| B6 | B5 + 特权障碍物 token | 验证精确障碍状态 |

各组实验应固定训练种子、课程配置、Actor 网络、reward function、batch size、learning rate、learn interval 和评估种子。

核心指标包括：

- 全体成功率和单智能体成功率；
- `0/3、1/3、2/3、3/3` 到达分布；
- 三类碰撞率与超时率；
- Q1/Q2 disagreement 和 TD error；
- 单步推理时间与每秒环境步数；
- 达到目标成功率所需的墙钟时间。

## 18. 结论

轻量结构化 Critic 的核心不是简单增加网络宽度，而是将高维原始输入转换为具有明确物理意义和交互关系的低维状态：

\[
\text{任务状态}
+\text{实际控制}
+\text{相对运动}
+\text{TTC/CPA 风险}
+\text{障碍方向摘要}.
\]

推荐第一版采用约 80 维结构化输入和独立双 Q MLP，在保持 Actor 局部观测不变的条件下，提高 Critic 对多智能体交叉避让、长期任务完成和碰撞风险的区分能力。该方案计算成本低于完整 Attention Critic，也适合作为后续 SAC 与 MAPPO 对照实验的统一 centralized state 表达。
