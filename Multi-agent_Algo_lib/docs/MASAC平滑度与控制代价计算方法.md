# MASAC 平滑度与控制代价计算方法

## 1. 指标目的

在多无人机连续控制任务中，成功率和碰撞率主要反映任务完成能力与安全性，但无法完整描述控制输出的工程质量。因此，验证脚本进一步计算平滑度代价与控制代价：

- 平滑度代价用于衡量实际加速度随时间变化的剧烈程度；
- 控制代价用于衡量完成任务所消耗的控制强度；
- 两项指标均基于环境实际执行的 `applied_accelerations`，而不是裁剪前的 `commanded_accelerations`。

使用 applied acceleration 可以使评价结果与无人机真实状态转移保持一致。commanded acceleration 仍被保存，用于计算加速度裁剪率和分析控制器饱和现象。

## 2. 数据定义

设场景中共有 (N) 个智能体，单个 episode 包含 (T) 个控制步，环境时间间隔为：

\[
\Delta t = \texttt{config.time\_step}
\]

智能体 (i) 在第 (t) 个控制步实际执行的三维加速度为：

\[
\mathbf a_{i,t}
=
\begin{bmatrix}
a^x_{i,t} & a^y_{i,t} & a^z_{i,t}
\end{bmatrix}^{\mathrm T}
\]

验证脚本从每个 frame 的 `applied_accelerations` 字段读取该变量。初始 frame 不对应一次实际控制执行，因此计算时使用 `frames[1:]`。

## 3. 平滑度代价

### 3.1 Jerk 计算

加速度的一阶时间差分定义为 jerk：

\[
\mathbf j_{i,t}
=
\frac{\mathbf a_{i,t}-\mathbf a_{i,t-1}}{\Delta t},
\qquad t=2,\ldots,T
\]

其平方模为：

\[
\left\|\mathbf j_{i,t}\right\|_2^2
=
(j^x_{i,t})^2+(j^y_{i,t})^2+(j^z_{i,t})^2
\]

### 3.2 单智能体平滑度代价

当前实现采用 episode 内平均平方 jerk：

\[
J^{\mathrm{smooth}}_i
=
\frac{1}{T-1}
\sum_{t=2}^{T}
\left\|\mathbf j_{i,t}\right\|_2^2
\]

对应实现为：

```python
jerk = np.diff(accelerations, axis=0) / dt
smoothness_costs = np.mean(np.sum(jerk ** 2, axis=2), axis=0)
```

当 episode 中不足两个有效加速度样本时，平滑度代价设置为 0。

### 3.3 Episode 级平滑度

团队 episode 平滑度为所有智能体平滑度代价的均值：

\[
J^{\mathrm{smooth}}_{\mathrm{episode}}
=
\frac{1}{N}
\sum_{i=1}^{N}J^{\mathrm{smooth}}_i
\]

输出字段为：

- 单机：`smoothness_cost`；
- episode：`smoothness_cost_mean`；
- 场景聚合：均值、标准差、最小值、最大值、中位数和 P95。

### 3.4 指标解释

该指标越小，说明实际加速度变化越平缓；数值较大通常表示策略存在高频修正、方向快速切换或加速度饱和后的突变。

若位置采用 m、时间采用 s，则 jerk 的量纲为 m/s³，当前平均平方 jerk 指标的量纲为 m²/s⁶。

该指标评价的是控制信号平滑性，而不是几何轨迹曲率。若需要评价轨迹几何形状，还应额外计算速度方向变化率、曲率或轨迹二阶差分。

由于 jerk 包含 (1/\Delta t)，平方 jerk 包含 (1/\Delta t^2)，不同时间步长配置下的结果不能直接比较。本工程批量验证使用相同的 `time_step`，因此不同场景之间可以进行横向比较。

## 4. 控制代价

### 4.1 单智能体控制代价

控制代价采用实际加速度平方模的离散时间积分：

\[
J^{\mathrm{control}}_i
=
\sum_{t=1}^{T}
\left\|\mathbf a_{i,t}\right\|_2^2\Delta t
\]

对应实现为：

```python
control_costs = (
    np.sum(np.sum(accelerations ** 2, axis=2), axis=0) * dt
)
```

### 4.2 Episode 级控制代价

团队 episode 控制代价采用单机控制代价均值：

\[
J^{\mathrm{control}}_{\mathrm{episode}}
=
\frac{1}{N}
\sum_{i=1}^{N}J^{\mathrm{control}}_i
\]

输出字段为：

- 单机：`control_cost`；
- episode：`control_cost_mean`；
- 场景聚合：均值、标准差、最小值、最大值、中位数和 P95。

### 4.3 指标解释

控制代价越小，表示策略整体使用的实际加速度越低。但该指标会随飞行时间累积，因此必须结合任务成功率、飞行时间和路径长度共同分析：

若位置采用 m、时间采用 s，则该离散积分控制代价的量纲为 m²/s³。

- 成功率相近时，控制代价更低通常表示控制效率更高；
- 控制代价低但任务失败，可能只是策略未采取有效动作；
- 控制代价较高但飞行时间显著缩短，可能体现速度与能耗之间的权衡；
- 碰撞导致的短 episode 可能产生较低累计控制代价，不能据此判断策略更优。

## 5. 与加速度裁剪率的关系

环境同时记录 commanded acceleration 与 applied acceleration。若某一维超过动力学加速度边界，环境会执行裁剪：

\[
\mathbf a^{\mathrm{applied}}_{i,t}
=
\operatorname{clip}
\left(
\mathbf a^{\mathrm{commanded}}_{i,t},
\mathbf a_{\min},
\mathbf a_{\max}
\right)
\]

验证脚本将任意维度发生差异视为一次裁剪事件，并计算：

\[
r^{\mathrm{clip}}_i
=
\frac{\text{发生裁剪的控制步数}}{T}
\]

当平滑度代价和裁剪率同时较高时，通常说明策略频繁输出接近或超过动力学边界的动作，需要进一步检查动作尺度、reward function 或 actor 输出分布。

## 6. 推荐分析方式

平滑度和控制代价不应脱离任务性能单独排序。推荐按照以下顺序分析：

1. 首先比较成功率和碰撞率，确认策略满足任务与安全要求；
2. 在成功率接近的方案之间比较路径长度和飞行时间；
3. 在任务效率接近的方案之间比较平滑度和控制代价；
4. 联合检查加速度裁剪率，判断高控制代价是否来自持续饱和；
5. 分别报告全部 episode 与成功 episode 的统计结果，避免失败或碰撞造成评价偏差。

## 7. 可视化对应关系

3D HTML 页面支持以下加速度显示模式：

- `Applied acceleration`：显示实际执行加速度，采用实线箭头；
- `Commanded acceleration`：显示裁剪前命令加速度，采用半透明虚线箭头；
- `Applied + Commanded`：同时显示两者，用于观察裁剪差异；
- `隐藏加速度`：关闭加速度箭头。

箭头方向表示三维加速度方向，箭头长度随加速度模长增加并设置显示上限。播放或拖动时间滑块时，箭头随当前 frame 同步更新。启用“历史箭头”后，页面对历史时间点进行降采样绘制，以避免长轨迹产生过度遮挡。

## 8. 当前训练奖励设置

### 8.1 单步奖励总体结构

设智能体 \(i\) 在控制步 \(t\) 的奖励为 \(r_{i,t}\)。在未触发成功、碰撞或超时事件时，当前环境使用的基础单步奖励为：

\[
r^{\mathrm{base}}_{i,t}
=
r^{\mathrm{progress}}_{i,t}
+r^{\mathrm{near}}_{i,t}
-p^{\mathrm{obs}}_{i,t}
-p^{\mathrm{bound}}_{i,t}
-p^{\mathrm{agent}}_{i,t}
-p^{\mathrm{step}}
\]

其中，\(r^{\mathrm{progress}}_{i,t}\) 用于鼓励智能体缩短目标距离，\(r^{\mathrm{near}}_{i,t}\) 用于增强目标邻域内的到达引导，三个势场项分别用于约束障碍物、环境边界和智能体间的安全距离，\(p^{\mathrm{step}}\) 用于抑制无效停留。

考虑碰撞、成功和超时事件后，完整奖励可表示为：

\[
r_{i,t}
=
r^{\mathrm{base}}_{i,t}
-p^{\mathrm{collision}}_{i,t}
+r^{\mathrm{success}}_{i,t}
-p^{\mathrm{timeout}}_{i,t}
\]

当前实现还会计算加速度代价与加速度裁剪代价，但这两项在奖励组合代码中已被注释，因此不参与实际训练奖励。

### 8.2 目标进度奖励

设当前步执行前、后智能体到目标点的距离分别为 \(d_{i,t-1}\) 和 \(d_{i,t}\)，则进度量为：

\[
\Delta d_{i,t}=d_{i,t-1}-d_{i,t}
\]

对应奖励为：

\[
r^{\mathrm{progress}}_{i,t}
=w_{\mathrm{progress}}\Delta d_{i,t}
\]

当前参数为 \(w_{\mathrm{progress}}=4.0\)。当智能体接近目标时，该项为正；当智能体远离目标时，该项为负。

### 8.3 近目标区域奖励

对于尚未获得成功奖励且距离大于目标容差的智能体，近目标奖励为：

\[
r^{\mathrm{near}}_{i,t}
=
0.2\,\mathbb I(d_{i,t}<1.0)
+0.6\,\mathbb I(d_{i,t}<0.6)
\]

其中，\(\mathbb I(\cdot)\) 为指示函数。两级奖励采用累加关系，而不是互斥关系。因此：

- 当 \(0.6\le d_{i,t}<1.0\) 时，单步近目标奖励为 0.2；
- 当 \(0.3<d_{i,t}<0.6\) 时，单步近目标奖励为 0.8；
- 当 \(d_{i,t}\le 0.3\) 时，智能体进入成功判定，不再计算近目标奖励。

### 8.4 障碍物方向性势场惩罚

静态障碍物和动态障碍物采用相同的方向性人工势场计算方式。对于处于影响范围内的障碍物，定义动作方向门控和速度方向门控：

\[
g_a=\max(0,\hat{\mathbf a}_{i,t}^{\mathrm T}\hat{\mathbf e}_{io}),
\qquad
g_v=\max(0,\hat{\mathbf v}_{i,t}^{\mathrm T}\hat{\mathbf e}_{io})
\]

其中，\(\hat{\mathbf e}_{io}\) 表示由智能体指向障碍物最近点的单位方向，\(\hat{\mathbf a}_{i,t}\) 表示根据原始策略动作推导的运动方向。当速度模长接近零时，当前实现令 \(g_v=0\)。组合方向门控为：

\[
g_{io}=g_a(0.5+0.5g_v)
\]

设智能体到障碍物表面的距离为 \(d_{io}\)，障碍物影响距离为 \(d_{0,o}=1.5\)，则惩罚为：

\[
p^{\mathrm{obs}}_{i,t}
=
\min\left[
2.0\sum_{o:d_{io}<d_{0,o}}
\left(\frac{1}{d_{io}}-\frac{1}{d_{0,o}}\right)^2g_{io},
20.0
\right]
\]

该机制仅在策略动作朝向障碍物时产生惩罚，从而降低对远离障碍物动作的无效约束。

### 8.5 边界方向性势场惩罚

边界势场沿三维工作空间各坐标轴的上下边界分别计算。设智能体到某一边界的有符号距离为 \(d_{ib}\)，边界影响距离为 \(d_{0,b}=0.6\)，方向门控 \(g_{ib}\) 与障碍物势场采用相同结构，则：

\[
p^{\mathrm{bound}}_{i,t}
=
\min\left[
0.3\sum_{b:d_{ib}<d_{0,b}}
\left(\frac{1}{\max(d_{ib},10^{-3})}-\frac{1}{d_{0,b}}\right)^2g_{ib},
20.0
\right]
\]

边界惩罚只约束朝向邻近边界的动作。若智能体已经越界，有符号距离会通过 \(10^{-3}\) 截断，以避免除零或数值发散。

### 8.6 智能体间势场惩罚

设智能体 \(i\) 与 \(j\) 的中心距离为 \(d_{ij}\)，智能体间势场影响距离为 \(d_{0,a}=1.2\)，则单个智能体接收的势场惩罚为：

\[
p^{\mathrm{agent}}_{i,t}
=
\sum_{j\ne i,\,d_{ij}<d_{0,a}}
1.0\left(
\frac{1}{\max(d_{ij},10^{-3})}
-\frac{1}{d_{0,a}}
\right)^2
\]

每对智能体产生的惩罚同时分配给双方。与障碍物和边界势场不同，该项当前仅依赖智能体间距离，尚未引入相对速度、相对运动方向或碰撞时间等信息，也未设置最大值截断。

### 8.7 固定步长惩罚

每个控制步对所有智能体施加固定惩罚：

\[
p^{\mathrm{step}}=0.01
\]

该项用于降低无效停留和过长飞行时间，但其作用需要结合进度奖励及终止奖励共同分析。

### 8.8 碰撞、成功与超时奖励

#### 碰撞惩罚

当前碰撞惩罚按照碰撞类型分别计算：

- 与静态或动态障碍物碰撞：80.0；
- 越过环境边界：80.0；
- 智能体间碰撞：20.0。

若同一智能体在同一步同时满足多种碰撞条件，各项惩罚将累加。任意智能体发生碰撞时，团队 episode 立即终止。

#### 成功奖励

目标到达容差为：

\[
d_{\mathrm{goal}}^{\mathrm{tol}}=0.3
\]

每个首次到达目标且当前步未发生碰撞的智能体获得一次成功奖励：

\[
r^{\mathrm{success}}_{i,t}
=
\frac{600.0}{N}
\]

其中，\(N\) 为场景中的智能体数量。例如，三智能体场景中单个智能体的成功奖励为 200.0。智能体获得成功奖励后会被冻结，避免其后续状态变化影响任务判定。所有智能体均到达目标后，episode 正常终止。

#### 超时惩罚

当前 episode 最大控制步数为 200。当未发生碰撞且未实现全体成功，并达到最大步数时，episode 被截断。对于尚未获得成功奖励的智能体，超时惩罚为：

\[
p^{\mathrm{timeout}}_{i,t}
=
20.0\frac{d_{i,t}}{D_{\mathrm{workspace}}}
\]

其中，\(D_{\mathrm{workspace}}\) 为工作空间上下边界构成的三维对角线长度。该设计使超时惩罚随剩余目标距离增大。

### 8.9 加速度相关代价的当前状态

环境当前计算并写入 `info` 的加速度代价为：

\[
p^{\mathrm{acc}}_{i,t}
=0.01\left\|\mathbf a^{\mathrm{applied}}_{i,t}\right\|_2^2
\]

加速度裁剪代价为：

\[
p^{\mathrm{clip}}_{i,t}
=0.05\left\|
\mathbf a^{\mathrm{commanded}}_{i,t}
-\mathbf a^{\mathrm{applied}}_{i,t}
\right\|_2^2
\]

但是，奖励组合代码中的 `- acceleration_penalties` 和 `- acceleration_clip_penalties` 当前均已被注释。因此，这两个量只用于记录与分析，不会影响策略梯度更新。验证阶段第 3 节和第 4 节定义的平滑度与控制代价属于评估指标，同样不等价于训练奖励项。

### 8.10 当前参数与启用状态

| 奖励或机制 | 配置字段 | 当前值 | 当前状态 |
|---|---|---:|---|
| 目标进度权重 | `step_reward_weight` | 4.0 | 已计入奖励 |
| 固定步长惩罚 | `step_penalty` | 0.01 | 已计入奖励 |
| 一级近目标半径/奖励 | `near_goal_bonus_radius_1` / `near_goal_bonus_1` | 1.0 / 0.2 | 已计入奖励 |
| 二级近目标半径/奖励 | `near_goal_bonus_radius_2` / `near_goal_bonus_2` | 0.6 / 0.6 | 已计入奖励，与一级累加 |
| 障碍物势场权重 | `obstacle_potential_weight` | 2.0 | 已计入奖励 |
| 障碍物影响距离 | `obstacle_influence_distance` | 1.5 | 已启用 |
| 障碍物势场上限 | `obstacle_potential_penalty_max` | 20.0 | 已启用 |
| 边界势场权重 | `boundary_potential_weight` | 0.3 | 已计入奖励 |
| 边界影响距离 | `boundary_influence_distance` | 0.6 | 已启用 |
| 边界势场上限 | `boundary_potential_penalty_max` | 20.0 | 已启用 |
| 边界距离下限 | `boundary_distance_epsilon` | \(10^{-3}\) | 已启用 |
| 智能体间势场权重 | `inter_agent_potential_weight` | 1.0 | 已计入奖励 |
| 智能体间影响距离 | `inter_agent_influence_distance` | 1.2 | 已启用 |
| 障碍物/边界碰撞惩罚 | `collision_penalty` | 80.0 | 已计入奖励 |
| 智能体间碰撞惩罚 | `inter_agent_collision_penalty` | 20.0 | 已计入奖励 |
| 成功奖励总尺度 | `success_bonus` | 600.0 | 按智能体数量均分 |
| 超时惩罚权重 | `timeout_penalty` | 20.0 | 已计入奖励 |
| 加速度惩罚权重 | `acceleration_penalty_weight` | 0.01 | 已计算，未计入奖励 |
| 加速度裁剪惩罚权重 | `acceleration_clip_penalty_weight` | 0.05 | 已计算，未计入奖励 |
| 动作引导 | `action_guidance_enabled` | `False` | 已关闭，且不属于奖励项 |

### 8.11 折扣回报

MASAC 当前使用的折扣因子为 \(\gamma=0.95\)。Critic 学习的目标不是单步即时奖励，而是折扣累计回报：

\[
G_{i,t}
=
\sum_{k=0}^{\infty}\gamma^k r_{i,t+k}
\]

因此，成功奖励、碰撞惩罚和超时惩罚会通过 bootstrapping 影响终止事件之前的 Q 值估计。\(\gamma\) 属于训练回报参数，不是即时 reward function 的独立组成项。

### 8.12 奖励分量记录字段

环境通过 `info` 分别输出各类奖励分量，用于训练诊断和实验复核：

- `reward_step`：目标进度奖励；
- `reward_near_goal_bonus`：近目标区域奖励；
- `reward_obstacle_potential_penalty`：障碍物势场惩罚；
- `reward_boundary_potential_penalty`：边界势场惩罚；
- `reward_inter_agent_potential_penalty`：智能体间势场惩罚；
- `reward_acceleration_penalty`：已计算但未计入总奖励的加速度代价；
- `reward_acceleration_clip_penalty`：已计算但未计入总奖励的裁剪代价；
- `reward_collision_penalty`：当前步累计碰撞惩罚；
- `reward_timeout_penalty`：超时距离惩罚。

这些字段用于形成“总奖励—分量贡献—终止事件”的对应关系，从而判断训练不稳定是由任务进度不足、安全惩罚过强、终止奖励尺度不匹配，还是控制输出饱和造成。
