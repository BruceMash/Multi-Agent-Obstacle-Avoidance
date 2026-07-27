from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Environment.multi_agent_dmp_env import MultiAgentEnvConfig


@dataclass
class MASACNetworkConfig:
    """Architecture parameters shared by MASAC actors and critics."""

    sensor_observation_dim: int | None = 439
    extra_observation_dim: int = 3
    ally_feature_dim: int = 7
    sensor_output_dim: int = 128
    ally_output_dim: int = 64
    sensor_hidden_dim: int = 128
    ally_hidden_dim: int = 128
    hidden_dim: int = 256
    num_sensor_layers: int = 2
    num_ally_layers: int = 2
    num_observation_layers: int = 2
    sensor_azimuth_bins: int = 24
    sensor_elevation_bins: int = 9
    sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0)
    sensor_include_previous_scan: bool = True
    ally_pooling: str = "mean_max"
    agent_pooling: str = "mean_max"
    critic_encoder: str = "attention"
    actor_log_std_min: float = -20.0
    actor_log_std_max: float = 2.0
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    temporal_steps: int = 4
    goal_distance_clip: float = 1.0
    dmp_k_alpha: float = 1.0
    dmp_k_beta: float = 1.0
    dmp_tau: float = 1.0
    forcing_term_min: float = -10.0
    forcing_term_max: float = 10.0
    acceleration_low: tuple[float, ...] | None = None
    acceleration_high: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.sensor_observation_dim is not None:
            self.sensor_observation_dim = int(self.sensor_observation_dim)
            if self.sensor_observation_dim <= 0:
                raise ValueError("sensor_observation_dim must be positive")

        integer_fields = (
            "extra_observation_dim",
            "ally_feature_dim",
            "sensor_output_dim",
            "ally_output_dim",
            "sensor_hidden_dim",
            "ally_hidden_dim",
            "hidden_dim",
            "num_sensor_layers",
            "num_ally_layers",
            "num_observation_layers",
            "sensor_azimuth_bins",
            "sensor_elevation_bins",
            "temporal_steps",
        )
        for field_name in integer_fields:
            setattr(self, field_name, int(getattr(self, field_name)))

        if self.extra_observation_dim < 0 or self.ally_feature_dim < 0:
            raise ValueError("observation block dimensions must be non-negative")
        if self.sensor_output_dim <= 0 or self.ally_output_dim <= 0:
            raise ValueError("encoder output dimensions must be positive")
        if self.sensor_hidden_dim <= 0 or self.ally_hidden_dim <= 0:
            raise ValueError("encoder hidden dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.sensor_azimuth_bins <= 0 or self.sensor_elevation_bins <= 0:
            raise ValueError("sensor ray bins must be positive")
        if self.temporal_steps <= 0 or self.temporal_steps > 5:
            raise ValueError("temporal_steps must be in [1, 5]")
        if min(
            self.num_sensor_layers,
            self.num_ally_layers,
            self.num_observation_layers,
        ) <= 0:
            raise ValueError("network layer counts must be positive")

        self.sensor_elevation_range_deg = tuple(
            float(value) for value in self.sensor_elevation_range_deg
        )
        self.sensor_include_previous_scan = bool(self.sensor_include_previous_scan)
        if (
            len(self.sensor_elevation_range_deg) != 2
            or self.sensor_elevation_range_deg[0] >= self.sensor_elevation_range_deg[1]
        ):
            raise ValueError("sensor_elevation_range_deg must be increasing")

        valid_pooling = {"mean", "max", "mean_max"}
        if self.ally_pooling not in valid_pooling:
            raise ValueError(f"unsupported ally_pooling: {self.ally_pooling}")
        if self.agent_pooling not in valid_pooling:
            raise ValueError(f"unsupported agent_pooling: {self.agent_pooling}")
        self.critic_encoder = str(self.critic_encoder)
        if self.critic_encoder not in {"attention", "mlp"}:
            raise ValueError("critic_encoder must be 'attention' or 'mlp'")

        self.actor_log_std_min = float(self.actor_log_std_min)
        self.actor_log_std_max = float(self.actor_log_std_max)
        if self.actor_log_std_min >= self.actor_log_std_max:
            raise ValueError("actor_log_std_min must be smaller than actor_log_std_max")

        self.goal_distance_clip = float(self.goal_distance_clip)
        self.dmp_k_alpha = float(self.dmp_k_alpha)
        self.dmp_k_beta = float(self.dmp_k_beta)
        self.dmp_tau = float(self.dmp_tau)
        self.forcing_term_min = float(self.forcing_term_min)
        self.forcing_term_max = float(self.forcing_term_max)
        if self.goal_distance_clip <= 0.0:
            raise ValueError("goal_distance_clip must be positive")
        if self.dmp_tau <= 0.0:
            raise ValueError("dmp_tau must be positive")
        if self.forcing_term_min >= self.forcing_term_max:
            raise ValueError("forcing_term_min must be smaller than forcing_term_max")

        if (self.action_low is None) != (self.action_high is None):
            raise ValueError("action_low and action_high must be both set or both None")
        if self.action_low is not None:
            self.action_low = tuple(float(value) for value in self.action_low)
            self.action_high = tuple(float(value) for value in self.action_high)
            if len(self.action_low) != len(self.action_high):
                raise ValueError("action_low and action_high must have the same length")
            if any(low >= high for low, high in zip(self.action_low, self.action_high)):
                raise ValueError("each action_low value must be smaller than action_high")

        if (self.acceleration_low is None) != (self.acceleration_high is None):
            raise ValueError("acceleration_low and acceleration_high must be both set or both None")
        if self.acceleration_low is not None:
            self.acceleration_low = tuple(float(value) for value in self.acceleration_low)
            self.acceleration_high = tuple(float(value) for value in self.acceleration_high)
            if len(self.acceleration_low) != len(self.acceleration_high):
                raise ValueError("acceleration_low and acceleration_high must have the same length")
            if any(low >= high for low, high in zip(self.acceleration_low, self.acceleration_high)):
                raise ValueError("each acceleration_low value must be smaller than acceleration_high")

    def encoder_kwargs(self) -> dict:
        return {
            "sensor_observation_dim": self.sensor_observation_dim,
            "extra_observation_dim": self.extra_observation_dim,
            "ally_feature_dim": self.ally_feature_dim,
            "sensor_output_dim": self.sensor_output_dim,
            "ally_output_dim": self.ally_output_dim,
            "sensor_hidden_dim": self.sensor_hidden_dim,
            "ally_hidden_dim": self.ally_hidden_dim,
            "hidden_dim": self.hidden_dim,
            "num_sensor_layers": self.num_sensor_layers,
            "num_ally_layers": self.num_ally_layers,
            "num_observation_layers": self.num_observation_layers,
            "sensor_azimuth_bins": self.sensor_azimuth_bins,
            "sensor_elevation_bins": self.sensor_elevation_bins,
            "sensor_elevation_range_deg": self.sensor_elevation_range_deg,
            "sensor_include_previous_scan": self.sensor_include_previous_scan,
            "ally_pooling": self.ally_pooling,
        }

    def actor_kwargs(self) -> dict:
        kwargs = self.encoder_kwargs()
        kwargs.update(
            log_std_min=self.actor_log_std_min,
            log_std_max=self.actor_log_std_max,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        return kwargs

    def critic_kwargs(self) -> dict:
        kwargs = self.encoder_kwargs()
        kwargs["agent_pooling"] = self.agent_pooling
        kwargs["critic_encoder"] = self.critic_encoder
        return kwargs


@dataclass(frozen=True)
class MASACExperimentConfig:
    """Experiment parameters for MASAC on MultiAgentDMPEnv."""

    # Environment identity
    environment_name: str = "multi_agent_dmp"
    map_name: str = "default"
    hyperparam_source: str = "masac"

    # Network architecture
    hidden_dim: int = 256
    sensor_hidden_dim: int = 128
    ally_hidden_dim: int = 128
    sensor_output_dim: int = 64
    ally_output_dim: int = 64
    num_sensor_layers: int = 2
    num_ally_layers: int = 2
    num_observation_layers: int = 2
    actor_log_std_min: float = -20.0
    actor_log_std_max: float = 2.0
    ally_pooling: str = "mean_max"
    agent_pooling: str = "mean_max"
    critic_encoder: str = "attention"
    temporal_steps: int = 3

    # Environment and dynamics
    num_agents: int = 3
    velocity_clip: tuple[float, float] = (-4.0, 4.0)
    accelerate_clip: tuple[float, float] = (-4.0, 4.0)
    time_step: float = 0.1
    sensing_radius: float = 5.0
    sensor_azimuth_bins: int = 16
    sensor_elevation_bins: int = 16
    sensor_elevation_range_deg: tuple[float, float] = (-80.0, 80.0)
    sensor_include_previous_scan: bool = False
    sensor_goal_distance_clip: float | None = None
    max_steps: int = 200
    goal_tolerance: float = 0.3
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (9.0, 4.5, 2.4),
    )
    randomize_start_goal: bool = True
    start_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.4),
        (1.3, 4.0, 2.0),
    )
    goal_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (7.7, 0.5, 0.4),
        (8.5, 4.0, 2.0),
    )
    min_start_distance: float = 0.6
    min_goal_distance: float = 0.0
    min_start_goal_distance: float = 6.0
    start_goal_max_attempts: int = 1000

    # 奖励与安全参数
    obstacle_potential_weight: float = 0.3  # 障碍物人工势场的整体权重，直接缩放每步障碍物惩罚。
    obstacle_influence_distance: float = 1.2  # 障碍物表面距离小于该值（米）时才计算势场惩罚。
    obstacle_potential_penalty_max: float = 2.0  # 单个智能体每步障碍物势场惩罚的上限。
    boundary_influence_distance: float = 0.5  # 智能体距离工作空间边界小于该值（米）时启用边界势场。
    boundary_potential_weight: float = 0.1  # 边界人工势场的整体权重。
    boundary_potential_penalty_max: float = 2.0  # 单个智能体每步边界势场惩罚的上限。
    boundary_distance_epsilon: float = 1e-3  # 边界势场计算中的最小距离，防止除零和数值爆炸。
    step_reward_weight: float = 4.0  # 目标距离每缩短一米获得的进度奖励系数。

    # 以下旧奖励字段仅为配置兼容保留；设为零时不影响当前奖励。
    step_penalty: float = 0.0  # 旧版固定每步惩罚，当前关闭。
    collision_penalty: float = 0.0  # 旧版统一碰撞惩罚，已由团队和局部碰撞惩罚替代。
    timeout_penalty: float = 0.0  # 旧版统一超时惩罚，已由 team_timeout_penalty 替代。
    success_bonus: float = 0.0  # 旧版统一成功奖励，已由个人和团队成功奖励替代。

    individual_success_bonus: float = 50.0  # 单个智能体首次到达自身目标时获得的一次性奖励。
    team_success_bonus: float = 150.0  # 全体智能体均到达目标时，每个智能体获得的一次性团队奖励。
    team_collision_penalty: float = 100.0  # 任意碰撞终止时，对所有智能体施加的统一惩罚。
    local_obstacle_collision_penalty: float = 40.0  # 与障碍物碰撞的责任智能体额外承担的惩罚。
    local_boundary_collision_penalty: float = 40.0  # 与环境边界碰撞的责任智能体额外承担的惩罚。
    local_inter_agent_collision_penalty: float = 20.0  # 智能体互撞时，涉事智能体各自额外承担的惩罚。
    team_timeout_penalty: float = 50.0  # episode 达到最大步数仍未完成时，每个智能体承担的惩罚。
    collision_margin: float = 0.0  # 碰撞检测的附加安全膨胀距离；零表示使用几何体原始边界。

    action_guidance_enabled: bool = False  # 是否在接近目标时逐渐削弱策略残差动作，让 DMP 基础轨迹主导收敛。
    action_guidance_radius: float = 1.0  # 距离目标小于该半径（米）时允许启用动作引导。
    action_guidance_initial_weight: float = 0.7  # 训练开始时接近目标后对策略残差动作的最大削弱权重。
    action_guidance_decay_steps: int = 500_000  # 引导权重线性衰减到零所需的环境步数。

    near_goal_bonus_radius_1: float = 0.0  # 第一层近目标奖励半径；零表示关闭该层奖励。
    near_goal_bonus_1: float = 0.0  # 进入第一层近目标区域时的每步奖励；当前关闭。
    near_goal_bonus_radius_2: float = 0.0  # 第二层近目标奖励半径；零表示关闭该层奖励。
    near_goal_bonus_2: float = 0.0  # 进入第二层近目标区域时的每步奖励；当前关闭。

    inter_agent_safe_distance: float = 0.6  # 两智能体中心距离低于该值（米）时判定发生互撞。
    inter_agent_collision_penalty: float = 0.0  # 旧版互撞惩罚字段，当前由团队和局部碰撞惩罚替代。
    inter_agent_potential_weight: float = 1.0  # 智能体间人工势场惩罚的整体权重。
    inter_agent_influence_distance: float = 1.2  # 智能体间距离小于该值（米）时启用智能体间势场。
    inter_agent_potential_penalty_max: float = 3.0  # 单个智能体每步智能体间势场惩罚的上限。

    stagnation_window: int = 30  # 用最近多少步的目标距离变化判断是否停滞。
    stagnation_progress_threshold: float = 0.08  # 窗口累计进度低于该值（米）时记为一次停滞。
    stagnation_patience: int = 10  # 连续多少次满足停滞条件后才开始施加惩罚。
    stagnation_penalty_start: float = 0.2  # 停滞惩罚首次触发时的每步惩罚值。
    stagnation_penalty_growth: float = 0.02  # 持续停滞时，每经过一步增加的惩罚值。
    stagnation_penalty_max: float = 0.5  # 单个智能体每步停滞惩罚的上限。

    nearest_agent_observation_count: int | None = None  # 观测中保留的最近队友数；None 表示保留全部队友。
    acceleration_penalty_weight: float = 0.01  # 加速度平方惩罚的计算权重；当前仅计算和记录，未计入总奖励。
    acceleration_clip_penalty_weight: float = 0.05  # 加速度被裁剪部分的惩罚权重；当前仅计算和记录。

    # 基于成功率的课程场景参数
    curriculum_enabled: bool = True  # 是否启用按成功率逐级增加障碍物难度的课程。
    curriculum_success_threshold: float = 0.8  # 滚动成功率达到该阈值后晋级到下一课程阶段。
    curriculum_phase2_box_counts: tuple[int, ...] = (1, 2, 3)  # phase 2 各等级的地面箱体数量。
    curriculum_phase2_sphere_counts: tuple[int, ...] = (1, 2, 3)  # phase 2 各等级的空中静态球数量。
    curriculum_phase3_dynamic_counts: tuple[int, ...] = (1, 2, 3)  # phase 3 各等级的动态球数量。
    curriculum_box_half_extent_range: tuple[float, float] = (0.20, 0.38)  # 箱体水平半尺寸的随机范围（米）。
    curriculum_box_height_range: tuple[float, float] = (0.45, 1.10)  # 箱体高度的随机范围（米）。
    curriculum_aerial_sphere_radius_range: tuple[float, float] = (0.18, 0.30)  # 空中静态球半径的随机范围（米）。
    curriculum_dynamic_sphere_radius_range: tuple[float, float] = (0.20, 0.30)  # 动态球半径的随机范围（米）。
    curriculum_dynamic_speed_range: tuple[float, float] = (0.25, 0.60)  # 动态球移动速度的随机范围（米/秒）。
    curriculum_aerial_min_center_height: float = 0.65  # 空中障碍物球心允许的最低高度（米）。
    curriculum_obstacle_safety_margin: float = 0.05  # 场景生成时障碍物外额外预留的安全间距（米）。
    curriculum_start_goal_clearance: float = 0.55  # 起点和目标点到障碍物至少保留的净空距离（米）。
    curriculum_obstacle_separation: float = 0.12  # 不同障碍物表面之间至少保留的间距（米）。
    curriculum_placement_attempts: int = 1000  # 每次生成场景时放置障碍物的最大随机尝试次数。
    curriculum_curved_turn_rate: float = 0.45  # 动态障碍物曲线运动的最大转向速率。
    curriculum_wandering_strength: float = 0.8  # 动态障碍物随机游走方向扰动的强度。

    # DMP
    dmp_dims: int = 3
    k_alpha: float = 3.2
    k_beta: float = 0.8
    alpha_s: float = 4.0
    dmp_tau: float = 2.5
    forcing_term_max: float = 10.0
    forcing_term_min: float = -10.0
    goal_offset_max: float = 1.0

    # MASAC training
    seed: int = 321
    total_steps: int = 500_000
    start_steps: int = 5_000
    # 随机采样期间提前更新网络，避免 start_steps 结束时切换到完全未训练的 Actor。
    learning_starts: int = 1_000
    # Actor 接管行为策略后继续短暂混合随机动作，避免动作分布在单步内突变。
    policy_transition_steps: int = 5_000
    batch_size: int = 256
    buffer_size: int = 1_000_000
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    gamma: float = 0.95
    soft_update_tau: float = 0.01
    learn_interval: int = 1
    updates_per_step: int = 1
    device: str = "auto"
    output_root: str = "artifacts/masac"
    save_interval: int = 50_000
    log_interval: int = 1_000
    progress_interval: int = 100
    success_window: int = 100
    disable_tensorboard: bool = False
    plain_progress: bool = False

    def build_dynamics_config(self) -> dict[str, Any]:
        return {
            "velocity_clip": self.velocity_clip,
            "accelerate_clip": self.accelerate_clip,
            "time_step": self.time_step,
        }

    def build_sensor_config(self) -> dict[str, Any]:
        sensor_config: dict[str, Any] = {
            "sensing_radius": self.sensing_radius,
            "azimuth_bins": self.sensor_azimuth_bins,
            "elevation_bins": self.sensor_elevation_bins,
            "elevation_range_deg": self.sensor_elevation_range_deg,
            "include_previous_scan": self.sensor_include_previous_scan,
        }
        if self.sensor_goal_distance_clip is not None:
            sensor_config["goal_distance_clip"] = self.sensor_goal_distance_clip
        return sensor_config

    def build_dmp_config(self) -> dict[str, Any]:
        return {
            "dt": self.time_step,
            "dims": self.dmp_dims,
            "K_alpha": self.k_alpha,
            "K_beta": self.k_beta,
            "alpha_s": self.alpha_s,
            "tau": self.dmp_tau,
            "forcing_term_max": self.forcing_term_max,
            "forcing_term_min": self.forcing_term_min,
            "goal_offset_max": self.goal_offset_max,
        }

    def build_env_config(self) -> MultiAgentEnvConfig:
        return MultiAgentEnvConfig(
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            goal_tolerance=self.goal_tolerance,
            obstacle_potential_weight=self.obstacle_potential_weight,
            obstacle_influence_distance=self.obstacle_influence_distance,
            obstacle_potential_penalty_max=self.obstacle_potential_penalty_max,
            step_reward_weight=self.step_reward_weight,
            step_penalty=self.step_penalty,
            collision_penalty=self.collision_penalty,
            timeout_penalty=self.timeout_penalty,
            success_bonus=self.success_bonus,
            individual_success_bonus=self.individual_success_bonus,
            team_success_bonus=self.team_success_bonus,
            team_collision_penalty=self.team_collision_penalty,
            local_obstacle_collision_penalty=self.local_obstacle_collision_penalty,
            local_boundary_collision_penalty=self.local_boundary_collision_penalty,
            local_inter_agent_collision_penalty=self.local_inter_agent_collision_penalty,
            team_timeout_penalty=self.team_timeout_penalty,
            collision_margin=self.collision_margin,
            workspace_bounds=self.workspace_bounds,
            randomize_start_goal=self.randomize_start_goal,
            start_position_bounds=self.start_position_bounds,
            goal_position_bounds=self.goal_position_bounds,
            min_start_distance=self.min_start_distance,
            min_goal_distance=self.min_goal_distance,
            min_start_goal_distance=self.min_start_goal_distance,
            start_goal_max_attempts=self.start_goal_max_attempts,
            boundary_influence_distance=self.boundary_influence_distance,
            boundary_potential_weight=self.boundary_potential_weight,
            boundary_potential_penalty_max=self.boundary_potential_penalty_max,
            boundary_distance_epsilon=self.boundary_distance_epsilon,
            action_guidance_enabled=self.action_guidance_enabled,
            action_guidance_radius=self.action_guidance_radius,
            action_guidance_initial_weight=self.action_guidance_initial_weight,
            action_guidance_decay_steps=self.action_guidance_decay_steps,
            near_goal_bonus_radius_1=self.near_goal_bonus_radius_1,
            near_goal_bonus_1=self.near_goal_bonus_1,
            near_goal_bonus_radius_2=self.near_goal_bonus_radius_2,
            near_goal_bonus_2=self.near_goal_bonus_2,
            inter_agent_safe_distance=self.inter_agent_safe_distance,
            inter_agent_collision_penalty=self.inter_agent_collision_penalty,
            inter_agent_potential_weight=self.inter_agent_potential_weight,
            inter_agent_influence_distance=self.inter_agent_influence_distance,
            inter_agent_potential_penalty_max=self.inter_agent_potential_penalty_max,
            stagnation_window=self.stagnation_window,
            stagnation_progress_threshold=self.stagnation_progress_threshold,
            stagnation_patience=self.stagnation_patience,
            stagnation_penalty_start=self.stagnation_penalty_start,
            stagnation_penalty_growth=self.stagnation_penalty_growth,
            stagnation_penalty_max=self.stagnation_penalty_max,
            nearest_agent_observation_count=self.nearest_agent_observation_count,
            acceleration_penalty_weight=self.acceleration_penalty_weight,
            acceleration_clip_penalty_weight=self.acceleration_clip_penalty_weight,
        )

    def build_env_config_dict(self) -> dict[str, Any]:
        return dict(vars(self.build_env_config()))

    def build_core_env_kwargs(self) -> dict[str, Any]:
        return {
            "dynamics_config": self.build_dynamics_config(),
            "sensor_config": self.build_sensor_config(),
            "dmp_config": self.build_dmp_config(),
            "env_config": self.build_env_config_dict(),
        }


MASAC_EXPERIMENT_CONFIG = MASACExperimentConfig()
