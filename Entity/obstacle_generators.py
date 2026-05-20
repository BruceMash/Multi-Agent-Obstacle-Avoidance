import numpy as np

from Entity.dynamic_obstacles import MovingSphereObstacle
from Entity.static_obstacles import AxisAlignedBoxObstacle, StaticCylinderObstacle, StaticSphereObstacle


def _check_overlap_with_existing(candidate_center, existing_obstacles, self_effective_radius):
    for obstacle in existing_obstacles:
        min_dist = self_effective_radius + obstacle.effective_radius + 0.2
        if float(np.linalg.norm(candidate_center - obstacle.center)) < min_dist:
            return True
    return False


def _axis_aligned_boxes_overlap(first_center, first_half_extents, second_center, second_half_extents, clearance=0.2):
    return bool(
        np.all(
            np.abs(first_center - second_center)
            < first_half_extents + second_half_extents + float(clearance)
        )
    )


_DEFAULT_PROTECTED_POINTS = [
    np.array([0.0, 0.0, 0.0], dtype=float),
    np.array([8.0, 0.0, 0.0], dtype=float),
]


def _resolve_protected_points(protected_points):
    if protected_points is None:
        protected_points = _DEFAULT_PROTECTED_POINTS

    resolved = []
    for point in protected_points:
        vector = np.asarray(point, dtype=float)
        if vector.shape != (3,):
            raise ValueError("each protected point must have shape (3,)")
        resolved.append(vector.copy())
    return resolved


def _resolve_rng(seed): # 随机种子
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _validate_bounds(name, value):
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError(f"{name} must be a 2x3 array-like value: [lower_bound, upper_bound]")

    lower_bound = bounds[0]
    upper_bound = bounds[1]
    if np.any(lower_bound >= upper_bound):
        raise ValueError(f"each {name} lower bound must be smaller than upper bound")
    return lower_bound, upper_bound


def StaticSpherePositionGenerate(center, radius, safety_margin, num,
                                 existing_obstacles=None, seed=None,
                                 protected_points=None):
    """
    随机生成训练用静态球形障碍物。

    参数说明：
    - center: 障碍物中心点的采样范围，形如 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    - radius: 球形障碍物半径
    - safety_margin: 安全边界
    - num: 需要生成的障碍物数量
    - existing_obstacles: 已存在的障碍物列表，新生成障碍物会避让它们
    - seed: 随机种子，传入相同值可复现相同结果
    - protected_points: 需要避开的位置列表，默认保护起点 [0,0,0] 和终点 [8,0,0]
    """
    bounds = np.asarray(center, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError("center must be a 2x3 array-like value: [lower_bound, upper_bound]")

    lower_bound = bounds[0]
    upper_bound = bounds[1]
    if np.any(lower_bound >= upper_bound):
        raise ValueError("each lower bound must be smaller than upper bound")

    obstacle_radius = float(radius)
    obstacle_safety_margin = float(safety_margin)
    obstacle_num = int(num)
    if obstacle_radius <= 0.0:
        raise ValueError("radius must be positive")
    if obstacle_safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative")
    if obstacle_num < 0:
        raise ValueError("num must be non-negative")

    rng = _resolve_rng(seed)
    obstacles = []
    effective_radius = obstacle_radius + obstacle_safety_margin

    protected = _resolve_protected_points(protected_points)
    protected_clearance = effective_radius + 0.8

    existing = list(existing_obstacles) if existing_obstacles is not None else []

    max_attempts = max(1000, obstacle_num * 200)
    for _ in range(max_attempts):
        if len(obstacles) >= obstacle_num:
            break

        candidate_center = rng.uniform(lower_bound, upper_bound)

        if any(np.linalg.norm(candidate_center - point) < protected_clearance for point in protected):
            continue

        if _check_overlap_with_existing(candidate_center, existing, effective_radius):
            continue

        min_center_distance = 2.0 * effective_radius + 0.2
        if any(np.linalg.norm(candidate_center - obstacle.center) < min_center_distance for obstacle in obstacles):
            continue

        obstacle = StaticSphereObstacle(
            center=candidate_center,
            radius=obstacle_radius,
            safety_margin=obstacle_safety_margin,
        ) 
        obstacles.append(obstacle)
        existing.append(obstacle)

    if len(obstacles) != obstacle_num:
        raise RuntimeError(
            f"failed to generate {obstacle_num} static sphere obstacles within {max_attempts} attempts"
        )
    return obstacles


def StaticBoxPositionGenerate(center, half_extents, safety_margin, num,
                              existing_obstacles=None, seed=None,
                              protected_points=None):
    """
    随机生成训练用静态长方体障碍物（轴对齐）。

    参数说明：
    - center: 障碍物中心点的采样范围，形如 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    - half_extents: 长方体半边长 [hx, hy, hz]
    - safety_margin: 安全边界
    - num: 需要生成的障碍物数量
    - existing_obstacles: 已存在的障碍物列表，新生成障碍物会避让它们
    - seed: 随机种子，传入相同值可复现相同结果
    - protected_points: 需要避开的位置列表，默认保护起点 [0,0,0] 和终点 [8,0,0]
    """
    bounds = np.asarray(center, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError("center must be a 2x3 array-like value: [lower_bound, upper_bound]")

    lower_bound = bounds[0]
    upper_bound = bounds[1]
    if np.any(lower_bound >= upper_bound):
        raise ValueError("each lower bound must be smaller than upper bound")

    half_extents = np.asarray(half_extents, dtype=float)
    if half_extents.shape != (3,):
        raise ValueError("half_extents must have shape (3,)")
    if np.any(half_extents <= 0.0):
        raise ValueError("half_extents must be positive")

    obstacle_safety_margin = float(safety_margin)
    obstacle_num = int(num)
    if obstacle_safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative")
    if obstacle_num < 0:
        raise ValueError("num must be non-negative")

    rng = _resolve_rng(seed)
    obstacles = []

    expanded_half = half_extents + obstacle_safety_margin
    effective_radius = float(np.linalg.norm(expanded_half))

    protected = _resolve_protected_points(protected_points)
    protected_clearance = effective_radius + 0.8

    existing = list(existing_obstacles) if existing_obstacles is not None else []

    max_attempts = max(1000, obstacle_num * 200)
    for _ in range(max_attempts):
        if len(obstacles) >= obstacle_num:
            break

        candidate_center = rng.uniform(lower_bound, upper_bound)

        if any(np.linalg.norm(candidate_center - point) < protected_clearance for point in protected):
            continue

        if _check_overlap_with_existing(candidate_center, existing, effective_radius):
            continue

        if any(
            _axis_aligned_boxes_overlap(
                candidate_center,
                expanded_half,
                obstacle.center,
                obstacle.expanded_half_extents,
            )
            for obstacle in obstacles
        ):
            continue

        obstacle = AxisAlignedBoxObstacle(
            center=candidate_center,
            half_extents=half_extents.copy(),
            safety_margin=obstacle_safety_margin,
        )
        obstacles.append(obstacle)
        existing.append(obstacle)

    if len(obstacles) != obstacle_num:
        raise RuntimeError(
            f"failed to generate {obstacle_num} static box obstacles within {max_attempts} attempts"
        )
    return obstacles


def StaticCylinderPositionGenerate(center, radius, half_height, safety_margin, num,
                                   existing_obstacles=None, seed=None,
                                   protected_points=None):
    """
    随机生成训练用静态圆柱体障碍物（Z 轴对齐）。

    参数说明：
    - center: 障碍物中心点的采样范围，形如 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    - radius: 圆柱体圆截面半径
    - half_height: 圆柱体半高
    - safety_margin: 安全边界
    - num: 需要生成的障碍物数量
    - existing_obstacles: 已存在的障碍物列表，新生成障碍物会避让它们
    - seed: 随机种子，传入相同值可复现相同结果
    - protected_points: 需要避开的位置列表，默认保护起点 [0,0,0] 和终点 [8,0,0]
    """
    bounds = np.asarray(center, dtype=float)
    if bounds.shape != (2, 3):
        raise ValueError("center must be a 2x3 array-like value: [lower_bound, upper_bound]")

    lower_bound = bounds[0]
    upper_bound = bounds[1]
    if np.any(lower_bound >= upper_bound):
        raise ValueError("each lower bound must be smaller than upper bound")

    obstacle_radius = float(radius)
    obstacle_half_height = float(half_height)
    obstacle_safety_margin = float(safety_margin)
    obstacle_num = int(num)
    if obstacle_radius <= 0.0:
        raise ValueError("radius must be positive")
    if obstacle_half_height <= 0.0:
        raise ValueError("half_height must be positive")
    if obstacle_safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative")
    if obstacle_num < 0:
        raise ValueError("num must be non-negative")

    rng = _resolve_rng(seed)
    obstacles = []

    effective_xy_radius = obstacle_radius + obstacle_safety_margin
    effective_half_height = obstacle_half_height + obstacle_safety_margin
    bounding_sphere_radius = float(
        np.sqrt(effective_xy_radius * effective_xy_radius + effective_half_height * effective_half_height)
    )

    protected = _resolve_protected_points(protected_points)
    protected_clearance = bounding_sphere_radius + 0.8

    existing = list(existing_obstacles) if existing_obstacles is not None else []

    max_attempts = max(1000, obstacle_num * 200)
    for _ in range(max_attempts):
        if len(obstacles) >= obstacle_num:
            break

        candidate_center = rng.uniform(lower_bound, upper_bound)

        if any(np.linalg.norm(candidate_center - point) < protected_clearance for point in protected):
            continue

        if _check_overlap_with_existing(candidate_center, existing, bounding_sphere_radius):
            continue

        collision = False
        for obstacle in obstacles:
            xy_distance = float(np.linalg.norm(candidate_center[:2] - obstacle.center[:2]))
            if xy_distance < 2.0 * effective_xy_radius + 0.2:
                z_distance = abs(candidate_center[2] - obstacle.center[2])
                if z_distance < 2.0 * effective_half_height + 0.2:
                    collision = True
                    break
        if collision:
            continue

        obstacle = StaticCylinderObstacle(
            center=candidate_center,
            radius=obstacle_radius,
            half_height=obstacle_half_height,
            safety_margin=obstacle_safety_margin,
        )
        obstacles.append(obstacle)
        existing.append(obstacle)

    if len(obstacles) != obstacle_num:
        raise RuntimeError(
            f"failed to generate {obstacle_num} static cylinder obstacles within {max_attempts} attempts"
        )
    return obstacles


def DynamicSpherePositionGenerate(center, radius, velocity, safety_margin, num,
                                  movement_bounds=None, existing_obstacles=None,
                                  seed=None, protected_points=None, min_speed=0.0):
    """
    随机生成训练用动态球形障碍物。

    参数说明：
    - center: 初始中心点采样范围，形如 [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    - radius: 球形障碍物半径
    - velocity: 初始速度采样范围，形如 [[vx_min, vy_min, vz_min], [vx_max, vy_max, vz_max]]
    - safety_margin: 安全边界
    - num: 需要生成的障碍物数量
    - movement_bounds: 动态障碍物运动边界，None 表示不限制反弹范围
    - existing_obstacles: 已存在的障碍物列表，新生成障碍物会避让它们
    - seed: 随机种子，传入相同值可复现相同结果
    - protected_points: 需要避开的位置列表，默认保护起点 [0,0,0] 和终点 [8,0,0]
    - min_speed: 初始速度模长下限，避免生成几乎静止的动态障碍物
    """
    lower_bound, upper_bound = _validate_bounds("center", center)
    velocity_lower, velocity_upper = _validate_bounds("velocity", velocity)

    obstacle_radius = float(radius)
    obstacle_safety_margin = float(safety_margin)
    obstacle_num = int(num)
    speed_floor = float(min_speed)
    if obstacle_radius <= 0.0:
        raise ValueError("radius must be positive")
    if obstacle_safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative")
    if obstacle_num < 0:
        raise ValueError("num must be non-negative")
    if speed_floor < 0.0:
        raise ValueError("min_speed must be non-negative")

    obstacle_bounds = None
    if movement_bounds is not None:
        movement_lower, movement_upper = _validate_bounds("movement_bounds", movement_bounds)
        obstacle_bounds = (movement_lower.copy(), movement_upper.copy())

    rng = _resolve_rng(seed)
    obstacles = []
    effective_radius = obstacle_radius + obstacle_safety_margin

    protected = _resolve_protected_points(protected_points)
    protected_clearance = effective_radius + 0.8
    existing = list(existing_obstacles) if existing_obstacles is not None else []

    sample_lower = lower_bound.copy()
    sample_upper = upper_bound.copy()
    if obstacle_bounds is not None:
        movement_lower, movement_upper = obstacle_bounds
        sample_lower = np.maximum(sample_lower, movement_lower + effective_radius)
        sample_upper = np.minimum(sample_upper, movement_upper - effective_radius)
        if np.any(sample_lower >= sample_upper):
            raise ValueError("center sampling range must leave room for the dynamic obstacle inside movement_bounds")

    max_attempts = max(1000, obstacle_num * 200)
    for _ in range(max_attempts):
        if len(obstacles) >= obstacle_num:
            break

        candidate_center = rng.uniform(sample_lower, sample_upper)
        candidate_velocity = rng.uniform(velocity_lower, velocity_upper)

        if np.linalg.norm(candidate_velocity) < speed_floor:
            continue

        if any(np.linalg.norm(candidate_center - point) < protected_clearance for point in protected):
            continue

        if _check_overlap_with_existing(candidate_center, existing, effective_radius):
            continue

        min_center_distance = 2.0 * effective_radius + 0.2
        if any(np.linalg.norm(candidate_center - obstacle.center) < min_center_distance for obstacle in obstacles):
            continue

        obstacle = MovingSphereObstacle(
            center=candidate_center,
            radius=obstacle_radius,
            velocity=candidate_velocity,
            safety_margin=obstacle_safety_margin,
            bounds=obstacle_bounds,
        )
        obstacles.append(obstacle)
        existing.append(obstacle)

    if len(obstacles) != obstacle_num:
        raise RuntimeError(
            f"failed to generate {obstacle_num} dynamic sphere obstacles within {max_attempts} attempts"
        )
    return obstacles
