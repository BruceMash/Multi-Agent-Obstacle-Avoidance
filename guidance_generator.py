"""
Sector-risk-based 3D guiding waypoint generation demo.

This script demonstrates:
LiDAR-like point cloud -> sector threat field -> implicit risk query ->
continuous guiding waypoint generation -> visualization.

Dependencies:
    pip install numpy matplotlib
"""

# 目前仅为想法的demo简易实现，没有集成到训练框架中。

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Basic utilities
# =========================

def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a vector. Return a zero vector if its norm is too small."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def cartesian_to_spherical(points: np.ndarray):
    """
    Convert Cartesian coordinates to spherical-like coordinates.

    Args:
        points: [N, 3], local coordinates.

    Returns:
        r:     [N], distance.
        theta: [N], azimuth angle, [-pi, pi].
        phi:   [N], elevation angle, [-pi/2, pi/2].
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    return r, theta, phi


def spherical_dir(theta: float, phi: float) -> np.ndarray:
    """Convert azimuth/elevation to a 3D unit direction."""
    return np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
    ])


# =========================
# 2. Obstacle and pseudo LiDAR points
# =========================

def generate_cylinder_points(
    center: np.ndarray,
    radius: float,
    height: float,
    n_theta: int = 160,
    n_z: int = 80,
) -> np.ndarray:
    """
    Generate surface points of a vertical cylinder.
    The cylinder axis is along z.
    """
    cx, cy, cz = center

    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(cz - height / 2, cz + height / 2, n_z)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x = cx + radius * np.cos(theta_grid)
    y = cy + radius * np.sin(theta_grid)

    points = np.stack([x.ravel(), y.ravel(), z_grid.ravel()], axis=1)
    return points


# =========================
# 3. Sector threat representation
# =========================

class SectorThreatField:
    """Sector-based implicit threat/risk field."""

    def __init__(
        self,
        n_theta: int = 36,
        n_phi: int = 13,
        r_max: float = 8.0,
        lidar_phi_min: float = np.deg2rad(-45),
        lidar_phi_max: float = np.deg2rad(45),
        r_safe: float = 0.45,
        sigma_d: float = 0.8,
        sigma_r: float = 0.5,
        density_alpha: float = 0.08,
        rho_out: float = 0.45,
        rho_occl: float = 0.55,
    ):
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.r_max = r_max

        self.theta_min = -np.pi
        self.theta_max = np.pi
        self.phi_min = -np.pi / 2
        self.phi_max = np.pi / 2

        self.dtheta = (self.theta_max - self.theta_min) / n_theta
        self.dphi = (self.phi_max - self.phi_min) / n_phi

        self.lidar_phi_min = lidar_phi_min
        self.lidar_phi_max = lidar_phi_max
        self.r_safe = r_safe
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.density_alpha = density_alpha
        self.rho_out = rho_out
        self.rho_occl = rho_occl

        self.d_min = np.full((n_theta, n_phi), r_max, dtype=float)
        self.count = np.zeros((n_theta, n_phi), dtype=float)
        self.p_dist = np.zeros((n_theta, n_phi), dtype=float)
        self.p_dens = np.zeros((n_theta, n_phi), dtype=float)
        self.p_unk = np.zeros((n_theta, n_phi), dtype=float)
        self.p_threat = np.zeros((n_theta, n_phi), dtype=float)

        self.sector_dirs = self._build_sector_dirs()
        self.visible_mask = self._build_visible_mask()

    def _build_sector_dirs(self) -> np.ndarray:
        """Build unit directions for all sector centers."""
        dirs = np.zeros((self.n_theta, self.n_phi, 3), dtype=float)

        for a in range(self.n_theta):
            theta_c = self.theta_min + (a + 0.5) * self.dtheta
            for b in range(self.n_phi):
                phi_c = self.phi_min + (b + 0.5) * self.dphi
                dirs[a, b] = spherical_dir(theta_c, phi_c)

        return dirs

    def _build_visible_mask(self) -> np.ndarray:
        """Build a mask indicating whether each sector is inside the LiDAR vertical FOV."""
        visible = np.zeros((self.n_theta, self.n_phi), dtype=bool)

        for b in range(self.n_phi):
            phi_c = self.phi_min + (b + 0.5) * self.dphi
            visible[:, b] = self.lidar_phi_min <= phi_c <= self.lidar_phi_max

        return visible

    def _sector_index(self, theta, phi):
        """Return sector indices for azimuth/elevation."""
        a = np.floor((theta - self.theta_min) / self.dtheta).astype(int)
        b = np.floor((phi - self.phi_min) / self.dphi).astype(int)

        a = np.clip(a, 0, self.n_theta - 1)
        b = np.clip(b, 0, self.n_phi - 1)

        return a, b

    def build_from_lidar_points(self, lidar_points: np.ndarray) -> None:
        """
        Build sector threat from current LiDAR points.
        LiDAR points are assumed to be in the local frame.
        """
        self.d_min.fill(self.r_max)
        self.count.fill(0.0)

        r, theta, phi = cartesian_to_spherical(lidar_points)

        valid = (
            (r > 1e-6)
            & (r <= self.r_max)
            & (phi >= self.lidar_phi_min)
            & (phi <= self.lidar_phi_max)
        )

        r_valid = r[valid]
        theta_valid = theta[valid]
        phi_valid = phi[valid]

        a_idx, b_idx = self._sector_index(theta_valid, phi_valid)

        np.add.at(self.count, (a_idx, b_idx), 1.0)
        np.minimum.at(self.d_min, (a_idx, b_idx), r_valid)

        has_obs = self.count > 0

        # Distance threat
        self.p_dist.fill(0.0)
        dist_excess = self.d_min - self.r_safe
        self.p_dist[has_obs] = np.where(
            self.d_min[has_obs] <= self.r_safe,
            1.0,
            np.exp(-dist_excess[has_obs] / self.sigma_d),
        )

        # Density threat
        self.p_dens = 1.0 - np.exp(-self.density_alpha * self.count)

        # Unknown / out-of-FOV threat
        self.p_unk.fill(0.0)
        self.p_unk[~self.visible_mask] = self.rho_out

        # Composite threat probability: P = 1 - Π(1 - p_i)
        self.p_threat = 1.0 - (
            (1.0 - self.p_dist)
            * (1.0 - self.p_dens)
            * (1.0 - self.p_unk)
        )
        self.p_threat = np.clip(self.p_threat, 0.0, 1.0)

    def query_risk(self, point: np.ndarray) -> float:
        """
        Query implicit risk at a local 3D point.
        This does not require building a full 3D voxel map.
        """
        point = np.asarray(point, dtype=float)
        r = np.linalg.norm(point)

        if r < 1e-6:
            return 0.0

        theta = np.arctan2(point[1], point[0])
        phi = np.arctan2(point[2], np.sqrt(point[0] ** 2 + point[1] ** 2))
        a, b = self._sector_index(theta, phi)

        has_obs = self.count[a, b] > 0
        d_obs = self.d_min[a, b]

        unknown_risk = self.p_unk[a, b]
        out_range_risk = self.rho_out if r > self.r_max else 0.0

        if has_obs:
            radial_obs_risk = self.p_threat[a, b] * np.exp(
                -0.5 * ((r - d_obs) / self.sigma_r) ** 2
            )
            occlusion_risk = (
                self.rho_occl * self.p_dist[a, b]
                if r > d_obs + self.r_safe
                else 0.0
            )
        else:
            radial_obs_risk = 0.0
            occlusion_risk = 0.0

        risk = 1.0 - (
            (1.0 - radial_obs_risk)
            * (1.0 - occlusion_risk)
            * (1.0 - unknown_risk)
            * (1.0 - out_range_risk)
        )

        return float(np.clip(risk, 0.0, 1.0))

    def query_line_risk(self, p0: np.ndarray, p1: np.ndarray, n_samples: int = 12) -> float:
        """Average risk along the line segment p0 -> p1."""
        risks = []
        for i in range(n_samples):
            alpha = (i + 1) / n_samples
            q = (1 - alpha) * p0 + alpha * p1
            risks.append(self.query_risk(q))

        return float(np.mean(risks))


# =========================
# 4. Guiding waypoint generation
# =========================

def generate_candidate_dirs(field: SectorThreatField, use_visible_only: bool = False) -> np.ndarray:
    """Generate candidate directions from sector-center directions."""
    dirs = field.sector_dirs.reshape(-1, 3)
    visible = field.visible_mask.reshape(-1)

    if use_visible_only:
        dirs = dirs[visible]

    return dirs


def generate_waypoints(
    field: SectorThreatField,
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    k_waypoints: int = 8,
    step_len: float = 0.65,
    candidate_dirs: np.ndarray | None = None,
    lambda_point_risk: float = 2.5,
    lambda_line_risk: float = 4.5,
    lambda_goal: float = 0.9,
    lambda_smooth: float = 0.25,
    lambda_height: float = 0.15,
    lambda_progress: float = 0.35,
) -> np.ndarray:
    """Recursively generate multiple guiding waypoints from implicit sector risk."""
    if candidate_dirs is None:
        candidate_dirs = generate_candidate_dirs(field, use_visible_only=False)

    waypoints = []
    current = agent_pos.copy()
    prev_dir = normalize(goal_pos - agent_pos)

    for _ in range(k_waypoints):
        goal_dir = normalize(goal_pos - current)

        best_cost = np.inf
        best_point = None
        best_dir = None

        for d in candidate_dirs:
            d = normalize(d)
            candidate = current + step_len * d

            point_risk = field.query_risk(candidate)
            line_risk = field.query_line_risk(current, candidate)

            goal_cost = 1.0 - np.dot(d, goal_dir)
            smooth_cost = 1.0 - np.dot(d, prev_dir)
            height_cost = abs(candidate[2] - current[2])

            progress = np.linalg.norm(current - goal_pos) - np.linalg.norm(candidate - goal_pos)

            cost = (
                lambda_point_risk * point_risk
                + lambda_line_risk * line_risk
                + lambda_goal * goal_cost
                + lambda_smooth * smooth_cost
                + lambda_height * height_cost
                - lambda_progress * progress
            )

            if cost < best_cost:
                best_cost = cost
                best_point = candidate
                best_dir = d

        waypoints.append(best_point)
        current = best_point
        prev_dir = best_dir

    return np.array(waypoints)


# =========================
# 5. Visualization
# =========================

def plot_cylinder(
    ax,
    center: np.ndarray,
    radius: float,
    height: float,
    color: str = "gray",
    alpha: float = 0.35,
) -> None:
    """Plot a cylinder surface."""
    cx, cy, cz = center

    theta = np.linspace(0, 2 * np.pi, 80)
    z = np.linspace(cz - height / 2, cz + height / 2, 40)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x = cx + radius * np.cos(theta_grid)
    y = cy + radius * np.sin(theta_grid)

    ax.plot_surface(x, y, z_grid, color=color, alpha=alpha, linewidth=0)


def visualize(
    field: SectorThreatField,
    cylinder_center: np.ndarray,
    cylinder_radius: float,
    cylinder_height: float,
    lidar_points: np.ndarray,
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    waypoints: np.ndarray,
) -> None:
    """Visualize 3D risk field, XY risk slice, and sector threat heatmap."""
    fig = plt.figure(figsize=(18, 5))

    # -------- 3D risk cloud --------
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.set_title("3D implicit risk field + generated waypoints")

    xs = np.linspace(0.0, 6.5, 36)
    ys = np.linspace(-3.0, 3.0, 28)
    zs = np.linspace(-1.6, 2.4, 18)

    grid_points = []
    risks = []

    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z])
                risk = field.query_risk(p)
                if risk > 0.25:
                    grid_points.append(p)
                    risks.append(risk)

    grid_points = np.array(grid_points)
    risks = np.array(risks)

    if len(grid_points) > 0:
        sc = ax1.scatter(
            grid_points[:, 0],
            grid_points[:, 1],
            grid_points[:, 2],
            c=risks,
            s=8,
            alpha=0.45,
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(sc, ax=ax1, shrink=0.65, label="risk probability")

    plot_cylinder(ax1, cylinder_center, cylinder_radius, cylinder_height)

    ax1.scatter(agent_pos[0], agent_pos[1], agent_pos[2], s=80, marker="o", label="agent")
    ax1.scatter(goal_pos[0], goal_pos[1], goal_pos[2], s=100, marker="*", label="goal")
    ax1.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        waypoints[:, 2],
        marker="o",
        linewidth=2.5,
        label="generated reference points",
    )
    ax1.scatter(
        lidar_points[:, 0],
        lidar_points[:, 1],
        lidar_points[:, 2],
        s=1,
        alpha=0.15,
        label="LiDAR obstacle points",
    )

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_xlim(-0.5, 7.0)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_zlim(-2.0, 2.8)
    ax1.legend(loc="upper left")

    # -------- XY slice risk map --------
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Risk slice at z = 0")

    xg = np.linspace(-0.2, 7.0, 180)
    yg = np.linspace(-3.0, 3.0, 150)
    x_grid, y_grid = np.meshgrid(xg, yg)
    z_grid = np.zeros_like(x_grid)
    risk_slice = np.zeros_like(x_grid)

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            p = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
            risk_slice[i, j] = field.query_risk(p)

    im = ax2.contourf(x_grid, y_grid, risk_slice, levels=30, cmap="inferno", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax2, label="risk probability")

    circle = plt.Circle(
        (cylinder_center[0], cylinder_center[1]),
        cylinder_radius,
        fill=False,
        linewidth=2,
    )
    ax2.add_patch(circle)

    ax2.scatter(agent_pos[0], agent_pos[1], s=80, marker="o", label="agent")
    ax2.scatter(goal_pos[0], goal_pos[1], s=100, marker="*", label="goal")
    ax2.plot(waypoints[:, 0], waypoints[:, 1], marker="o", linewidth=2.5, label="waypoints")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()

    # -------- Sector threat heatmap --------
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Sector threat probability")

    threat = field.p_threat.T
    extent = [
        np.rad2deg(field.theta_min),
        np.rad2deg(field.theta_max),
        np.rad2deg(field.phi_min),
        np.rad2deg(field.phi_max),
    ]

    im2 = ax3.imshow(
        threat,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(im2, ax=ax3, label="sector threat probability")

    ax3.axhline(np.rad2deg(field.lidar_phi_min), linestyle="--", linewidth=1)
    ax3.axhline(np.rad2deg(field.lidar_phi_max), linestyle="--", linewidth=1)
    ax3.set_xlabel("azimuth θ [deg]")
    ax3.set_ylabel("elevation φ [deg]")

    plt.tight_layout()
    plt.show()


# =========================
# 6. Main demo
# =========================

def main() -> None:
    np.random.seed(7)

    # Agent and target in local frame
    agent_pos = np.array([0.0, 0.0, 0.0])
    goal_pos = np.array([7.0, 0.0, 0.0])

    # One cylinder obstacle near the straight path
    cylinder_center = np.array([3.2, 0.0, 0.0])
    cylinder_radius = 0.75
    cylinder_height = 2.5

    # Pseudo LiDAR points from cylinder surface
    lidar_points = generate_cylinder_points(
        center=cylinder_center,
        radius=cylinder_radius,
        height=cylinder_height,
        n_theta=180,
        n_z=90,
    )

    # Add small noise to simulate LiDAR measurement noise
    lidar_points = lidar_points + np.random.normal(scale=0.015, size=lidar_points.shape)

    # Build sector threat field
    field = SectorThreatField(
        n_theta=36,
        n_phi=13,
        r_max=8.0,
        lidar_phi_min=np.deg2rad(-45),
        lidar_phi_max=np.deg2rad(45),
        r_safe=0.45,
        sigma_d=0.9,
        sigma_r=0.5,
        density_alpha=0.06,
        rho_out=0.45,
        rho_occl=0.6,
    )
    field.build_from_lidar_points(lidar_points)

    candidate_dirs = generate_candidate_dirs(field, use_visible_only=False)

    # Generate continuous guiding reference points
    waypoints = generate_waypoints(
        field=field,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        k_waypoints=9,
        step_len=0.65,
        candidate_dirs=candidate_dirs,
    )

    print("Generated reference points:")
    for i, waypoint in enumerate(waypoints, start=1):
        risk = field.query_risk(waypoint)
        print(
            f"w{i}: [{waypoint[0]: .3f}, {waypoint[1]: .3f}, {waypoint[2]: .3f}], "
            f"risk={risk:.3f}"
        )

    visualize(
        field=field,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
        lidar_points=lidar_points,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        waypoints=waypoints,
    )


if __name__ == "__main__":
    main()
