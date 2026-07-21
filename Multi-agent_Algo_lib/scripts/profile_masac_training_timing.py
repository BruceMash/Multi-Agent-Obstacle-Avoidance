from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
for path in (PROJECT_ROOT, ALGO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from MASAC.MASAC import MASAC
from MASAC.config import MASAC_EXPERIMENT_CONFIG
from MASAC.curriculum import build_curriculum_stages
from train_masac_multi_agent_dmp import (
    agent_dict_to_matrix,
    build_critic_action_matrix,
    build_dim_info,
    build_env,
    build_network_config,
    matrix_to_agent_dict,
    resolve_device,
    vector_to_agent_dict,
)


class TimingStats:
    def __init__(self, device: torch.device):
        self.device = device
        self.total = defaultdict(float)
        self.calls = defaultdict(int)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def add(self, name: str, seconds: float) -> None:
        self.total[name] += float(seconds)
        self.calls[name] += 1

    def time_block(self, name: str):
        return _TimedBlock(self, name)

    def print_summary(
        self,
        *,
        wall_time: float,
        batch_size: int,
        num_agents: int,
        temporal_steps: int,
    ) -> None:
        print("\n=== Timing summary ===")
        print(f"wall_time: {wall_time:.3f}s")
        print(f"batch_size: {batch_size}")
        print(f"num_agents: {num_agents}")
        print(f"temporal_steps: {temporal_steps}")

        learn_calls = self.calls.get("learn_total", 0)
        if learn_calls:
            mean_learn = self.total["learn_total"] / learn_calls
            batch_per_sec = batch_size / mean_learn if mean_learn > 0 else 0.0
            frames_per_sec = (
                batch_size * num_agents * temporal_steps / mean_learn
                if mean_learn > 0
                else 0.0
            )
            print(f"learn_calls: {learn_calls}")
            print(f"mean_learn: {mean_learn * 1000.0:.3f} ms")
            print(f"batch_throughput: {batch_per_sec:.2f} samples/s")
            print(f"agent_time_frame_throughput: {frames_per_sec:.2f} frames/s")
        else:
            print("learn_calls: 0")

        print("\nstage                              calls     total(s)    mean(ms)   wall(%)")
        print("-" * 76)
        for name, total in sorted(
            self.total.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            calls = self.calls[name]
            mean_ms = total / max(calls, 1) * 1000.0
            wall_percent = total / wall_time * 100.0 if wall_time > 0 else 0.0
            print(
                f"{name:<34} {calls:>6d} {total:>11.3f} "
                f"{mean_ms:>11.3f} {wall_percent:>8.2f}"
            )


class _TimedBlock:
    def __init__(self, stats: TimingStats, name: str):
        self.stats = stats
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.stats.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stats.synchronize()
        self.stats.add(self.name, time.perf_counter() - self.start)
        return False


class ProfiledMASAC(MASAC):
    def __init__(self, *args, timing: TimingStats, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing = timing

    def sample(self, batch_size):
        with self.timing.time_block("learn/sample_total"):
            total_size = len(self.buffers[self.agent_x])
            with self.timing.time_block("learn/sample_indices"):
                indices = np.random.choice(total_size, batch_size, replace=False)

            obs, action, reward, next_obs, done = {}, {}, {}, {}, {}
            obs_mask, next_obs_mask = {}, {}
            next_action = {}
            next_log_pi = {}
            for agent_id, buffer in self.buffers.items():
                with self.timing.time_block("learn/replay_sample_one_agent"):
                    sampled = buffer.sample(
                        indices,
                        sequence_length=self.temporal_steps,
                    )
                if self.temporal_steps > 1:
                    (
                        obs[agent_id],
                        action[agent_id],
                        reward[agent_id],
                        next_obs[agent_id],
                        done[agent_id],
                        obs_mask[agent_id],
                        next_obs_mask[agent_id],
                    ) = sampled
                else:
                    (
                        obs[agent_id],
                        action[agent_id],
                        reward[agent_id],
                        next_obs[agent_id],
                        done[agent_id],
                    ) = sampled
                    obs_mask[agent_id] = None
                    next_obs_mask[agent_id] = None

                with torch.no_grad():
                    with self.timing.time_block("learn/target_actor_one_agent"):
                        next_actor_action, next_log_pi[agent_id] = self.agents[
                            agent_id
                        ].actor_target(
                            next_obs[agent_id],
                            temporal_mask=next_obs_mask[agent_id],
                        )
                        next_action[agent_id] = self.actor_action_to_critic_action(
                            next_obs[agent_id],
                            next_actor_action,
                            temporal_mask=next_obs_mask[agent_id],
                        )

            return (
                obs,
                action,
                reward,
                next_obs,
                done,
                obs_mask,
                next_obs_mask,
                next_action,
                next_log_pi,
            )

    def learn(self, batch_size, gamma, tau):
        with self.timing.time_block("learn_total"):
            (
                obs,
                action,
                reward,
                next_obs,
                done,
                obs_mask,
                next_obs_mask,
                next_action,
                next_log_pi,
            ) = self.sample(batch_size)

            for agent_id, agent in self.agents.items():
                with torch.no_grad():
                    with self.timing.time_block("learn/target_q_one_agent"):
                        q1_next_target, q2_next_target = agent.critic_target(
                            next_obs,
                            next_action,
                            temporal_masks=next_obs_mask,
                        )
                        q_next_target = torch.min(q1_next_target, q2_next_target)

                        if self.entropy_way_c == "0":
                            stacked_next_log_pi = torch.stack(
                                [
                                    next_log_pi[other_id]
                                    for other_id in self.agents.keys()
                                ],
                                dim=1,
                            ).sum(dim=1)
                            entropy_next = -stacked_next_log_pi
                        elif self.entropy_way_c == "1":
                            entropy_next = -next_log_pi[agent_id]
                        else:
                            raise ValueError(
                                f"unsupported entropy_way_c: {self.entropy_way_c}"
                            )

                        q_target = reward[agent_id] + gamma * (
                            1 - done[agent_id]
                        ) * (
                            q_next_target
                            + self.alphas[agent_id].alpha.detach() * entropy_next
                        )

                q_target = q_target.detach()
                with self.timing.time_block("learn/critic_zero_grad"):
                    agent.critic_optimizer.zero_grad()

                with self.timing.time_block("learn/critic_q1_backward_one_agent"):
                    q1 = agent.critic.forward_q1(
                        obs,
                        action,
                        temporal_masks=obs_mask,
                    )
                    q1_loss = F.mse_loss(q1, q_target)
                    q1_loss.backward()
                    del q1, q1_loss

                with self.timing.time_block("learn/critic_q2_backward_one_agent"):
                    q2 = agent.critic.forward_q2(
                        obs,
                        action,
                        temporal_masks=obs_mask,
                    )
                    q2_loss = F.mse_loss(q2, q_target)
                    q2_loss.backward()
                    del q2, q2_loss

                with self.timing.time_block("learn/critic_step_one_agent"):
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
                    agent.critic_optimizer.step()

                new_action = {}
                new_log_pi = {}
                with self.timing.time_block("learn/actor_sample_actions_one_agent"):
                    for other_id, other_agent in self.agents.items():
                        if other_id == agent_id:
                            sampled_action, sampled_log_pi = other_agent.actor(
                                obs[other_id],
                                temporal_mask=obs_mask[other_id],
                            )
                        else:
                            with torch.no_grad():
                                sampled_action, sampled_log_pi = other_agent.actor(
                                    obs[other_id],
                                    temporal_mask=obs_mask[other_id],
                                )
                        new_action[other_id] = self.actor_action_to_critic_action(
                            obs[other_id],
                            sampled_action,
                            temporal_mask=obs_mask[other_id],
                        )
                        new_log_pi[other_id] = sampled_log_pi

                critic_requires_grad = [
                    critic_param.requires_grad
                    for critic_param in agent.critic.parameters()
                ]
                for critic_param in agent.critic.parameters():
                    critic_param.requires_grad_(False)
                try:
                    with self.timing.time_block("learn/actor_critic_forward_one_agent"):
                        if self.action_way == "0":
                            mixed_action = dict(action)
                            mixed_action[agent_id] = new_action[agent_id]
                            q1_pi, q2_pi = agent.critic(
                                obs,
                                mixed_action,
                                temporal_masks=obs_mask,
                            )
                        elif self.action_way == "1":
                            q1_pi, q2_pi = agent.critic(
                                obs,
                                new_action,
                                temporal_masks=obs_mask,
                            )
                        else:
                            raise ValueError(
                                f"unsupported action_way: {self.action_way}"
                            )

                        if self.entropy_way_a == "0":
                            stacked_log_pi = torch.stack(
                                [
                                    new_log_pi[other_id]
                                    for other_id in self.agents.keys()
                                ],
                                dim=1,
                            ).sum(dim=1)
                            entropy = -stacked_log_pi
                        elif self.entropy_way_a == "1":
                            entropy = -new_log_pi[agent_id]
                        else:
                            raise ValueError(
                                f"unsupported entropy_way_a: {self.entropy_way_a}"
                            )

                        q_pi = torch.min(q1_pi, q2_pi)
                        actor_loss = (
                            -q_pi
                            - self.alphas[agent_id].alpha.detach() * entropy
                        ).mean()

                    with self.timing.time_block("learn/actor_backward_one_agent"):
                        agent.update_actor(actor_loss)
                finally:
                    for critic_param, requires_grad in zip(
                        agent.critic.parameters(),
                        critic_requires_grad,
                    ):
                        critic_param.requires_grad_(requires_grad)

                if self.adaptive_alpha:
                    with self.timing.time_block("learn/alpha_update_one_agent"):
                        alpha_loss = (
                            self.alphas[agent_id].alpha
                            * (
                                entropy
                                - self.alphas[agent_id].target_entropy
                            ).detach()
                        ).mean()
                        self.alphas[agent_id].update_alpha(alpha_loss)

            with self.timing.time_block("learn/target_update"):
                self.update_target(tau)


def parse_args() -> argparse.Namespace:
    config = MASAC_EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser(
        description="Profile MASAC training stage timing without saving models."
    )
    parser.add_argument("--seed", type=int, default=int(config.seed))
    parser.add_argument("--total-steps", type=int, default=3000)
    parser.add_argument("--start-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=int(config.batch_size))
    parser.add_argument("--buffer-size", type=int, default=int(config.buffer_size))
    parser.add_argument("--actor-lr", type=float, default=float(config.actor_lr))
    parser.add_argument("--critic-lr", type=float, default=float(config.critic_lr))
    parser.add_argument("--gamma", type=float, default=float(config.gamma))
    parser.add_argument("--tau", type=float, default=float(config.soft_update_tau))
    parser.add_argument("--learn-interval", type=int, default=int(config.learn_interval))
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=int(config.updates_per_step),
    )
    parser.add_argument("--temporal-steps", type=int, default=int(config.temporal_steps))
    parser.add_argument("--hidden-dim", type=int, default=int(config.hidden_dim))
    parser.add_argument(
        "--sensor-hidden-dim",
        type=int,
        default=int(config.sensor_hidden_dim),
    )
    parser.add_argument(
        "--ally-hidden-dim",
        type=int,
        default=int(config.ally_hidden_dim),
    )
    parser.add_argument(
        "--sensor-output-dim",
        type=int,
        default=int(config.sensor_output_dim),
    )
    parser.add_argument(
        "--ally-output-dim",
        type=int,
        default=int(config.ally_output_dim),
    )
    parser.add_argument(
        "--num-sensor-layers",
        type=int,
        default=int(config.num_sensor_layers),
    )
    parser.add_argument(
        "--num-ally-layers",
        type=int,
        default=int(config.num_ally_layers),
    )
    parser.add_argument(
        "--num-observation-layers",
        type=int,
        default=int(config.num_observation_layers),
    )
    parser.add_argument(
        "--actor-log-std-min",
        type=float,
        default=float(config.actor_log_std_min),
    )
    parser.add_argument(
        "--actor-log-std-max",
        type=float,
        default=float(config.actor_log_std_max),
    )
    parser.add_argument(
        "--critic-encoder",
        type=str,
        choices=("attention", "mlp"),
        default=str(config.critic_encoder),
    )
    parser.add_argument("--device", type=str, default=str(config.device))
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument(
        "--final-stage-only",
        action="store_true",
        help="Profile the final curriculum stage instead of the empty scene.",
    )
    return parser.parse_args()


def run_profile() -> None:
    args = parse_args()
    args.report_interval = max(1, int(args.report_interval))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    experiment_config = MASAC_EXPERIMENT_CONFIG
    profile_stage = None
    if args.final_stage_only:
        stages = build_curriculum_stages(
            experiment_config.curriculum_phase2_box_counts,
            experiment_config.curriculum_phase2_sphere_counts,
            experiment_config.curriculum_phase3_dynamic_counts,
        )
        profile_stage = stages[-1]
    env = build_env(experiment_config, profile_stage)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    agent_ids = [f"agent_{index}" for index in range(int(env.num_agents))]
    dim_info = build_dim_info(env, agent_ids)
    network_config = build_network_config(env, experiment_config, args)
    device = resolve_device(args.device)
    timing = TimingStats(device)

    policy = ProfiledMASAC(
        dim_info=dim_info,
        is_continue=True,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        buffer_size=int(args.buffer_size),
        device=device,
        network_config=network_config,
        timing=timing,
    )

    print("MASAC timing profiler")
    print(f"device: {device}")
    print(f"total_steps: {int(args.total_steps)}")
    print(f"start_steps: {int(args.start_steps)}")
    print(f"batch_size: {int(args.batch_size)}")
    print(f"learn_interval: {int(args.learn_interval)}")
    print(f"updates_per_step: {int(args.updates_per_step)}")
    print(f"temporal_steps: {int(args.temporal_steps)}")
    print(f"critic_encoder: {args.critic_encoder}")
    print(f"scenario_stage: {profile_stage.name if profile_stage else 'empty'}")

    obs_matrix, _ = env.reset(seed=args.seed)
    obs = matrix_to_agent_dict(obs_matrix, agent_ids)
    min_learn_size = max(int(args.batch_size), int(args.temporal_steps))
    episode_count = 0
    episode_step = 0

    wall_start = time.perf_counter()
    for global_step in range(1, int(args.total_steps) + 1):
        if global_step <= int(args.start_steps):
            with timing.time_block("loop/random_action"):
                action_matrix = env.action_space.sample().astype(np.float32)
                action = matrix_to_agent_dict(action_matrix, agent_ids)
        else:
            with timing.time_block("loop/select_action"):
                action = policy.select_action(obs)
            with timing.time_block("loop/action_pack"):
                action_matrix = agent_dict_to_matrix(action, agent_ids)
                action_matrix = np.clip(
                    action_matrix,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)
                action = matrix_to_agent_dict(action_matrix, agent_ids)

        with timing.time_block("loop/env_step"):
            next_obs_matrix, rewards, terminated, truncated, info = env.step(
                action_matrix
            )

        with timing.time_block("loop/post_step_pack"):
            next_obs = matrix_to_agent_dict(next_obs_matrix, agent_ids)
            reward = vector_to_agent_dict(rewards, agent_ids)
            critic_action_matrix = build_critic_action_matrix(info, env)
            critic_action = matrix_to_agent_dict(critic_action_matrix, agent_ids)
            done_for_buffer = {
                agent_id: bool(terminated)
                for agent_id in agent_ids
            }

        with timing.time_block("loop/replay_add"):
            policy.add(obs, critic_action, reward, next_obs, done_for_buffer)

        episode_step += 1
        obs = next_obs

        can_learn = len(policy.buffers[policy.agent_x]) >= min_learn_size
        if (
            can_learn
            and global_step > int(args.start_steps)
            and global_step % int(args.learn_interval) == 0
        ):
            for _ in range(int(args.updates_per_step)):
                policy.learn(
                    batch_size=int(args.batch_size),
                    gamma=float(args.gamma),
                    tau=float(args.tau),
                )

        if bool(terminated or truncated):
            episode_count += 1
            episode_step = 0
            with timing.time_block("loop/env_reset"):
                obs_matrix, _ = env.reset()
                obs = matrix_to_agent_dict(obs_matrix, agent_ids)

        if global_step % int(args.report_interval) == 0:
            elapsed = time.perf_counter() - wall_start
            learn_calls = timing.calls.get("learn_total", 0)
            mean_learn = (
                timing.total["learn_total"] / learn_calls
                if learn_calls
                else 0.0
            )
            print(
                f"step={global_step} "
                f"buffer={len(policy.buffers[policy.agent_x])} "
                f"episodes={episode_count} "
                f"learn_calls={learn_calls} "
                f"mean_learn_ms={mean_learn * 1000.0:.2f} "
                f"elapsed={elapsed:.1f}s"
            )

    wall_time = time.perf_counter() - wall_start
    timing.print_summary(
        wall_time=wall_time,
        batch_size=int(args.batch_size),
        num_agents=len(agent_ids),
        temporal_steps=int(args.temporal_steps),
    )


def main() -> None:
    run_profile()


if __name__ == "__main__":
    main()
