# -*- coding: utf-8 -*-
# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
# 上流(unitree_rl_lab)の play スクリプトの構成を参考に自作した、プレゼン用クリーン再生+録画。上流のコードは含まない。モーションを必ず先頭(t=0)から開始し、
# リセット時の姿勢/速度/関節ノイズをゼロ化 → 「静止→動作→静止」の映像になる。
import argparse
from importlib.metadata import version
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play RSL-RL agent cleanly from motion t=0.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import types
import torch
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric, entry_point_key="play_env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # ===== リセットノイズをゼロ化(プレゼン用の決め打ち初期状態) =====
    try:
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        print("[play_clean] reset noise zeroed")
    except Exception as e:
        print("[play_clean] noise zero failed:", e)

    # ===== 近接カメラ(ロボット追従) =====
    try:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (3.4, 3.4, 1.15)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.05)
        env_cfg.viewer.resolution = (1280, 720)
        print("[play_clean] tracking camera set")
    except Exception as e:
        print("[play_clean] viewer override failed:", e)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ===== RSI無効化: モーションを常に先頭から開始 =====
    try:
        term = env.unwrapped.command_manager.get_term("motion")

        def _start_zero(self, env_ids):
            self.time_steps[env_ids] = 0

        term._adaptive_sampling = types.MethodType(_start_zero, term)
        print("[play_clean] motion start forced to t=0")
    except Exception as e:
        print("[play_clean] t=0 patch failed:", e)

    if args_cli.video:
        video_kwargs = {"video_folder": os.path.join(log_dir, "videos", "play_clean"),
                        "step_trigger": lambda step: step == 0,
                        "video_length": args_cli.video_length, "disable_logger": True}
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    # 生きた静止: 冒頭HOLD_START歩はモーション時計を0に固定(物理で姿勢保持)、
    # 1周(motion末尾)到達後はHOLD_ENDまで最終フレームに固定(ループによる座り直しを防ぐ)
    HOLD_START = 50
    term2 = None
    last_idx = 0
    try:
        term2 = env.unwrapped.command_manager.get_term("motion")
        last_idx = int(term2.motion.time_step_total - 1)
    except Exception as e:
        print("[play_clean] hold patch unavailable:", e)

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if term2 is not None:
                if timestep < HOLD_START:
                    term2.time_steps[:] = 0
                elif term2.time_steps[0] >= last_idx:
                    term2.time_steps[:] = last_idx
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
