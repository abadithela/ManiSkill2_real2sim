"""
# Apurva Badithela
# 12/13/24
# A script to command the WidowX robot to go to particular waypoints to check if the robot is setup correctly

Simple script for real-to-sim eval using the prepackaged visual matching setup in ManiSkill2.
Example:
    cd {path_to_simpler_env_repo_root}
    python simpler_env/command_robot_traj.py --policy octo  --logging-root ./results_compare_traj/  
"""

import argparse
import os

import mediapy as media
import numpy as np
import tensorflow as tf
from sapien.core import Pose
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
import gymnasium as gym
from mani_skill2_real2sim.utils.sapien_utils import look_at, normalize_vector
import sys
sys.path.append("..")
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from pdb import set_trace as st

# =======================================================================
# Parser and logging dir setup
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose") # end-effector pose
    parser.add_argument("--logging-root", type=str, default="./results_compare_traj")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--add-segmentation", action="store_true")
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))

    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs
    return args

args = parse_args()
os.makedirs(args.logging_root, exist_ok=True)

# =======================================================
# Main function to control the robot:
def main(init_qpos, recorded_traj):
    exp_length = len(recorded_traj)
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    if "robot" in args.env_kwargs:
        if "widowx" in args.env_kwargs["robot"]:
            pose = look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.7])
            args.env_kwargs["render_camera_cfgs"] = {
                "render_camera": dict(p=pose.p, q=pose.q)
            }

    
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        camera_cfgs={"add_segmentation": args.add_segmentation},
        **args.env_kwargs
    )

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    env_reset_options = {}
    if (not hasattr(env, "prepackaged_config")) or (not env.prepackaged_config):
        """
        Change the following reset options as you want to debug the environment
        """
        names_in_env_id_fxn = lambda name_list: any(
            name in args.env_id for name in name_list
        )
        
        if names_in_env_id_fxn(["PutCarrotOnPlate"]):
            init_rot_quat = Pose(q=[0, 0, 0, 1]).q
            if env.robot_uid == "irom_widowx":
                # Check how I can pass in init_qpos
                env_reset_options = {
                    "obj_init_options": {},
                    "robot_init_options": {
                        "init_xy": [0.27,0.22],
                        'init_height': env.scene_table_height + 0.04,
                        "init_rot_quat": init_rot_quat,
                    },
                }
            env_reset_options["obj_init_options"]["episode_id"] = 0
    
    obs, info = env.reset(options=env_reset_options)
    image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
    images = [image]
    print("Reset info:", info)
    after_reset = False

    if "wx250s" in env.agent.robot.name:
        print(
            "3rd view camera pose",
            env.unwrapped._cameras["3rd_view_camera"].camera.pose,
        )
        print(
            "3rd view camera pose wrt robot base",
            env.agent.robot.pose.inv()
            * env.unwrapped._cameras["3rd_view_camera"].camera.pose,
        )
    print("robot pose", env.agent.robot.pose)

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    is_widowx = "wx250s" in env.agent.robot.name
    is_gripper_delta_target_control = (
        env.agent.controller.controllers["gripper"].config.use_target
        and env.agent.controller.controllers["gripper"].config.use_delta
    )

    def get_reset_gripper_action():
        # open gripper at initialization
        return 1

    gripper_action = get_reset_gripper_action()

    EE_ACTION = 0.02
    EE_ROT_ACTION = 0.1

    # print("obj pose", env.obj.pose, "tcp pose", env.tcp.pose)
    print("qpos", env.agent.robot.get_qpos())

    #### Go over trajectory:
    k = 0
    while k < exp_length:
        if has_base:
            base_action = np.zeros([4])  # hardcoded # Base does not move
        ee_action=recorded_traj[k]
        # -------------------------------------------------------------------------- #
        # Post-process action
        # -------------------------------------------------------------------------- #
        action_dict = dict(base=base_action, arm=ee_action)
        if has_gripper:
            action_dict["gripper"] = gripper_action
        action = env.agent.controller.from_action_dict(action_dict)

        print("action", action)
        obs, reward, terminated, truncated, info = env.step(action)

        if is_gripper_delta_target_control:
            gripper_action = 0

        # print("obj pose", env.obj.pose, "tcp pose", env.tcp.pose)
        print("tcp pose wrt robot base", env.agent.robot.pose.inv() * env.tcp.pose)
        print("qpos", env.agent.robot.get_qpos())
        print("reward", reward)
        print("terminated", terminated, "truncated", truncated)
        print("info", info)

        k += 1