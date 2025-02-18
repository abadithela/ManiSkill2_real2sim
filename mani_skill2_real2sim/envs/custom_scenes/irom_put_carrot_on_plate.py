# New environment for IROM Lab setup
# Experiment: Put Carrot on Plate
# Reference: https://github.com/simpler-env/SimplerEnv/blob/main/ADDING_NEW_ENVS_ROBOTS.md

from collections import OrderedDict
from typing import List
import argparse

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim import ASSET_DIR

from .base_env import CustomBridgeObjectsInSceneEnv
from .move_near_in_scene import MoveNearInSceneEnv
import math
from pdb import set_trace as st 

class PutOnInSceneEnvIROM(MoveNearInSceneEnv):
    def reset(self, *args, **kwargs):
        self.consecutive_grasp = 0
        return super().reset(*args, **kwargs)

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            moved_correct_obj=False,
            moved_wrong_obj=False,
            is_src_obj_grasped=False,
            consecutive_grasp=False,
            src_on_target=False,
        )

    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose one randomly."""

        if model_ids is None:
            src_model_id = random_choice(self.model_ids, self._episode_rng)
            tgt_model_id = (self.model_ids.index(src_model_id) + 1) % len(
                self.model_ids
            )
            model_ids = [src_model_id, tgt_model_id]

        return super()._set_model(model_ids, model_scales)

    def evaluate(self, success_require_src_completely_on_target=True, z_flag_required_offset=0.02, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # whether moved the correct object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2] - self.source_obj_pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
            self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
        if is_src_obj_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        
        offset = pos_src - pos_tgt
        xy_flag = (
            np.linalg.norm(offset[:2])
            <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[2] > 0) and (
            offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag and z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contacts = self._scene.get_contacts()
            flag = True
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            tgt_obj_name = self.episode_target_obj.name
            ignore_actor_names = [tgt_obj_name] + robot_link_names
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                other_obj_contact_actor_name = None
                if actor_0.name == self.episode_source_obj.name:
                    other_obj_contact_actor_name = actor_1.name
                elif actor_1.name == self.episode_source_obj.name:
                    other_obj_contact_actor_name = actor_0.name
                if other_obj_contact_actor_name is not None:
                    # the object is in contact with an actor
                    contact_impulse = np.sum(
                        [point.impulse for point in contact.points], axis=0
                    )
                    if (other_obj_contact_actor_name not in ignore_actor_names) and (
                        np.linalg.norm(contact_impulse) > 1e-6
                    ):
                        # the object has contact with an actor other than the robot link or the target object, so the object is not yet put on the target object
                        flag = False
                        break
            src_on_target = src_on_target and flag

        success = src_on_target

        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] or consecutive_grasp
        )

        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            src_on_target=src_on_target,
            episode_stats=self.episode_stats,
            success=success,
        )

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"put {src_name} on {tgt_name}"


class PutOnBridgeInSceneEnvIROM(PutOnInSceneEnvIROM, CustomBridgeObjectsInSceneEnv):
    def __init__(
        self,
        source_obj_name: str = None,
        target_obj_name: str = None,
        xy_configs: List[np.ndarray] = None,
        quat_configs: List[np.ndarray] = None,
        warm = False,
        **kwargs,
    ):
        self._source_obj_name = source_obj_name
        self._target_obj_name = target_obj_name
        self._xy_configs = xy_configs
        self._quat_configs = quat_configs
        self.warm=warm
        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "irom_widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "irom_bench"
        ret["camera_cfgs"] = {"add_segmentation": True}
        # This is the RGB overlay path that needs to change
        if not self.warm:
            ret["rgb_overlay_path"] = str(
                ASSET_DIR / "real_inpainting/irom_lab_camera_imgs/20250125-162809/init_img.jpg"
            )
        else:
            ret["rgb_overlay_path"] = str(
                ASSET_DIR / "real_inpainting/irom_lab_camera_imgs/20250125-162809/warm.jpg"
            )
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]

        return ret

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        
        self.set_episode_rng(seed)
        # Model scales:
        # options["model_scales"] = [0.75, 1.154] # Carrot and plate
        options["model_scales"] = [1.25, 1.154] # Carrot and plate

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        
        if "init_xys" not in obj_init_options.keys():
            # Episodes are defined by the distribution of xy configs of objects and robot poses and their rotations
            # Runs a random episode from the total number of possible episodes
            episode_id = obj_init_options.get(
                "episode_id",
                self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
            )
            # xy_config for objects and robot for a particular episode
            xy_config = self._xy_configs[
                (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
                // len(self._quat_configs)
            ]
            
            # quat_config for objects and robot for a particular episode
            quat_config = self._quat_configs[episode_id % len(self._quat_configs)]

            options["model_ids"] = [self._source_obj_name, self._target_obj_name]
            obj_init_options["source_obj_id"] = 0
            obj_init_options["target_obj_id"] = 1
            obj_init_options["init_xys"] = xy_config
            obj_init_options["init_rot_quats"] = quat_config
            options["obj_init_options"] = obj_init_options
        else:
            episode_id = 0
            options["model_ids"] = [self._source_obj_name, self._target_obj_name]
            obj_init_options["source_obj_id"] = 0
            obj_init_options["target_obj_id"] = 1
            obj_init_options["init_rot_quats"] = self._quat_configs[0]
            options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        # Real bench is 29.5 cm wide; the sim asset is 29.3 cm.
        # For the sim asset, lower-left table corner (a,b) with respect to table center (tx,ty) is:
        # a - tx = 0.146803 and b-ty = -0.379149 m.
        # Due to the scene_offset parameter, the table center (tx, ty) is the origin for the simulation, and the coordinate 
        # with respect to which all objects are measured.
        
        qpos = None
        if "robot_init_options" in options.keys():
            if "qpos" in options["robot_init_options"].keys():
                qpos = options["robot_init_options"]["qpos"]

        self.robot_init_xy = [0.185,0.215]
        self.robot_init_height = self.scene_table_height + 0.04
        self.robot_init_quat = [0,0,0,1]
        options["robot_init_options"] = {
            "init_xy": self.robot_init_xy, # [0.185,0.22]
            # "init_xy": init_xy,
            'init_height': self.robot_init_height,
            "init_rot_quat": self.robot_init_quat,
            "qpos": qpos,
        }
        return False

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)
            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)

# Initializes scene for carrot on plate task
@register_env("PutCarrotOnPlateInScene-v0_IROM", max_episode_steps=64)
class PutCarrotOnPlateInSceneIROM(PutOnBridgeInSceneEnvIROM):
    def __init__(self, **kwargs):
        self.is_dir_light = kwargs.get("is_dir_light", True)
        self.is_ambient_light = kwargs.get("is_ambient_light", True)
        self.dir_light_position = kwargs.get("dir_light_position", [-0.5, 0, -math.sqrt(3)/2])
        self.dir_light_color = kwargs.get("dir_light_color", [0.5,0.5,0.5])
        self.ambient_light_color = kwargs.get("ambient_light_color", [0.3,0.3,0.3])
        self.shadow = kwargs.get("shadow", True)
        self.dir_light_scale = kwargs.get("dir_light_scale", 10)
        self.shadow_map_size = kwargs.get("shadow_map_size", 2048)

        ## Original:
        source_obj_name = "bridge_carrot_generated"
        target_obj_name = "bridge_plate_objaverse_larger_irom"
        carrot_scale = 1.25
        plate_scale = 1.154

        # source_obj_name = "bridge_carrot_generated_modified"
        # target_obj_name = "bridge_plate_objaverse_larger_irom"
        self.params = kwargs
        self.xy_center = np.array([-0.0775, 0.17]) # Left red button position from lower left corner.
        self.carrot_center = np.array([-0.05375, 0.019]) + self.xy_center
        self.carrot_left = np.array([-0.05375, -0.0325]) + self.xy_center
        self.carrot_right = np.array([-0.05375, 0.0775]) + self.xy_center
        self.plate = np.array([-0.14, 0.355]) # From origin
        xy_configs = [np.array([self.carrot_left, self.plate]), np.array([self.carrot_center, self.plate]), np.array([self.carrot_right, self.plate])]
        quat_configs = [
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
        ]

        # Applying the model scaling
        self.warm = False # Apply the warm filter or not
        
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            warm=self.warm,
            **kwargs,
        )

    def get_plate(self):
        return self.plate
    
    def get_carrot(self):
        return self.carrot_left, self.carrot_center, self.carrot_right
    
    def get_xy_center(self):
        return self.xy_center
    
    def get_language_instruction(self, **kwargs):
        # return "put carrot on plate" # Original put carrot on plate command
        return "place the carrot on yellow plate" # Different command
    
    def _initialize_actors(self):
        # Move the robot far away to avoid collision
        # self.agent.robot.set_pose(sapien.Pose([-10, 0, 0])) # Original
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))
        super()._initialize_actors()

    def add_lighting_params(self, **light_kwargs):
        self.is_dir_light = light_kwargs.get("is_dir_light", True)
        self.is_ambient_light = light_kwargs.get("is_ambient_light", True)
        self.dir_light_position = light_kwargs.get("dir_light_position", [-0.5, 0, -math.sqrt(3)/2])
        self.dir_light_color = light_kwargs.get("dir_light_color", [0.5,0.5,0.5])
        self.ambient_light_color = light_kwargs.get("ambient_light_color", [0.3,0.3,0.3])
        self.shadow = light_kwargs.get("shadow", True)
        self.dir_light_scale = light_kwargs.get("dir_light_scale", 10)
        self.shadow_map_size = light_kwargs.get("shadow_map_size", 2048)
        
    def _setup_lighting(self):
        if self.bg_name is not None:
            return
        self.enable_shadow = self.shadow ## Check this and make sure it doesn't break things anywhere else
        if self.is_ambient_light:
            self._scene.set_ambient_light(self.ambient_light_color) # Keeping the magnitude same keeps the color the same
        
        if self.is_dir_light:
            self._scene.add_directional_light(
                self.dir_light_position, # originally on top: [0,0,-1]
                self.dir_light_color,
                position=[0, 0, 1],
                shadow=self.enable_shadow,
                scale=self.dir_light_scale,
                shadow_map_size=self.shadow_map_size,
            )

    def set_carrot_poses(self, center, left, right):
        self.carrot_center = center
        self.carrot_left = left
        self.carrot_right = right
