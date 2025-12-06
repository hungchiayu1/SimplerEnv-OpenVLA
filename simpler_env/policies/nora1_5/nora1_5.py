
from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
from .modelling_expert import VLAWithExpert
from typing import Optional, Callable, Dict, Any, Union, Type, TypeVar

import tensorflow as tf
import dlimp as dl
import PIL.Image as Image
def resize_image(image1):
    #image1 = ds_combined[0]['observation.images.scene']
    #image1 = image1.reshape(480,640,3)
    #print(image1.shape,)
    image1 = tf.cast(image1, dtype=tf.uint8)
    #image1 = image1.numpy().transpose(1,2,0)
    image1 = dl.transforms.resize_image(image1, size=(224,224))
    
    #image1 = Image.fromarray(image1.numpy())
    return image1.numpy()

class Nora1_5Inference:
    # Action token range for the Nora model's vocabulary
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712

    def __init__(
        self,
        saved_model_path: str = "declare-lab/nora-1.5-fractal-dpo",
        # --- MODIFIED: Added policy_setup parameter ---
        policy_setup: str = "widowx_bridge",
        unnorm_key: Optional[str] = None, # Can now be inferred from policy_setup
        horizon: int = 1,
        pred_action_horizon: int = 5,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        action_ensemble_temp: float = -0.8,
        device: Optional[str] = None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # --- ADDED: Logic to handle policy_setup ---
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            # NOTE: Nora's norm_stats.json only contains widowx_bridge.
            # You would need to add google_robot stats for this to be fully equivalent.
            # Defaulting to widowx_bridge for now.
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f"Policy setup '{policy_setup}' is not supported.")
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"*** Using device: {self.device} ***")
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        print(f"*** Loading fast tokenizer for action decoding ***")
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        #print(f"*** Loading main processor from: {saved_model_path} ***")
        self.processor = AutoProcessor.from_pretrained('declare-lab/nora', trust_remote_code=True)

        #print(f"*** Loading model from: {saved_model_path} ***")
        
        self.model = VLAWithExpert.from_pretrained(saved_model_path).to(self.device)
        self.model = self.model.to(torch.bfloat16)
        
        self.model.eval()
        

        

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble_temp = action_ensemble_temp

        if action_ensemble_temp == 0.0:
            self.action_ensemble = False
        else:   
            self.action_ensemble = True
            
        if self.action_ensemble:
            from simpler_env.utils.action.action_ensemble import ActionEnsembler
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, action_ensemble_temp
            )
        self.task_description = None
        
        # --- ADDED: State variables for sticky gripper ---
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        print(f"Policy reset for new task: {task_description}")
        if self.action_ensemble:
            self.action_ensembler.reset()
        
        # --- ADDED: Reset sticky gripper state variables ---
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
    
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8, "Input image must be uint8"
        
        
        unnorm_action = self._get_unnormalized_action(image, self.task_description)[0]
        print(unnorm_action)
        if self.action_ensemble:
            unnorm_action = self.action_ensembler.ensemble_action(unnorm_action)[None]
        
        raw_action = {
            "world_vector": np.array(unnorm_action[0, :3]),
            "rotation_delta": np.array(unnorm_action[0, 3:6]),
            "open_gripper": np.array(unnorm_action[0, 6:7]),
        }
        
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale

        roll, pitch, yaw = raw_action["rotation_delta"]
        rot_ax, rot_angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = rot_ax * rot_angle * self.action_scale

        # --- MODIFIED: Replaced simple gripper logic with the full policy_setup block ---
        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action
            
       
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        # --- END MODIFICATION ---

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    @torch.inference_mode()
    def _get_unnormalized_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        #resize_image
        pil_image = Image.fromarray(self._resize_image(image))
        
        #print(type(pil_image))
        #print(pil_image.shape)
        normalized_action = self.model.sample_actions(pil_image,instruction, 10)
        
        

        action_stats = self._get_action_stats(self.unnorm_key)

        mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_stats["q99"]), np.array(action_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) + action_low,
            normalized_action,
        )
        
       # unnorm_actions = 0.5 * (normalized_action + 1) * (action_high - action_low) + action_low
        #print(unnorm_actions)
        #print(actions)
        return actions

    def _resize_image(self, image: np.ndarray) -> np.ndarray:

        return resize_image(image)

        #return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def _get_action_stats(self, unnorm_key: str) -> Dict[str, Any]:
        if unnorm_key not in self.model.norm_stats:
            raise KeyError(
                f"The `unnorm_key` '{unnorm_key}' is not in the set of available dataset statistics. "
                f"Please choose from: {list(self.model.modelnorm_stats.keys())}"
            )
        return self.model.norm_stats[unnorm_key]["action"]

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[Dict[str, np.ndarray]], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(images[:: max(1, len(images) // 10)], axis=1)

        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array([
            np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]])
            for a in predicted_raw_actions
        ])
        
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
            axs[action_label].grid(True)

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        axs["image"].set_xticks([])
        axs["image"].set_yticks([])

        handles, labels = axs[ACTION_DIM_LABELS[0]].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close(fig)