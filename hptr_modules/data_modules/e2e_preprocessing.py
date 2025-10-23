# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict
from omegaconf import DictConfig
import torch
from torch import nn, Tensor

class End2EndPreProcessing(nn.Module):
    def __init__(
        self,
        stitch_view: bool,
        time_step_current: int,
        data_size: DictConfig
    ) -> None:
        super().__init__()

        self.stitch_view = stitch_view
        self.model_kwargs = {"e2e_model": True}

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: agent Dict
            # ego view
                agent/front_left/img: [batch_size, 1079, 972, 3]
                agent/front_left/calib/intr: [batch_size, 9]
                agent/front_left/calib/extr: [batch_size, 16]
                agent/front/img: [batch_size, 1079, 972, 3]
                agent/front/calib/intr: [batch_size, 9]
                agent/front/calib/extr: [batch_size, 16]
                agent/front_right/img: [batch_size, 1079, 972, 3]
                agent/front_right/calib/intr: [batch_size, 9]
                agent/front_right/calib/extr: [batch_size, 16]
                agent/side_left/img: [batch_size, 1079, 972, 3]
                agent/side_left/calib/intr: [batch_size, 9]
                agent/side_left/calib/extr: [batch_size, 16]
                agent/side_right/img: [batch_size, 1079, 972, 3]
                agent/side_right/calib/intr: [batch_size, 9]
                agent/side_right/calib/extr: [batch_size, 16]
                agent/rear_left/img: [batch_size, 587, 972, 3]
                agent/rear_left/calib/intr: [batch_size, 9]
                agent/rear_left/calib/extr: [batch_size, 16]
                agent/rear/img: [batch_size, 551, 972, 3]
                agent/rear/calib/intr: [batch_size, 9]
                agent/rear/calib/extr: [batch_size, 16]
                agent/rear_right/img: [batch_size, 587, 972, 3]
                agent/rear_right/calib/intr: [batch_size, 9]
                agent/rear_right/calib/extr: [batch_size, 16]

            # ego routing intent
                agent/intent: [batch_size, 1]

            # ego twist (current step linear and angular velocities)
                agent/vel: [batch_size, 6]

            # ego history
                history/agent/pose: [batch_size, 16]
                history/agent/pos: [batch_size, 16, 2]
                history/agent/vel: [batch_size, 16, 2]
                history/agent/acc: [batch_size, 16, 2]

            # ego total traj
                agent/pos: [batch_size, 36, 2]
            
            # rater scores
                gt/preference_scores: [batch_size, 3]

            # ego ground truth
                gt/pos: [batch_size, 20, 3]

        Returns: add following keys to batch Dict

            # input images
                input/front_left_img: [batch_size, 1079, 972, 3]
                input/front_img: [batch_size, 1079, 972, 3]
                input/front_right_img: [batch_size, 1079, 972, 3]
                input/side_left_img: [batch_size, 1079, 972, 3]
                input/side_right_img: [batch_size, 1079, 972, 3]
                input/rear_left_img: [batch_size, 587, 972, 3]
                input/rear_img: [batch_size, 551, 972, 3]
                input/rear_right_img: [batch_size, 587, 972, 3]

                input/stitched_img: [batch_size, 7120, 972, 3]

            # input current status
                input/intent: [batch_size]
                input/vel: [batch_size, 6]

            # input history trajectory
                input/ego_attr: [batch_size, 16, 7]

            # reference
                ref/pos:  [batch_size, 20, 2]
                ref/preference_scores: [batch_size, 3]

        """

        # Ego view at current step
        batch["input/front_left_img"] = batch["agent/front_left/img"]
        batch["input/front_img"] = batch["agent/front/img"]
        batch["input/front_right_img"] = batch["agent/front_right/img"]
        batch["input/side_left_img"] = batch["agent/side_left/img"]
        batch["input/side_right_img"] = batch["agent/side_right/img"]
        batch["input/rear_left_img"] = batch["agent/rear_left/img"]
        batch["input/rear_img"] = batch["agent/rear/img"]
        batch["input/rear_right_img"] = batch["agent/rear_right/img"]

        if self.stitch_view:

            #     stitching order 
            #      _______________]______
            #     |              ___  [0]\
            #     [6]  /|    [7]|   \    |  forward
            #     [5] | |       |    | [1]  ------>
            #     [4]  \|    [3]|___/    |
            #     |___________________[2]/
            #                     ]

            batch["input/stitched_img"] = torch.cat(
                [
                    batch["input/front_left_img"],
                    batch["input/front_img"],
                    batch["input/front_right_img"],
                    batch["input/side_right_img"],
                    batch["input/rear_right_img"],
                    batch["input/rear_img"],
                    batch["input/rear_left_img"],
                    batch["input/side_left_img"],
                ], 
                dim=1
                )
            
        # Ego routing intent one hot
        ohe = torch.eye(3).to(batch["agent/intent"].device)
        batch["input/intent"] = batch["input/intent"] = ohe[batch["agent/intent"]-1]

        # Ego twist at current step [batch_size, 6]
        batch["input/vel"] = batch["agent/vel"]
        # vx, vy, vz, wx, wy, wz 

        # Ego history trajectory [batch_size,16, 6]
        batch["input/ego_attr"] = torch.cat(
            [
                batch["history/agent/pos"],
                batch["history/agent/vel"],
                batch["history/agent/acc"]
            ],
            dim=-1
        ) # x, y, vx, vy, ax, az 

        # Check for ground truth (only in train & val)
        # [batch_size, 20, 2] 
        if "gt/pos" in batch:
            batch["ref/pos"] = batch["gt/pos"]

        # Check for rater scores (Not available!)
        if "gt/preference_scores" in batch:
            batch["ref/preference_scores"] = batch["gt/preference_scores"]
        
        return batch