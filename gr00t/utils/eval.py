# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path

def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    guidance_scale=0.0,
    plot=False,
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []

    prior = None  # ← Initial prior is None

    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        # Build input vector
        concat_state = np.concatenate(
            [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
        )
        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        state_joints_across_time.append(concat_state)
        gt_action_joints_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0:
            print(f"Inferencing at step: {step_count}")

            # Run policy with guidance and prior
            action_chunk = policy.get_action(data_point, prior=prior, guidance_scale=guidance_scale)

            # Save current prediction as next prior
            prior = torch.tensor(
                np.stack(
                    [np.concatenate([np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys], axis=0)
                     for j in range(action_horizon)]
                ),
                dtype=torch.float32
            ).unsqueeze(0)  # Shape: (1, H, A)

            # Add action predictions for this horizon
            for j in range(action_horizon):
                pred = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys], axis=0
                )
                pred_action_joints_across_time.append(pred)

    # Convert to numpy arrays
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]

    assert (
        state_joints_across_time.shape
        == gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    )

    # MSE
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    # Plotting (optional)
    if plot:
        num_of_joints = state_joints_across_time.shape[1]
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))
        fig.suptitle(f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}", fontsize=16)

        for i, ax in enumerate(axes):
            ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action")

            for j in range(0, steps, action_horizon):
                ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point" if j == 0 else "")

            ax.set_title(f"Joint {i}")
            ax.legend()

        plt.tight_layout()
        plt.show()

    return mse
