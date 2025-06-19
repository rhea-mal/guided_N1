import os
import torch
import gr00t
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy

MODEL_PATH = "nvidia/GR00T-N1-2B"

task="gr1_arms_only.CanSort"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
# DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
DATASET_PATH = os.path.join(REPO_PATH, f"demo_data/{task}")

EMBODIMENT_TAG = "gr1"

device = "cuda" if torch.cuda.is_available() else "cpu"
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# Ensure the videos directory exists
videos_dir = os.path.join(REPO_PATH, "videos")
os.makedirs(videos_dir, exist_ok=True)

data_config = DATA_CONFIG_MAP["gr1_arms_only"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

print(policy.model)

# Create the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)


import matplotlib.pyplot as plt

traj_id = 0
max_steps = 150

state_joints_across_time = []
gt_action_joints_across_time = []
images = []

sample_images = 6

for step_count in range(max_steps):
    data_point = dataset.get_step_data(traj_id, step_count)
    state_joints = data_point["state.right_arm"][0]
    gt_action_joints = data_point["action.right_arm"][0]
    
   
    state_joints_across_time.append(state_joints)
    gt_action_joints_across_time.append(gt_action_joints)

    # We can also get the image data
    if step_count % (max_steps // sample_images) == 0:
        image = data_point["video.ego_view"][0]
        images.append(image)

# Size is (max_steps, num_joints == 7)
state_joints_across_time = np.array(state_joints_across_time)
gt_action_joints_across_time = np.array(gt_action_joints_across_time)


# Plot the joint angles across time
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 2*7))

for i, ax in enumerate(axes):
    ax.plot(state_joints_across_time[:, i], label="state joints")
    ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
    ax.set_title(f"Joint {i}")
    ax.legend()

plt.tight_layout()
plt.show()

# Save joint plot
joint_plot_path = os.path.join(videos_dir, f"{task}_traj{traj_id}_joint_plot.png")
plt.savefig(joint_plot_path)
plt.close(fig)
print(f"Saved joint plot to {joint_plot_path}")

# Plot and save images
fig, axes = plt.subplots(nrows=1, ncols=sample_images, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")
plt.tight_layout()
image_grid_path = os.path.join(videos_dir, f"{task}_traj{traj_id}_ego_images.png")
plt.savefig(image_grid_path)
plt.close(fig)
print(f"Saved ego-view image grid to {image_grid_path}")