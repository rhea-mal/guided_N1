from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
import ipdb
# get the data config
data_config = DATA_CONFIG_MAP["gr1_arms_only"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="demo_data/robot_sim.PickNPlace",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.GR1, # the embodiment to use
)

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1-2B",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1,
    device="cuda"
)
vanilla, _ = policy.get_action(dataset[0])  # unpack properly
prior = None
for i in range(2):
    action_chunk, prior = policy.get_action(dataset[0], prior)
import numpy as np

diff = np.linalg.norm(vanilla["action.left_arm"] - action_chunk["action.left_arm"])
print(diff)


