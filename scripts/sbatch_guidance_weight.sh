#!/bin/bash

############################################################################################################

#SBATCH --partition=iris
##SBATCH --partition=iris-hi

#SBATCH --chdir=/iris/u/rheamal/Isaac-GR00T
#SBATCH --output=slurm/PickNPlace_guidanceSweep_2.out
#SBATCH --job-name=Can

#SBATCH --time=9:30:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris9,iris-hgx-1

############################################################################################################

echo "Nodes allocated for this job: $SLURM_NODELIST"
date
source /sailhome/rheamal/.bashrc
conda activate gr00t || source /sailhome/rheamal/tools/miniconda/etc/profile.d/conda.sh && conda activate gr00t

export PYTHONUNBUFFERED=1

############################################################################################################

SEED=1
NSAMPLE=1
HORIZONS="8 4 1"
GUIDANCES="0 -1 -10 -100 -500 -1000 -5000"

for AH in $HORIZONS; do
    for GUIDANCE in $GUIDANCES; do
        echo "Running: ACTION_HORIZON=${AH}, GUIDANCE_SCALE=${GUIDANCE}"

        python scripts/eval_policy.py \
          --plot \
          --model_path nvidia/GR00T-N1-2B\
          --action_horizon ${AH} \
          --guidance_scale ${GUIDANCE} \
        
        # python scripts/eval_policy.py \
        #   --plot \
        #   --dataset_path demo_data/gr1_arms_only.CanSort \
        #   --model_path nvidia/GR00T-N1-2B \
        #   --data_config gr1_arms_only \
        #   --embodiment_tag gr1 \
        #   --steps 150 \
        #   --trajs 1 \
        #   --action_horizon ${AH} \
        #   --guidance_scale ${GUIDANCE}


        echo "Completed: ACTION_HORIZON=${AH}, GUIDANCE_SCALE=${GUIDANCE}"
    done
done

date
