# Self-Guided Action Diffusion (Self-GAD)

Self-GAD is a lightweight inference-time guidance method for improving diffusion-based robot policies. By nudging each action prediction toward a prior trajectory, Self-GAD improves performance, robustness, and sample efficiency.

---

## Overview

Self-GAD adds a small update step during the diffusion process to guide the policy toward previously predicted actions. This improves temporal consistency and helps the policy adapt to dynamic or noisy environments.

---

## Key Features

- **Plug-in Guidance**: Works with existing diffusion-based policies like GR00T-N1.
- **Inference-Time Only**: No additional training needed.
- **Robust**: Performs better in dynamic and out-of-distribution settings.
- **Sample-Efficient**: Outperforms coherence sampling with fewer samples.

---

## Method Illustration

<img src="media/adaptive_new.jpeg" width="700" alt="Self-GAD Guidance Illustration" />

Self-GAD nudges predicted actions toward prior trajectories during diffusion denoising.

---

## Results on GR00T-N1

<img src="media/N1.jpeg" width="700" alt="Self-GAD on GR00T-N1" />

Self-GAD improves closed-loop success rates on RoboCasa and DexMG with GR00T-N1.

---

## How to Use

```python
# Run inference with Self-GAD
action = policy.get_action(obs, prior=prev_action, guidance_scale=0.7)
