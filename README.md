# Teaching a Bipedal Robot to Walk with PPO (JAX + MuJoCo MJX)

A 3D box-shaped biped learns to walk forward from scratch using **Proximal Policy Optimization (PPO)**, trained entirely on the GPU/TPU with **JAX**, **Flax**, and **MuJoCo MJX**. Thousands of simulated robots are stepped in parallel, so a walking gait emerges in a short amount of training time.

## Highlights

- **Massively parallel simulation:** 3,072 environments run at once via `jax.vmap` + `jax.jit` on MJX.
- **From-scratch PPO:** custom implementation with GAE, clipped objective, and minibatch epochs (no RL library).
- **Actor-critic network in Flax:** separate 2×256 ReLU MLPs for policy and value, with a learnable log-std.
- **Pretrained checkpoint included:** watch the trained walker immediately in the MuJoCo viewer.

## Project Structure

| File | Description |
|------|-------------|
| `main.ipynb` | Training notebook: environment, reward, network, PPO loop, checkpointing |
| `bipedal.xml` | MuJoCo model of the biped (torso, 2 legs, 8 motors) |
| `render.py` | Loads a checkpoint and renders the policy in the interactive MuJoCo viewer |
| `biped_checkpoint.pkl` | Trained weights (params + optimizer state + iteration) |

## The Robot

A simple box-based biped defined in `bipedal.xml`:

- **Torso** with a free joint (6 DoF)
- **Two legs**, each with: hip (x and y axes), knee, and ankle
- **8 actuators** (motors, gear 15, control range [-1, 1])
- Joint limits: hip-x ±20°, hip-y ±45°, knee −80° to 0°, ankle ±20°
- Physics timestep 0.005 s (RK4); 5 physics steps per control step (control at 40 Hz)

## Environment

**Observation (153 dims):** concatenation of
`qpos[2:]` (13) + `qvel` (14) + `cinert[1:]` (70) + `cvel[1:]` (42) + `qfrc_actuator` (14)

**Action (8 dims):** motor commands, clipped to [-1, 1].

**Reward per control step:**

```
reward = 1.25 * forward_velocity  +  5.0 * is_healthy  -  0.1 * sum(action^2)
```

- Forward velocity is measured from the torso center-of-mass displacement along x.
- *Healthy* means torso height `z` stays within `[0.55, 1.3]`.

**Termination:** the episode ends (and the env auto-resets with small noise) when the torso leaves the healthy range, or is truncated after 1,500 steps.

## Training Setup (PPO)

| Hyperparameter | Value |
|----------------|-------|
| Parallel envs | 3,072 |
| Rollout length | 40 steps |
| Minibatches | 24 |
| Epochs per update | 8 |
| Discount (γ) | 0.97 |
| GAE (λ) | 0.95 |
| Clip ε | 0.2 |
| Learning rate | 3e-4 (Adam) |
| Max grad norm | 0.5 |
| Network | 2×256 ReLU (separate actor & critic) |

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install the CUDA build of JAX following the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### 2. Watch the trained walker

`render.py` expects `biped_checkpoint.pkl` and `bipedal.xml` in the same folder as the script. If you rename the checkpoint, update the path inside `render.py` to match.

```bash
python render.py
```

An interactive MuJoCo window opens with a camera that follows the robot. During playback the policy uses the **deterministic mean action** (no exploration noise).

> On macOS, the passive viewer may require running with `mjpython render.py`.

### 3. Train your own

Open `main.ipynb` and run the cells top to bottom. Training periodically saves checkpoints as a pickle containing `params`, `opt_state`, and `iteration`, which `render.py` can load directly.

## Tech Stack

Python · JAX · Flax · Optax · MuJoCo / MJX · NumPy

## Ideas for Improvement

- Add observation/reward normalization
- Add a target-velocity command or turning
- Train on uneven terrain or with domain randomization
- Export training curves and a demo GIF/video for this README
