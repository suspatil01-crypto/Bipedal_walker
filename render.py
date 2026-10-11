import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time

ACTION_DIM = 8
max_global_norm = 0.5
OBS_DIM = 153
learning_rate = 3e-4
physics_steps_per_control_step = 5

default_kernel_init = nn.initializers.orthogonal(jnp.sqrt(2))

def gaussian_dist(action, mean, std):
    var = jnp.square(std)
    return -0.5 * (jnp.square(action - mean) / var + 2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi))

class ActorCritic(nn.Module):
    action_dim : int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(features=256, kernel_init=default_kernel_init)(obs)
        x = nn.relu(x)
        x = nn.Dense(features=256, kernel_init=default_kernel_init)(x)
        x = nn.relu(x)
        value = nn.Dense(features=1, kernel_init=default_kernel_init)(x)
        value = jnp.squeeze(value, axis=-1)

        a = nn.Dense(features=256, kernel_init=default_kernel_init)(obs)
        a = nn.relu(a)
        a = nn.Dense(features=256, kernel_init=default_kernel_init)(a)
        a = nn.relu(a)
        mean = nn.Dense(features=self.action_dim, kernel_init=default_kernel_init)(a)

        log_std = self.param('log_std', lambda rng, shape: jnp.zeros(shape,), (self.action_dim,))
        std = jnp.exp(log_std)
        action = mean
        clipped_action = jnp.clip(action, -1.0, 1.0)
        log_prob = jnp.sum(gaussian_dist(action, mean, std), axis=-1)

        return clipped_action, action, value, log_prob, mean, std

model = ActorCritic(ACTION_DIM)

key = jax.random.PRNGKey(seed=0)
key, init_key, noise_key = jax.random.split(key, 3)
dummy_state = jnp.zeros(shape=(OBS_DIM))

variables = model.init({'params':init_key, 'noise':noise_key}, dummy_state)
params = variables['params']

tx = optax.chain(optax.clip_by_global_norm(max_global_norm), optax.adam(learning_rate=learning_rate))
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def load_checkpoint(path, state):
    with open(path, 'rb') as f:
        ck = pickle.load(f)
    state = state.replace(params=jax.tree_util.tree_map(jnp.asarray, ck['params']),
                          opt_state=jax.tree_util.tree_map(jnp.asarray, ck['opt_state']))
    return state, ck['iteration']

state, _ = load_checkpoint('biped_checkpoint.pkl', state)

log_std = state.params['log_std']

mj_model = mujoco.MjModel.from_xml_path('bipedal.xml')
mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
mj_model.opt.iterations = 6
mj_model.opt.ls_iterations = 6
mj_data = mujoco.MjData(mj_model)

ctrl_dt = mj_model.opt.timestep * physics_steps_per_control_step

def make_obs(data):
    return np.concatenate([data.qpos[2:], data.qvel, data.cinert[1:].ravel(), data.cvel[1:].ravel(), data.qfrc_actuator])

mujoco.mj_forward(mj_model, mj_data)
obs = make_obs(mj_data)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 90
    while viewer.is_running():
        t0 = time.time()
        action, _, _, _, _, _ = model.apply({'params':state.params}, obs)
        mj_data.ctrl[:] = action
        mujoco.mj_forward(mj_model, mj_data)

        for i in range(physics_steps_per_control_step):
            mujoco.mj_step(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)
        obs = make_obs(mj_data)
        viewer.cam.lookat[:] = mj_data.qpos[:3]
        viewer.sync()

        sleep = ctrl_dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


