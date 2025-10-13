import numpy as np
import torch

# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        state, reward, done, info = self.env.step(action[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self):
        return np.array([self.env.reset()])

# TODO: this is probably a better case for inheritance than for a wrapper
# Gives an interface to exploit mirror symmetry
class SymmetricEnv:    
    def __init__(self, env_fn, mirrored_obs=None, mirrored_act=None, clock_inds=None, obs_fn=None, act_fn=None):

        assert (bool(mirrored_act) ^ bool(act_fn)) and (bool(mirrored_obs) ^ bool(obs_fn)), \
            "You must provide either mirror indices or a mirror function, but not both, for \
             observation and action."

        if mirrored_act:
            self.act_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_act))

        elif act_fn:
            assert callable(act_fn), "Action mirror function must be callable"
            self.mirror_action = act_fn

        if mirrored_obs:
            self.obs_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_obs))

        elif obs_fn:
            assert callable(obs_fn), "Observation mirror function must be callable"
            self.mirror_observation = obs_fn

        self.clock_inds = clock_inds
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def mirror_action(self, action):
        return action @ self.act_mirror_matrix

    def mirror_observation(self, obs):
        return obs @ self.obs_mirror_matrix

    # To be used when there is a clock in the observation. In this case, the mirrored_obs vector inputted
    # when the SymmeticEnv is created should not move the clock input order. The indices of the obs vector
    # where the clocks are located need to be inputted.
    def mirror_clock_observation(self, obs):
        # Handle different input shapes
        orig_shape = obs.shape
        if obs.dim() == 1:  # Single obs [features]
            obs = obs.unsqueeze(0).unsqueeze(0)  # To [1, 1, features] (batch=1, seq=1)
            is_recurrent = False
            history_len = 1
        elif obs.dim() == 2:  # Non-recurrent [batch, features] or flat history [batch, history*features]
            obs = obs.unsqueeze(1)  # To [batch, 1, features] for consistency
            is_recurrent = False
            history_len = obs.shape[2] // self.base_obs_len  # Assume flat if not seq
        elif obs.dim() == 3:  # Recurrent [batch, seq_len, features]
            is_recurrent = True
            history_len = obs.shape[1]  # seq_len as history
        else:
            raise ValueError(f"Unsupported obs shape: {orig_shape}")

        mirror_obs_batch = torch.zeros_like(obs)
        for block in range(history_len):
            if is_recurrent:
                obs_ = obs[:, block, :]  # [batch, features]
            else:
                # For flat 3D (after unsqueeze), slice features
                start = self.base_obs_len * block
                end = self.base_obs_len * (block + 1)
                obs_ = obs[:, 0, start:end]  # [batch, features]

            mirror_obs = obs_ @ self.obs_mirror_matrix
            clock = mirror_obs[:, self.clock_inds]
            for i in range(clock.shape[1]):
                mirror_obs[:, self.clock_inds[i]] = torch.sin(
                    torch.asin(clock[:, i]) + torch.pi
                )
            if is_recurrent:
                mirror_obs_batch[:, block, :] = mirror_obs
            else:
                mirror_obs_batch[:, 0, start:end] = mirror_obs
        
        # Restore original shape
        if obs.dim() == 1:
            mirror_obs_batch = mirror_obs_batch.squeeze(0).squeeze(0)
        elif orig_shape == 2:
            mirror_obs_batch = mirror_obs_batch.squeeze(1)  # Back to [batch, features]
        
        return mirror_obs_batch


def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat
