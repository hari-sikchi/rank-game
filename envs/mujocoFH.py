# Fixed Horizon wrapper of mujoco environments
import gym
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

class MujocoFH(gym.Env):
    def __init__(self, env_name, T=1000, r=None, obs_mean=None, obs_std=None, seed=1):
        self.env = gym.make(env_name)
        self.T = T
        self.r = r
        self.original_mass = np.copy(self.env.model.body_mass)
        self.original_inertia = np.copy(self.env.model.body_inertia)
        self.original_friction = np.copy(self.env.model.geom_friction)
        assert (obs_mean is None and obs_std is None) or (obs_mean is not None and obs_std is not None)
        self.obs_mean, self.obs_std = obs_mean, obs_std

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if 'pen' in env_name:
            self.max_episode_len=100
        elif 'door' in env_name or 'hammer' in env_name:
            self.max_episode_len=200
        else:
            self.max_episode_len=1000
        self.seed(seed)
        
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None

        self.obs = self.env.reset()
        self.obs = self.normalize_obs(self.obs)
        return self.obs.copy()



    def change_env(self, scale=1.0):
        mass = np.copy(self.original_mass)
        inertia = np.copy(self.original_inertia)
        friction = np.copy(self.original_friction)

        self.scale = scale
        
        # change mass
        mass = self.scale  * mass  # 0.5~2.5*mass
        inertia = self.scale * inertia
             
        # change friction
        friction[4,0] = self.scale* friction[4,0]
        self.env.model.geom_friction[:] = friction
        self.env.reset()   

        print("Original body mass: {}".format(self.original_mass))
        print("Adapted body mass: {}".format(self.env.model.body_mass))
        return


    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            self.obs, r, done, info = self.env.step(action)
            self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                r = self.r(prev_obs)

            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
    
    def normalize_obs(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs



class HopperInverse(gym.Env):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_len=1000

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.obs = self.env.reset()
        return self.obs.copy()
    def render(self):
        self.env.render("rgb_array")
    
    def step(self, action):
        posbefore = self.env.sim.data.qpos[0]

        prev_obs = self.obs.copy()
        self.obs, r, done, info = self.env.step(action)
        posafter, height, ang = self.env.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posbefore - posafter) / self.env.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        done = False
         
        return self.obs.copy(), reward, done, done




class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/fixed_swimmer.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.site_xpos[0,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.site_xpos[0,0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
        
