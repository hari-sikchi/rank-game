import numpy as np


def evaluate_real_return(policy, env, n_episodes, horizon, deterministic):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ret = 0
        for t in range(horizon):
            action = policy(obs, deterministic)
            obs, rew, done, _ = env.step(action) # NOTE: assume rew=0 after done=True for evaluation
            ret += rew 
            if done:
                break
        returns.append(ret)
    
    return np.mean(returns)
