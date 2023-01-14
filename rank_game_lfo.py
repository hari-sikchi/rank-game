import os
import sys, os, time
import numpy as np
import gym
import envs
import torch
from logging_utils.logx import EpochLogger
import json, copy
import d4rl
import d4rl.gym_mujoco
from reward_agent.ranking_losses import rank_pal_auto, rank_ral_auto
from reward_agent.reward_model import MLPReward
from policy_agent.sac import ReplayBuffer, SAC
import eval
import pickle
import random
from ruamel.yaml import YAML

def reproduce(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def try_evaluate(itr: int, policy_type: str, sac_info):

    update_time = itr * v['reward']['gradient_step']
    if v['sac']['epochs']<1.0:
        env_steps = itr * 1 * v['env']['T']
    else:
        env_steps = itr * v['sac']['epochs'] * v['env']['T']
    agent_emp_states = samples[0].copy()

    metrics = {}
    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], True)
    metrics['Real Det Return'] = real_return_det
    print(f"real det return avg: {real_return_det:.2f}")
    logger.log_tabular("Real Det Return", round(real_return_det, 2))
    logger.log_tabular(f"{policy_type} Update Time", update_time)
    logger.log_tabular(f"{policy_type} Env Steps", env_steps)
    return real_return_det

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--sac_epochs', type=float, default=1)
    parser.add_argument('--irl_epochs', type=int, default=1)
    parser.add_argument('--regularization', type=str, default="none")
    parser.add_argument('--max_reward_iterations', type=int, default=-1)
    parser.add_argument('--reward_regularization', type=str, default="weight_decay")
    parser.add_argument('--weight_decay_coeff', type=float, default=0.01)
    parser.add_argument('--reward_bound', type=float, default=10.0)
    
    parser.add_argument('--expert_episodes', type=int, default=1)
    parser.add_argument('--obj', type=str, default='rank-pal-auto')
    parser.add_argument('--exp_name', type=str, default='dump')
    parser.add_argument('--config', type=str, default='configs/hopper.yml')
    args = parser.parse_args()

    yaml = YAML()
    v = yaml.load(open(args.config))

    # Overwrite config parameters with user arguments
    v['obj']=args.obj
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = args.seed
    v['irl']['epochs']=args.irl_epochs
    v['irl']['expert_episodes']=args.expert_episodes
    v['sac']['epochs']=args.sac_epochs
    v['reward']['clamp_magnitude']=args.reward_bound
    v['exp_name']=args.exp_name
    if args.max_reward_iterations==-1:
        args.max_reward_iterations=None
    if(args.regularization!="none"):
        v['irl']['regularization']=args.regularization
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    reproduce(seed)
    pid=os.getpid()
    

    # Setup logging
    exp_id = f"results/{env_name}/" + v['exp_name'] # task/obj/date structure
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    log_folder = exp_id + '/'+exp_id+'_s'+str(seed) 
    logger_kwargs={'output_dir':log_folder, 'exp_name':exp_id}
    logger = EpochLogger(**logger_kwargs)
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp rank_game.py {log_folder}')
    os.system(f'cp reward_agent/ranking_losses.py {log_folder}')
    with open(os.path.join(logger.output_dir, 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    os.makedirs(os.path.join(log_folder, 'plt'),exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'model'),exist_ok=True)

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # Load expert dataset used for imitation
    if( torch.is_tensor(torch.load(f'expert_data/states/{env_name}_airl.pt'))):
        expert_trajs = torch.load(f'expert_data/states/{env_name}_airl.pt').numpy()[:, :, state_indices]
        expert_actions = torch.load(f'expert_data/actions/{env_name}_airl.pt').numpy()[:num_expert_trajs,:,:]
    else:
        expert_trajs = torch.load(f'expert_data/states/{env_name}_airl.pt')[:, :, state_indices]
        expert_actions = torch.load(f'expert_data/actions/{env_name}_airl.pt')[:num_expert_trajs,:,:]


    expert_trajs = expert_trajs[:num_expert_trajs, :, :] # select first expert_episodes
    expert_samples = expert_trajs.copy().reshape(-1, len(state_indices))

    # Initilialize reward as a neural network
    if v['reward']['type']=='vanilla':
        reward_func = MLPReward(len(state_indices), **v['reward'], reward_regularization=args.reward_regularization, device=device).to(device)
    else:
        raise NotImplementedError

    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=v['reward']['lr'], 
        weight_decay=0.01, betas=(v['reward']['momentum'], 0.999))
        

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    past_samples = None
    
    start_time = time.time()
    for itr in range(v['irl']['n_itrs']):
        if v['rl']=='sac':
            if v['sac']['reinitialize'] or itr == 0:
                replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=v['sac']['buffer_size'])
                sac_agent = SAC(env_fn, replay_buffer,
                    steps_per_epoch=v['env']['T'],
                    update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
                    max_ep_len=v['env']['T'],
                    seed=seed,
                    start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                    reward_state_indices=state_indices,
                    device=device,
                    **v['sac']
                )
            sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac

        sac_info, samples = sac_agent.update_and_collect(print_out=True,debug=False)
        agent_emp_states = samples[0].copy()
        if 'ral' in v['obj']:
            if past_samples is None:
                past_samples = agent_emp_states
            else:
                past_samples = np.concatenate((past_samples,agent_emp_states),axis=0)
        agent_emp_states = agent_emp_states.reshape(-1,agent_emp_states.shape[2]) # n*T states


        incorrect_ordering_ratio = None
        identity_firing_ratio = None
        # Update the reward player
        reward_losses = []
        loss = 0
        for _ in range(v['reward']['gradient_step']):
            if v['obj'] == 'rank-pal-auto':
                loss,_ = rank_pal_auto(v['obj'], samples, expert_samples, reward_func,reward_optimizer, device, regularization=v['irl']['regularization'],reward_bound=args.reward_bound,epochs=v['irl']['epochs'],max_iterations=args.max_reward_iterations)
            elif v['obj'] == 'rank-ral-auto':
                incorrect_ordering_ratio, loss = rank_ral_auto(v['obj'], past_samples, expert_samples, reward_func,reward_optimizer, device, regularization=v['irl']['regularization'],epochs=v['irl']['epochs'],max_iterations=args.max_reward_iterations)
            # elif v['obj'] == 'contrastive-maxentirl-sigmoid-validation':
            #     incorrect_ordering_ratio, loss = contrastive_maxentirl_sigmoid_loss_validation(v['obj'], past_samples, expert_samples, reward_func,reward_optimizer, device, regularization=v['irl']['regularization'],epochs=v['irl']['epochs'])
            
        # evaluating the learned reward

        real_return_det = try_evaluate(itr, "Running", sac_info)
        if real_return_det > max_real_return_det:
            max_real_return_det = real_return_det
            # Save the policy player
            torch.save(sac_agent.ac.state_dict(), logger.output_dir+f"/best_policy.pkl")


        if incorrect_ordering_ratio is not None:
            logger.log_tabular("IncorrectOrderingRatio", incorrect_ordering_ratio)
        else:
            logger.log_tabular("IncorrectOrderingRatio", -1)
        if identity_firing_ratio is not None:
            logger.log_tabular("IdentityFiringRatio", identity_firing_ratio)
        else:
            logger.log_tabular("IdentityFiringRatio", -1)
        logger.log_tabular("Time Elasped", time.time()-start_time)
        logger.log_tabular("Iteration", itr)
        logger.log_tabular("Ranking Loss", loss)
        logger.log_tabular("Learned Reward Eval", np.array(sac_info[0]).mean())
        if v['sac']['automatic_alpha_tuning']:
            logger.log_tabular("alpha", sac_agent.alpha.item())
        
        # if v['irl']['save_interval'] > 0 and (itr % v['irl']['save_interval'] == 0 or itr == v['irl']['n_itrs']-1):
        #     # import ipdb;ipdb.set_trace()
        #     torch.save(reward_func.state_dict(), logger.output_dir+f"/reward_model_{itr}.pkl")
        #     torch.save(sac_agent.ac.state_dict(), logger.output_dir+f"/policy_{itr}.pkl")

        logger.dump_tabular()