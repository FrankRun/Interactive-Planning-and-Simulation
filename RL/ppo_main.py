import os
import glob
import time
import torch
import numpy as np
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo import PPO_continuous
from cheryenvironment import SumoEnv
#%%
def evaluate_policy(args, env, agent, state_norm,seed):
    times = 5
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            # a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times

def test(args, env_name,sumofile_dir,sumocfg,seed,framestep,is_vdm):
    checkpoint_path = "PPO_preTrained"+ '/' + env_name + '/'  

    env = SumoEnv(sumofile_dir,sumocfg,seed,is_vdm)
    args.state_dim = env.statenum
    args.action_dim = env.actionnum
    args.max_episode_steps = env.max_episode_length # Maximum number of steps per episode
    
    test_times=5
    agent = PPO_continuous(args)
    agent.load(checkpoint_path,framestep)
    evaluate_reward=0
    for i in range(test_times):
        s = env.reset()

        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            # a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done = env.test(action)

            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
    env.close()
    return evaluate_reward/test_times

def main(args, env_name, number, seed,sumofile_dir,sumocfg,is_vdm):
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)    
    checkpoint_path = directory
    print("save checkpoint path : " + checkpoint_path)
    
    
    env = SumoEnv(sumofile_dir,sumocfg,seed,is_vdm)
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.statenum
    args.action_dim = env.actionnum
    # args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.max_episode_length # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    update_times = 0
    i_episode = 0
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    evaluate_flag = 0
    
    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        i_episode+=1
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                update_times+=1
                replay_buffer.count = 0
                evaluate_flag =1
            # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps % args.evaluate_freq == 0:
            ##Evaluate the policy every 'update' steps
            if evaluate_flag == 1 and done:
                evaluate_flag = 0
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env, agent, state_norm,seed)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/{}_seed_{}.npy'.format( env_name, seed), np.array(evaluate_rewards))
                    
            if total_steps%20000==0:
                agent.save(checkpoint_path,total_steps)
    env.close()

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    No=5
    env_name = "chery_highway"+'%d'%(No)

    sumofile_dir = '../'
    seed=58
    sumocfg = 'jinghugaosu.sumocfg'
    print("training environment name : " + env_name)

    # main(args, env_name=env_name, number=No, seed=seed,sumofile_dir=sumofile_dir,sumocfg=sumocfg,is_vdm=True)
    test(args, env_name=env_name,sumofile_dir=sumofile_dir,sumocfg=sumocfg,seed=seed,framestep=2980000,is_vdm=True)
