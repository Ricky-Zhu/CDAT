from stable_baselines import SAC, TD3, TRPO, PPO2
import gym
from modified_envs import *
import yaml
from stable_baselines.common.policies import MlpPolicy as mlp_standard
from stable_baselines.sac.policies import FeedForwardPolicy as ffp_sac
from stable_baselines.td3.policies import FeedForwardPolicy as ffp_td3
from stable_baselines.ddpg.noise import NormalActionNoise
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import numpy as np


def training_policy(args):
    training_algo = args.algo
    training_steps = args.training_steps
    env = args.env
    args_env = args.args_env

    model_name = training_algo + '_' + env + '_' + str(training_steps) + '.pkl'

    with open('target_policy_params.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = args[training_algo][args_env]

    if training_algo == "SAC":

        class CustomPolicy(ffp_sac):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   feature_extraction="mlp", layers=[256, 256])

        model = SAC(CustomPolicy, env,
                    verbose=1,
                    tensorboard_log='TBlogs/initial_policy_training',
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    ent_coef=args['ent_coef'],
                    learning_starts=args['learning_starts'],
                    learning_rate=args['learning_rate'],
                    train_freq=args['train_freq'],
                    )
    elif training_algo == "TD3":

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=float(args['noise_std']) * np.ones(n_actions))

        class CustomPolicy2(ffp_td3):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy2, self).__init__(*args, **kwargs,
                                                    feature_extraction="mlp", layers=[400, 300])

        model = TD3(CustomPolicy2, env,
                    verbose=1,
                    tensorboard_log='TBlogs/initial_policy_training',
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    gamma=args['gamma'],
                    gradient_steps=args['gradient_steps'],
                    learning_rate=args['learning_rate'],
                    learning_starts=args['learning_starts'],
                    action_noise=action_noise,
                    train_freq=args['train_freq'],
                    )

    elif training_algo == "TRPO":

        model = TRPO(mlp_standard, env,
                     verbose=1,
                     tensorboard_log='TBlogs/initial_policy_training',
                     timesteps_per_batch=args['timesteps_per_batch'],
                     lam=args['lam'],
                     max_kl=args['max_kl'],
                     gamma=args['gamma'],
                     vf_iters=args['vf_iters'],
                     vf_stepsize=args['vf_stepsize'],
                     entcoeff=args['entcoeff'],
                     cg_damping=args['cg_damping'],
                     cg_iters=args['cg_iters']
                     )
    elif training_algo == "PPO2":
        model = PPO2(mlp_standard,
                     env,
                     n_steps=int(args['n_steps'] / env.num_envs),
                     nminibatches=args['nminibatches'],
                     lam=args['lam'],
                     gamma=args['gamma'],
                     ent_coef=args['ent_coef'],
                     noptepochs=args['noptepochs'],
                     learning_rate=args['learning_rate'],
                     cliprange=args['cliprange'],
                     verbose=1,
                     tensorboard_log='data/TBlogs/initial_policy_training',
                     )

    else:
        raise NotImplementedError

    model.learn(total_timesteps=training_steps,
                tb_log_name=model_name.split('/')[-1],
                log_interval=10, )
    model.save(model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--algo', default='SAC', type=str)
    parser.add_argument('--training_steps', default=1e6, type=int)
    parser.add_argument('--args_env', default='HalfCheetah-v2', type=str)

    args = parser.parse_args()
    training_policy(args)
