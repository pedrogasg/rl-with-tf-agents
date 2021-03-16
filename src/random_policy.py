
import gym 

import numpy as np

from utils import compute_avg_return

from tf_agents.environments import suite_pybullet, suite_gym

from tf_agents.policies import random_tf_policy

from tf_agents.environments import tf_py_environment


if __name__ == '__main__':
    
    py_env = suite_gym.load('CartPole-v0')
    py_env.render(mode="human")
    env = tf_py_environment.TFPyEnvironment(py_env)
    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())
    step = env.reset()
    episodes_count = 100

    compute_avg_return(env, policy, episodes_count)
