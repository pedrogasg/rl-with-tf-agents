
import gym 

import numpy as np

from tf_agents.environments import suite_pybullet, suite_gym

from tf_agents.policies import random_tf_policy

from tf_agents.environments import tf_py_environment


if __name__ == '__main__':
    
    py_env = suite_gym.load('HumanoidPyBulletEnv-v0')
    py_env.render(mode="human")
    env = tf_py_environment.TFPyEnvironment(py_env)
    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())
    step = env.reset()

    for i in range(1000):

        done = False
        game_rew = 0

        while not done:
            action = policy.action(step)
            step = env.step(action.action)
            done = step.is_last()
            game_rew += step.reward
            env.render(mode="human")

            if done:
                print(step)
                print('Episode %d finished, reward:%d' % (i, game_rew))
                env.reset()

    env.close()
