
import gym

from utils import compute_avg_return

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling



from tf_agents.environments import suite_pybullet, suite_gym

from tf_agents.utils import common

from tf_agents.trajectories import trajectory

from tf_agents.policies import random_tf_policy

from tf_agents.environments import tf_py_environment

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork

from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

if __name__ == '__main__':
    
    py_env = suite_gym.load('CartPole-v0')
    py_env.render(mode="human")
    env = tf_py_environment.TFPyEnvironment(py_env)


    fc_layer_params = (100,)
    learning_rate = 1e-3
    episodes_count = 10
    num_iterations = 250
    replay_buffer_capacity = 2000
    collect_episodes_per_iteration = 2
    
    log_interval = 25
    eval_interval = 50
    
    actor = ActorDistributionNetwork(env.observation_spec(), env.action_spec(), fc_layer_params=fc_layer_params)
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    train_step_counter = tf.Variable(0)

    agent = ReinforceAgent(env.time_step_spec(), env.action_spec(), actor_network=actor, optimizer=optimizer, normalize_returns=True, train_step_counter=train_step_counter)

    policy = agent.policy
    
    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=replay_buffer_capacity)

    agent.train = common.function(agent.train)

    def collect_episode(environment, policy, num_episodes):

        episode_counter = 0
        environment.reset()

        while episode_counter < num_episodes:
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

            if traj.is_boundary():
                episode_counter += 1

    agent.train_step_counter.assign(0)

    compute_avg_return(env, policy, episodes_count)

    for _ in range(num_iterations):
        collect_episode(env, policy, collect_episodes_per_iteration)
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        replay_buffer.clear()
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            compute_avg_return(env, policy, episodes_count)

    #TODO    replay_observer = [replay_buffer.add_batch]