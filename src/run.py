
import gym

from utils import compute_avg_return

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, Constant



from tf_agents.environments import suite_pybullet, suite_gym

from tf_agents.utils import common

from tf_agents.trajectories import trajectory

from tf_agents.policies import random_tf_policy

from tf_agents.environments import tf_py_environment

from tf_agents.agents.dqn.dqn_agent import DqnAgent

from tf_agents.policies.random_tf_policy import RandomTFPolicy

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver

from tf_agents.train.utils import strategy_utils

from tf_agents.specs import tensor_spec

from tf_agents.networks.sequential import Sequential


if __name__ == '__main__':
    
    py_env = suite_gym.load('CartPole-v0')
    py_env.render(mode="human")
    env = tf_py_environment.TFPyEnvironment(py_env)


    fc_layer_params = [100,50]
    learning_rate = 1e-3
    episodes_count = 10
    num_iterations = 20000
    replay_buffer_capacity = 2000
    collect_episodes_per_iteration = 2
    initial_collect_steps = 100

    log_interval = 200
    eval_interval = 1000

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    
    dense_layers = [Dense(
        num_units,
        activation=relu,
        kernel_initializer=VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal')) for num_units in fc_layer_params]

    q_values_layer = Dense(
        num_actions,
        activation=None,
        kernel_initializer=RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=Constant(-0.2))

    q_net = Sequential(dense_layers + [q_values_layer])

    
    optimizer = Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    with strategy.scope():
        agent = DqnAgent(env.time_step_spec(), env.action_spec(), q_network=q_net, optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=train_step_counter)

    policy = agent.policy
    
    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=replay_buffer_capacity)
    
    
    agent.train = common.function(agent.train)

    agent.train_step_counter.assign(0)
    
    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
    
    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)

    #driver = DynamicEpisodeDriver(env, policy, replay_observer, num_episodes=collect_episodes_per_iteration)

    random_policy = RandomTFPolicy(env.time_step_spec(), env.action_spec())

    collect_data(env, random_policy, replay_buffer, 100)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=env.batch_size, 
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    collect_data(env, agent.collect_policy, replay_buffer, 2)
    #final_time_step, policy_state = driver.run()

    def _reduce_loss(loss):
      return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
    
    for _ in range(num_iterations):
        #final_time_step, policy_state = driver.run(final_time_step, policy_state)

        collect_data(env, agent.collect_policy, replay_buffer, 2)
        experience, _ = next(iterator)
        loss_info = strategy.run(agent.train, args=(experience,))
        train_loss = tf.nest.map_structure(_reduce_loss, loss_info)
        replay_buffer.clear()
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            compute_avg_return(env, policy, episodes_count)
    


    #TODO    