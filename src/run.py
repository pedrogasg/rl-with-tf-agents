
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, Constant

from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.train.learner import Learner
from tf_agents.environments import suite_gym
from tf_agents.train.utils import train_utils
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks.sequential import Sequential
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from tf_agents.policies import random_tf_policy
from drivers import TFRenderDriver

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
        
    with strategy.scope():
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
        train_step_counter = train_utils.create_train_step()        
        agent = DqnAgent(env.time_step_spec(), env.action_spec(), q_network=q_net, optimizer=optimizer, td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=train_step_counter)
    

    policy = agent.policy
        
    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=replay_buffer_capacity)
    
    agent.train_step_counter.assign(0)
        
    replay_observer = [replay_buffer.add_batch]
    driver = TFDriver(env, agent.collect_policy, replay_observer, max_steps=1)

    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())
    random_driver = TFDriver(env, random_policy, replay_observer, max_steps=100, max_episodes=100)         
    average = AverageReturnMetric()
    metrics_observer = [average]

    metrics_driver = TFRenderDriver(env, agent.policy, metrics_observer, max_episodes=10)

    def experience_fn():
        with strategy.scope():
            return replay_buffer.as_dataset(
                                            num_parallel_calls=3, 
                                            sample_batch_size=64, 
                                            num_steps=2).prefetch(3)    

    learner = Learner(
                '/tmp/models',
                train_step_counter,
                agent,
                experience_fn,
                strategy=strategy)

    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)

    time_step, policy_state = random_driver.run(time_step, policy_state)

    for _ in range(num_iterations):
        time_step, policy_state = driver.run(time_step, policy_state)
        train_loss = learner.run(iterations=1)
        step = learner.train_step_numpy

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            average.reset()
            time_step, policy_state = metrics_driver.run(time_step, policy_state)
            print('The average reward is ', average.result().numpy())

 