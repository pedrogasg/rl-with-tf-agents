
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import VarianceScaling, RandomUniform, Constant

from tf_agents.utils import common
from tf_agents.specs import tensor_spec
from tf_agents.train.learner import Learner
from tf_agents.environments import suite_pybullet
from tf_agents.train.utils import train_utils
from tf_agents.drivers.tf_driver import TFDriver
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.behavioral_cloning.behavioral_cloning_agent import BehavioralCloningAgent
from tf_agents.networks.sequential import Sequential
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from tf_agents.policies import random_tf_policy
from drivers import TFRenderDriver

from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.policy_saver import PolicySaver


if __name__ == '__main__':
    
    py_env = suite_pybullet.load('AntBulletEnv-v0')
    py_env.render(mode="human")
    env = tf_py_environment.TFPyEnvironment(py_env)

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    replay_buffer_capacity = 2000
    learning_rate = 1e-3
    fc_layer_params = [128,64,64]

    num_iterations = 100

    log_interval = 2
    eval_interval = 2


    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    
    num_actions = action_tensor_spec.shape[0]

    with strategy.scope():
        collect_policy = tf.saved_model.load('/tmp/models/expert/AntBulletEnv-v0')
        
        dense_layers = [Dense(
            num_units,
            activation=relu) for num_units in fc_layer_params]

        output_layer = Dense(
            num_actions,
            activation=None)

        cloning_net = Sequential(dense_layers + [output_layer])
        optimizer = Adam(learning_rate=learning_rate)
        train_step_counter = train_utils.create_train_step()
        agent = BehavioralCloningAgent(env.time_step_spec(), env.action_spec(), cloning_network=cloning_net, optimizer=optimizer) 

    policy = agent.policy
        
    replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=env.batch_size, max_length=replay_buffer_capacity)
    
    agent.train_step_counter.assign(0)

    replay_observer = [replay_buffer.add_batch]
    with strategy.scope():
        driver = TFDriver(env, collect_policy, replay_observer, max_episodes=100)

    average = AverageReturnMetric()
    metrics_observer = [average]
    metrics_driver = TFRenderDriver(env, policy, metrics_observer, max_episodes=10)

    def experience_fn():
        with strategy.scope():
            return replay_buffer.as_dataset(
                                            num_parallel_calls=3, 
                                            sample_batch_size=64).prefetch(3)


    learner = Learner(
                '/tmp/models/test/behavior_cloning',
                train_step_counter,
                agent,
                experience_fn,
                strategy=strategy,
                run_optimizer_variable_init=False)

    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    for _ in range(num_iterations):
        time_step, policy_state = driver.run(time_step, policy_state)
        train_loss = learner.run(iterations=10)
        step = learner.train_step_numpy

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            average.reset()
            time_step, policy_state = metrics_driver.run(time_step, policy_state)
            print('The average reward is ', average.result().numpy())

    