from utils import compute_avg_return

from tf_agents.environments import suite_gym

from tf_agents.policies import random_tf_policy

from tf_agents.environments import tf_py_environment

from tf_agents.metrics.tf_metrics import AverageReturnMetric

from drivers import TFRenderDriver

if __name__ == '__main__':
    
    py_env = suite_gym.load('CartPole-v0')
    py_env.render(mode="human")
    env = tf_py_environment.TFPyEnvironment(py_env)
    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

    average = AverageReturnMetric()
    metrics_observer = [average]
    metrics_driver = TFRenderDriver(env, policy, metrics_observer, max_episodes=5)

    time_step = env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    
    average.reset()
    time_step, policy_state = metrics_driver.run(time_step, policy_state)
    print('The average reward is ', average.result().numpy())
    step = env.reset()
    episodes_count = 100

    compute_avg_return(env, policy, episodes_count)
