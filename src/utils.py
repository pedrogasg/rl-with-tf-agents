def compute_avg_return(env, policy, episodes_count):
    step = env.reset()
    total_reward = 0.0
    
    for i in range(episodes_count):

        done = False
        game_reward = 0.0

        while not done:
            action = policy.action(step)
            step = env.step(action.action)
            game_reward += step.reward
            done = step.is_last()
            env.render(mode="human")

            if done:
                print('Episode %d finished, reward:%d' % (i, game_reward))
                env.reset()
        total_reward += game_reward
    avg = total_reward / episodes_count
    print('The average reward is %s' % avg.numpy()[0])
    #env.close()