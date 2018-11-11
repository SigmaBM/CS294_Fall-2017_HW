from random import random

import gym
# os.chdir(os.path.join(os.path.abspath(os.path.curdir), "hw1/"))
import load_policy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_util
from keras import Sequential
from keras.layers import Dense


def roll_out(env_name, policy_fn, render=False, max_timesteps=1000):
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    observations = []
    actions = []

    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        action = policy_fn(obs[None, :])
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
        if render:
            env.render()
        if steps >= max_steps:
            break

    print('return: ', totalr)

    return {'observations': np.array(observations),
            'actions': np.array(actions),
            'returns': np.array([totalr])}


def build_network(output_units, hidden_layers, units_per_layer):
    model = Sequential()
    for _ in range(hidden_layers):
        model.add(Dense(units=units_per_layer, activation='relu'))

    model.add(Dense(units=output_units))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def report_results(expert_returns, novice_returns, config):
    plt.plot(novice_returns)
    plt.plot(expert_returns)
    plt.xlabel('Rollout')
    plt.ylabel('Return')
    plt.title('{}: {} demonstrations, network size {}, {} epochs \n'.format(config['env'], config['demos'],
                                                                              config['nn_size'], config['epochs']))
    plt.savefig('dagger_result.png')


def dagger(env_name, num_rollouts=20, max_timesteps=1000, beta_base=0.5, nn_hidden_layers=2, nn_units_per_layer=64,
           epochs=5):
    def get_beta():
        return pow(beta_base, i)

    data_set = None

    print('loading and building expert policy')
    expert_policy = load_policy.load_policy('experts/{}-v1.pkl'.format(env_name.split('-')[0]))
    print('loaded and built')

    env = gym.make(env_name)
    action_dim = env.action_space.shape[0]
    novice_policy = build_network(action_dim, nn_hidden_layers, nn_units_per_layer)

    def policy(x):
        if get_beta() >= random():
            return expert_policy(x)
        else:
            return novice_policy.predict(x)

    with tf.Session():
        tf_util.initialize()

        # Run DAgger
        for i in range(num_rollouts):

            print("Rollout: %i/%i" % (i, num_rollouts))

            data = roll_out(env_name, policy, max_timesteps=max_timesteps)
            data['actions'] = expert_policy(data['observations'])

            # Aggregate data
            if data_set is None:
                data_set = data
            else:
                for key in ('observations', 'actions', 'returns'):
                    data_set[key] = np.vstack((data_set[key], data[key]))

            # Update the novice policy
            novice_policy.fit(data_set['observations'], data_set['actions'], verbose=2, epochs=epochs, batch_size=64)

    print('reporting results')
    config = {'env': env_name,
              'demos': data_set['observations'].shape[0],
              'nn_size': '({}x{})'.format(nn_hidden_layers, nn_units_per_layer),
              'epochs': epochs}
    report_results(data_set['returns'][0], data_set['returns'], config)

    return novice_policy


if __name__ == "__main__":
    pi = dagger('Humanoid-v2', num_rollouts=200, nn_units_per_layer=256)
