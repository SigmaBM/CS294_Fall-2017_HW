import os
import pickle

import gym
# os.chdir(os.path.join(os.path.abspath(os.path.curdir), "hw1/"))
import load_policy
import numpy as np
import tensorflow as tf
import tf_util
from keras import Sequential
from keras.layers import Dense


def generate_expert_data(expert_policy_file, env_name, max_timesteps=None, num_rollouts=20):
    # See if we can load data from file
    data_pkl_path = expert_policy_file.replace(".pkl", "") + env_name + str(num_rollouts) + ".pkl"
    if os.path.exists(data_pkl_path):
        print('loading expert rollouts from file ' + data_pkl_path)
        with open(data_pkl_path, 'rb') as f:
            return pickle.load(f)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    # Roll out policy
    with tf.Session():
        tf_util.initialize()
        data = roll_out(env_name, policy_fn, max_timesteps=max_timesteps, num_rollouts=num_rollouts)

    # Save data for subsequent runs
    with open(data_pkl_path, 'wb') as f:
        print('saving data to file ' + data_pkl_path)
        pickle.dump(data, f)

    return data


def roll_out(env_name, policy_fn, render=False, max_timesteps=1000, num_rollouts=20):
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
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
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return {'observations': np.array(observations),
            'actions': np.array(actions),
            'returns': np.array(returns)}


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
    with open('behavioural_cloning_results.txt', 'a') as f:
        f.write('{}: {} demonstrations, network size {}, {} epochs \n'.format(config['env'], config['demos'],
                                                                           config['nn_size'], config['epochs']))
        f.write('-----------------------------------\n')
        f.write('Agent | Mean Return | Std of Return\n')
        f.write('===================================\n')
        f.write('Expert | {:.2f} | {:.2f}\n'.format(np.mean(expert_returns), np.std(expert_returns)))
        f.write('Novice | {:.2f} | {:.2f}\n'.format(np.mean(novice_returns), np.std(novice_returns)))
        f.write('-----------------------------------\n\n')


def main(env_name, expert_rollouts, nn_hidden_layers=2, nn_units_per_layer=64, epochs=5):
    expert = env_name.split('-')[0]
    training_data = generate_expert_data('experts/{}-v1.pkl'.format(expert), env_name, num_rollouts=expert_rollouts)

    action_dim = training_data['actions'][0].shape[1]
    behavioural_clone = build_network(action_dim, nn_hidden_layers, nn_units_per_layer)

    print('train clone')
    behavioural_clone.fit(training_data['observations'], training_data['actions'].reshape([-1, action_dim]),
                          epochs=epochs, batch_size=64)
    print('test clone')
    novice_data = roll_out(env_name, policy_fn=lambda x: behavioural_clone.predict(x), render=False)

    config = {'env': env_name,
              'demos': training_data['observations'].shape[0],
              'nn_size': '({}x{})'.format(nn_hidden_layers, nn_units_per_layer),
              'epochs': epochs}
    report_results(training_data['returns'], novice_data['returns'], config)


if __name__ == "__main__":
    if os.path.exists('behavioural_cloning_results.txt'):
        os.remove('behavioural_cloning_results.txt')

    for env in ('Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Reacher-v2', 'Walker2d-v2'):
        main(env, expert_rollouts=200)

    # # Grid search over neural net size to see if there is a better config for Humanoid
    # for num_rollouts in (200, 500):
    #     for num_layers in (1, 2, 3):
    #         for layer_size in (32, 64, 128, 256, 512, 1024):
    #             main('Humanoid-v2', num_rollouts, num_layers, layer_size)
