"""Data collection script."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        # print('f', obs['color'][0].shape)
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            act, all_id_image, selected_id_image = agent.act(obs, info, _+1)

            # found_objects, hmaps = dataset.get_objects_vild(obs) #, selected_mask)
            # print('objects', len(found_objects), found_objects)

            # if len(found_objects) < 6:
            #     print('not enough objects')
            #     found_objects = [found_objects[0] for j in range(6)]
            #     hmaps = [hmaps[0] for j in range(6)]

            # template = 'There is'
            # for x in range(6):
            #     template += f' a {found_objects[x]}'
            # template += '. '
    
            episode.append((obs, act, all_id_image, selected_id_image, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break

        # found_objects1, hmaps = dataset.get_objects_vild(obs) #, selected_mask)
        # print('objects_goal', len(found_objects), found_objects)
        episode.append((obs, None, all_id_image, selected_id_image, reward, info))

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99: #and len(found_objects) == 6:
            dataset.add(seed, episode)


if __name__ == '__main__':
    main()
