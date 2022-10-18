"""Put Blocks in Bowl Task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy


class PretrainSemanticSpatialUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.pos_eps = 0.05
        self.lang_template = "pick the {loc_pick} {color} {obj}"
        self.loc_dict = ['left', 'right', 'top', 'bottom', 'middle']
        self.batch_size = 10
        self.lang_goals = []
        self.labels = []
        self.found_objs = {}

    def reset(self, env):
        super().reset(env)
        # n_bowls = np.random.randint(1, 4)
        # n_blocks = np.random.randint(1, n_bowls + 1)

        # loc_pick_id = np.random.randint(0, 3)

        self.lang_goals = []
        self.labels = []
        self.found_objs = {}

        n_bowls = 3
        n_blocks = 3

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        # bowl_poses = []
        bowls = []
        bowls_td = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            # print('bowl_pose', bowl_pose)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[0] + [1])
            # bowl_poses.append(bowl_pose)
            bowls.append([bowl_id, bowl_pose])

        bowls.sort(key=lambda x: x[1][0][0])
        self.lang_goals.append(self.lang_template.format(loc_pick='left', color=selected_color_names[0], obj='bowl'))
        self.labels.append([bowls[0][0], []]) 
        self.found_objs[bowls[0][0]] = [f'a {selected_color_names[0]} bowl', []]
        self.lang_goals.append(self.lang_template.format(loc_pick='middle', color=selected_color_names[0], obj='bowl'))
        self.labels.append([bowls[1][0], []]) 
        self.found_objs[bowls[1][0]] = [f'a {selected_color_names[0]} bowl', []]
        self.lang_goals.append(self.lang_template.format(loc_pick='right', color=selected_color_names[0], obj='bowl'))
        self.labels.append([bowls[2][0], []])      
        self.found_objs[bowls[2][0]] = [f'a {selected_color_names[0]} bowl', []]

        bowls_td = copy.deepcopy(bowls)
        bowls_td.sort(key=lambda x: x[1][0][1])
        self.lang_goals.append(self.lang_template.format(loc_pick='top', color=selected_color_names[0], obj='bowl'))
        self.labels.append([bowls_td[0][0], []]) 
        self.lang_goals.append(self.lang_template.format(loc_pick='bottom', color=selected_color_names[0], obj='bowl'))
        self.labels.append([bowls_td[2][0], []])     
        #place_bowl_pose = bowls[loc_place_id][1]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        blocks = []
        blocks_td = []
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[1] + [1])
            blocks.append([block_id, block_pose])
        blocks.sort(key=lambda x: x[1][0][0])
        self.lang_goals.append(self.lang_template.format(loc_pick='left', color=selected_color_names[1], obj='block'))
        self.labels.append([blocks[0][0], []]) 
        self.found_objs[blocks[0][0]] = [f'a {selected_color_names[1]} block', []]
        self.lang_goals.append(self.lang_template.format(loc_pick='middle', color=selected_color_names[1], obj='block'))
        self.labels.append([blocks[1][0], []]) 
        self.found_objs[blocks[1][0]] = [f'a {selected_color_names[1]} block', []]
        self.lang_goals.append(self.lang_template.format(loc_pick='right', color=selected_color_names[1], obj='block'))
        self.labels.append([blocks[2][0], []])      
        self.found_objs[blocks[2][0]] = [f'a {selected_color_names[1]} block', []]

        blocks_td = copy.deepcopy(blocks)
        blocks_td.sort(key=lambda x: x[1][0][1])
        self.lang_goals.append(self.lang_template.format(loc_pick='top', color=selected_color_names[1], obj='block'))
        self.labels.append([blocks_td[0][0], []]) 
        self.lang_goals.append(self.lang_template.format(loc_pick='bottom', color=selected_color_names[1], obj='block'))
        self.labels.append([blocks_td[2][0], []])     
        # print('blocks', blocks)

        # self.goals.append(([(pick_block_id, (np.pi / 2, None))], np.ones((1, 1)),
        #                    place_bowl_pose, False, True, 'pose', None, 1))


        # Colors of distractor objects.
        distractor_bowl_colors = [[c, utils.COLORS[c]] for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [[c, utils.COLORS[c]] for c in utils.COLORS if c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            colorname = colors[n_distractors % len(colors)][0]
            color = colors[n_distractors % len(colors)][1]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1
            if is_block:
                self.found_objs[obj_id] = [f'a {colorname} block', []]
            else:
                self.found_objs[obj_id] = [f'a {colorname} bowl', []]

    def get_dataset(self, env):
        _, hmap, obj_mask = self.get_true_image(env)
        found_objs = []
        for obj_id, content in self.found_objs.items():
            temp_mask = np.uint8(obj_mask == obj_id)
            temp_mask = np.float32(temp_mask)
            found_objs.append([content[0], temp_mask])

        labels = []
        for i in range(self.batch_size):
            temp_mask = np.uint8(obj_mask == self.labels[i][0])
            temp_mask = np.float32(temp_mask)
            labels.append(temp_mask)

        #     # plt.imshow(temp_mask)
        #     # plt.savefig(f'./{colorid}.jpg')
        #     # plt.close()


        # plt.subplot(2, 3, 1)
        # plt.imshow(labels[0])
        # plt.subplot(2, 3, 2)
        # plt.imshow(labels[1])
        # plt.subplot(2, 3, 3)
        # plt.imshow(labels[2])
        # plt.subplot(2, 3, 4)
        # plt.imshow(labels[3])
        # plt.subplot(2, 3, 5)
        # plt.imshow(labels[4])
        # plt.show()
        # plt.savefig('./cc.jpg')
        # plt.close()

        return self.batch_size, self.lang_goals, found_objs, labels

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PretrainSemanticSpatialSeenColors(PretrainSemanticSpatialUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PretrainSemanticSpatialFull(PretrainSemanticSpatialUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors