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
import time

class PretrainSemanticSpatialLeftUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.pos_eps = 0.05
        self.lang_template = "pick the {loc_pick} {color} {obj}"
        self.loc_dict = ['left', 'right', 'top', 'bottom', 'middle']
        self.batch_size = 128
        self.lang_goals = 'left'

        self.found_objs = []
        self.batch = {}

    def reset(self, env):
        super().reset(env)

        self.found_objs = {}
        self.batch = {}

        n_blocks = 18

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 1)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        blocks = []
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append([block_id, block_pose])  # block_pose: ((x,y,z), rotation)

        # sort blocks by x
        blocks.sort(key=lambda x: x[1][0][0])

        _, hmap, obj_mask = self.get_true_image(env)
        for i in range(n_blocks):
            # temp_mask: [320, 160] one-hot
            temp_mask = np.uint8(obj_mask == blocks[i][0])
            temp_mask = np.float32(temp_mask)
            self.found_objs[blocks[i][0]] = temp_mask

        while len(self.batch) < self.batch_size:
            selected_obj = np.random.choice(n_blocks, 3, replace=False)
            selected_obj.sort()
            if (selected_obj[0], selected_obj[1], selected_obj[2]) in self.batch:
                continue
            else:
                self.batch[(selected_obj[0], selected_obj[1], selected_obj[2])] = blocks[selected_obj[0]][0] # left block id (label)

        # # check batch
        # for a, b in self.batch.items():
        #     plt.subplot(2, 3, 1)
        #     plt.imshow(self.found_objs[blocks[a[0]][0]])
        #     plt.subplot(2, 3, 2)
        #     plt.imshow(self.found_objs[blocks[a[1]][0]])
        #     plt.subplot(2, 3, 3)
        #     plt.imshow(self.found_objs[blocks[a[2]][0]])
        #     plt.subplot(2, 3, 4)
        #     plt.imshow(self.found_objs[b])
        #     plt.show()
        #     plt.savefig('./batch.jpg')
        #     plt.close()
        #     time.sleep(1)


    def get_dataset(self, env):

        return self.batch_size, self.lang_goals, self.found_objs, self.batch

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PretrainSemanticSpatialLeftSeenColors(PretrainSemanticSpatialLeftUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PretrainSemanticSpatialLeftFull(PretrainSemanticSpatialLeftUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors