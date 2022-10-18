"""Put Blocks in Bowl Task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
import pybullet as p


class PutBlockInBowlSpatialUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {loc_pick} {pick} block in the {loc_place} {place} bowl"
        self.task_completed_desc = "done placing blocks in bowls."
        self.loc_dict = ['left', 'middle', 'right']
        self.pick_attr_label = {}  #format: [obj_id, label]
        self.pick_spat_label = {}  #format: [attr_obj_id, label]
        self.place_attr_label = {}  #format: [obj_id, label]
        self.place_spat_label = {}  #format: [attr_obj_id, label]

    def reset(self, env):
        super().reset(env)
        # n_bowls = np.random.randint(1, 4)
        # n_blocks = np.random.randint(1, n_bowls + 1)

        loc_pick_id = np.random.randint(0, 3)
        loc_place_id = np.random.randint(0, 3)
        # print(loc_pick_id, loc_place_id)

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
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            # print('bowl_pose', bowl_pose)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[1] + [1])
            # bowl_poses.append(bowl_pose)
            bowls.append([bowl_id, bowl_pose])
            self.place_attr_label[bowl_id] = 1
            self.pick_attr_label[bowl_id] = 0
            self.place_spat_label[bowl_id] = 0
        bowls.sort(key=lambda x: x[1][0][0])
        place_bowl_pose = bowls[loc_place_id][1]
        self.place_spat_label[bowls[loc_place_id][0]] = 1

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append([block_id, block_pose])
            self.pick_attr_label[block_id] = 1
            self.place_attr_label[block_id] = 0
            self.pick_spat_label[block_id] = 0
        blocks.sort(key=lambda x: x[1][0][0])
        pick_block_id = blocks[loc_pick_id][0]
        self.pick_spat_label[pick_block_id] = 1
        # print('blocks', blocks)
        # print('pick_block_id', pick_block_id)

        # Goal: put each block in a different bowl.
        self.goals.append(([(pick_block_id, (np.pi / 2, None))], np.ones((1, 1)),
                           place_bowl_pose, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(loc_pick=self.loc_dict[loc_pick_id], pick=selected_color_names[0], loc_place=self.loc_dict[loc_place_id],
                                                         place=selected_color_names[1]))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]

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
            color = colors[n_distractors % len(colors)]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1
            self.pick_attr_label[obj_id] = 0
            self.place_attr_label[obj_id] = 0
        # print('self.pick_attr_label', self.pick_attr_label)
        # print('self.place_attr_label', self.place_attr_label)
        # print('self.pick_spat_label', self.pick_spat_label)
        # print('self.place_spat_label', self.place_spat_label)

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockInBowlSpatialSeenColors(PutBlockInBowlSpatialUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PutBlockInBowlSpatialFull(PutBlockInBowlSpatialUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors