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
        self.lang_template = "pick the {color_pick} block in the middle of {loc_place_1}{color_place_1} {obj_place_1} and {loc_place_2}{color_place_2} {obj_place_2}"
        self.loc_dict = ['left ', 'right ', 'top ', 'bottom ', 'middle ']
        self.batch_size = 2
        self.lang_goals = []
        self.labels = []
        self.found_objs = {}
        self.obj_pos = {}

    def find_target(self, ids, loc_place):
        loc = []
        for id in ids:
            loc.append([id, [self.obj_pos[id][0][0], self.obj_pos[id][0][1]]])

        if loc_place == 'left ':
            loc.sort(key=lambda x: x[1][0])
            return loc[0][0]
        if loc_place == 'middle ':
            loc.sort(key=lambda x: x[1][0])
            return loc[1][0]
        if loc_place == 'right ':
            loc.sort(key=lambda x: x[1][0])
            return loc[2][0]
        if loc_place == 'top ':
            loc.sort(key=lambda x: x[1][1])
            return loc[0][0]
        if loc_place == 'bottom ':
            loc.sort(key=lambda x: x[1][1])
            return loc[2][0]


    def reset(self, env):
        super().reset(env)
        # n_bowls = np.random.randint(1, 4)
        # n_blocks = np.random.randint(1, n_bowls + 1)

        # loc_pick_id = np.random.randint(0, 3)

        self.obj_pos = {}
        self.batch = []

        n_blocks = 6
        n_bowls = 6

        all_color_names = self.get_colors()

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowls = []
        while len(bowls) < n_bowls:
        # for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            if not bowl_pose:
                continue
            if not bowl_id:
                continue    
            # p.changeVisualShape(bowl_id, -1, rgbaColor=colors[0] + [1])
            # bowl_poses.append(bowl_pose)
            bowls.append(bowl_id)
            self.obj_pos[bowl_id] = bowl_pose

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        blocks = []
        while len(blocks) < n_blocks:
        # for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            # p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            if not block_pose:
                continue
            if not block_id:
                continue    
            blocks.append(block_id)  # block_pose: ((x,y,z), rotation)            
            self.obj_pos[block_id] = block_pose
            # print('block_pose', block_pose)

        # sort blocks by x
        # blocks.sort(key=lambda x: x[1][0][0])

        while len(self.batch) < self.batch_size:
            obj_changed = {}
            label = {}
            label_sp = {}
            selected_color_names = random.sample(all_color_names, 3)
            colors = [utils.COLORS[cn] for cn in selected_color_names]
            # distractor_colors = [[c, utils.COLORS[c]] for c in utils.COLORS if c not in selected_color_names]
            distractor_colors = [c for c in utils.COLORS if c not in selected_color_names]

            loc_place_1 = np.random.choice(self.loc_dict, 1, replace=False)[0]
            loc_place_2 = np.random.choice(self.loc_dict, 1, replace=False)[0]
            lang_goal = self.lang_template.format(color_pick=selected_color_names[0], loc_place_1=loc_place_1,\
                                     color_place_1=selected_color_names[1], obj_place_1='block', loc_place_2=loc_place_2,\
                                     color_place_2=selected_color_names[2], obj_place_2='bowl')

            selected_blocks = np.random.choice(blocks, 4, replace=False)
            # change color of the pick block
            p.changeVisualShape(selected_blocks[0], -1, rgbaColor=colors[0] + [1])
            obj_changed[selected_blocks[0]] = []
            label[selected_blocks[0]] = 0
            label_sp[selected_blocks[0]] = 0
            # change color of 3 place blocks
            for i in range(1, 4):
                p.changeVisualShape(selected_blocks[i], -1, rgbaColor=colors[1] + [1])
                obj_changed[selected_blocks[i]] = []
                label[selected_blocks[i]] = 1
                label_sp[selected_blocks[i]] = 0

            id_block = self.find_target(selected_blocks[1:4], loc_place_1)
            label_sp[id_block] = 1
            # change color of distractor blocks
            for i, item in enumerate(blocks):
                if item not in obj_changed:
                    c = np.random.choice(distractor_colors, 1, replace=False)[0]
                    p.changeVisualShape(item, -1, rgbaColor=utils.COLORS[c] + [1])
                    obj_changed[item] = []
                    label[item] = 0
                    label_sp[item] = 0

            selected_bowls = np.random.choice(bowls, 3, replace=False)
            # change color of 3 place bowls
            for i in range(3):
                p.changeVisualShape(selected_bowls[i], -1, rgbaColor=colors[2] + [1])
                obj_changed[selected_bowls[i]] = []
                label[selected_bowls[i]] = 2
                label_sp[selected_bowls[i]] = 0

            id_bowl = self.find_target(selected_bowls, loc_place_2)
            label_sp[id_bowl] = 1

            # change color of distractor bowls
            for i, item in enumerate(bowls):
                if item not in obj_changed:
                    c = np.random.choice(distractor_colors, 1, replace=False)[0]
                    p.changeVisualShape(item, -1, rgbaColor=utils.COLORS[c] + [1])
                    obj_changed[item] = []
                    label[item] = 0
                    label_sp[item] = 0

            rgb, hmap, obj_mask = self.get_true_image(env)
            # print(rgb.shape) (320, 160, 3)
            for i, item in enumerate(blocks):
                temp_mask = np.uint8(obj_mask == item)
                temp_mask = np.float32(temp_mask)
                obj_changed[item] = np.uint8(rgb * np.asarray([temp_mask, temp_mask, temp_mask]).transpose(1,2,0))
                plt.imshow(obj_changed[item])
                plt.savefig(f'./block{i}.jpg')
                plt.close()
            for i, item in enumerate(bowls):
                temp_mask = np.uint8(obj_mask == item)
                temp_mask = np.float32(temp_mask)
                obj_changed[item] = np.uint8(rgb * np.asarray([temp_mask, temp_mask, temp_mask]).transpose(1,2,0))
                plt.imshow(obj_changed[item])
                plt.savefig(f'./bowl{i}.jpg')
                plt.close()
                # plt.imshow(img)
                # plt.savefig(f'./aaa.jpg')
                # plt.close()
                # print(1)

            imgs = copy.deepcopy(obj_changed)
            labels = copy.deepcopy(label)
            labels_sp = copy.deepcopy(label_sp)
            lang_goals = copy.deepcopy(lang_goal)
            print(labels_sp)
            # print(labels)
            # print(lang_goals)
            self.batch.append({'imgs':imgs, 'labels':labels, 'lang_goal':lang_goals, 'labels_sp':labels_sp})


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
        return self.batch_size, self.loc_dict, self.obj_pos, self.batch

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