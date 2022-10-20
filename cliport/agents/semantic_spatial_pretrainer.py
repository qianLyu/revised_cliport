import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
from cliport.models.core.attention import Attention
from cliport.models.core.transport import Transport
from cliport.models.streams.two_stream_attention import TwoStreamAttention
from cliport.models.streams.two_stream_transport import TwoStreamTransport

from cliport.models.streams.two_stream_attention import TwoStreamAttentionLat
from cliport.models.streams.two_stream_transport import TwoStreamTransportLat

from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import build_model, load_clip, tokenize
from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat, FusionConcat, FusionMult
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy
import random

from cliport.models.ssp_transformer import SSPTransformer
import clip

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model, preprocess = clip.load("RN50", device=device) # Load any model
# model = model.eval() # Inference Only

class SSPTrainerAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds
        # self._load_clip()

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        # self.crop_size = 64
        # self.n_rotations = cfg['train']['n_rotations']

        # self.pix_size = 0.003125
        # # self.in_shape = (320, 160, 6)
        # self.in_shape = (320, 160, 8)
        # self.cam_config = cameras.RealSenseD415.CONFIG
        # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr'])
            # 'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }
        self.criterion = nn.L1Loss(reduction='mean')

        # print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    # def _load_clip(self):
    #     model, self.preprocess = load_clip("RN50", device=self.device)
    #     self.clip_rn50 = build_model(model.state_dict()).to(self.device)
    #     del model

    # def encode_image(self, img):
    #     with torch.no_grad():
    #         img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
    #     return img_encoding, img_im

    # def encode_text(self, x):
    #     with torch.no_grad():
    #         tokens = tokenize([x]).to(self.device)
    #         text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

    #     text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
    #     return text_feat, text_emb, text_mask

    def _build_model(self):
        # stream_one_fcn = 'plain_resnet'
        # stream_two_fcn = 'clip_lingunet'
        self.attention = SSPTransformer(
            img_size = 320*160, 
            img_in = 2048*7*7,
            img_out = 1024,
            output_dim=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

        # self.attention.load_state_dict(torch.load(os.path.join(self.cfg['train']['train_dir'], 'pretrained_ae_checkpoints/pretrained_ae.pth')))
        self.attention.load_state_dict(torch.load(os.path.join('/home/luoqian/revised_cliport', 'pretrained_ae_checkpoints/pretrained_ae.pth')))

        # for param in self.attention.autoencoder.parameters():
        #     param.requires_grad = False

        # self.attention.build_trans(encoder_hidden_dim=16)


    def attn_forward(self, inp, softmax=True):
        inp_color = inp['inp_color']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_color, lang_goal)
        return out

    def attn_training_step(self, max_obj_num, token_type_index, position_index, emb, batch_size, labels, backprop=True, compute_err=False):

        total_length = max_obj_num * 2 + 1
        out, label = self.attention.forward(total_length, token_type_index, position_index, emb, batch_size)
        return self.attn_criterion(backprop, out, label)

    def act(self, obs, selected_mask, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs, selected_mask)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }

    # def _build_model(self):
    #     self.attention = None
    #     self.transport = None
    #     raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()


    def attn_criterion(self, backprop, out, label):

        #label = torch.from_numpy(np.array([label])).to(dtype=torch.float, device=out.device)

        # Get loss.
        # loss = self.cross_entropy_with_logits(out, label)

        loss = self.criterion(out, label)
        # print('loss', loss)
        # print('out, label, loss', out, label, loss)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        return loss

    def process_data(self, max_obj_num, batch_size, lang_goals, found_objs_imgs, labels):
        # x = np.array([found_objs_imgs[0][0], found_objs_imgs[0][0], found_objs_imgs[0][0]])
        token_type_index = []
        position_index = []
        emb = []
        emb_decoder = []
        relabel = []
        # label = []
        batch_size = 3
        e = []
        # for j in range(batch_size):
            # t = []
            # p = []
            # e = []
            # e_d = []
            # r = []
            # t.append(0)
            # p.append(0)
            # random.shuffle(found_objs_imgs[j])
            # for i in range(3):
            #     x = torch.tensor([found_objs_imgs[j][0]], requires_grad = True).cuda() #[1, 320, 160]
            #     x = torch.flatten(x, start_dim=1)
            #     y = self.attention.img_encoder_forward(x)
            #     e.append(y)


            # e.append([found_objs_imgs[j][0], found_objs_imgs[j][0]])
            # for i in range(1, 3):
            #     e.append([found_objs_imgs[j][i], found_objs_imgs[j][i]])  #np.zeros_like(found_objs_imgs[j][i])])
            # random.shuffle(e)
            # emb.append(e)
        e_left = []
        e_left.append([found_objs_imgs[0][0], found_objs_imgs[0][0]])
        for i in range(1, 3):
            e_left.append([found_objs_imgs[0][i], np.zeros_like(found_objs_imgs[0][i])])
        random.shuffle(e_left)
        emb.append(e_left)
        e_middle = []
        e_middle.append([found_objs_imgs[1][1], found_objs_imgs[1][1]])
        e_middle.append([found_objs_imgs[1][0], np.zeros_like(found_objs_imgs[1][0])])
        e_middle.append([found_objs_imgs[1][2], np.zeros_like(found_objs_imgs[1][2])])
        random.shuffle(e_middle)
        emb.append(e_middle)
        e_right = []
        e_right.append([found_objs_imgs[2][2], found_objs_imgs[2][2]])
        e_right.append([found_objs_imgs[2][0], np.zeros_like(found_objs_imgs[2][0])])
        e_right.append([found_objs_imgs[2][1], np.zeros_like(found_objs_imgs[2][1])])
        random.shuffle(e_right)
        emb.append(e_right)


        relabel = torch.tensor(labels[:batch_size]).cuda()
        # emb = torch.stack(emb, dim = 0)
        emb = torch.tensor(emb).cuda()

        return token_type_index, position_index, emb, batch_size, relabel


    def training_step(self, batch, batch_idx):
        self.attention.train()
        # self.transport.train()

        batch_size = batch['batch_size']
        lang_goals = batch['lang_goals']
        # found_objs_words = batch['found_objs_words']
        found_objs_imgs = batch['found_objs_imgs']
        labels = batch['labels']

        # Get training losses.

        step = self.total_steps + 1

        max_obj_num = 20

        token_type_index, position_index, emb, batch_size, relabel = self.process_data(max_obj_num, batch_size, lang_goals, found_objs_imgs, labels)
        labels = torch.tensor(labels).cuda()

        loss0 = self.attn_training_step(max_obj_num, token_type_index, position_index, emb, batch_size, relabel)
        # if isinstance(self.transport, Attention):
        #     loss1, err1 = self.attn_training_step(frame, compute_err=True)
        # else:
        #     loss1, err1 = self.transport_training_step(frame, compute_err=True)
        total_loss = loss0
        self.log('tr/affordanceloss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            affordanceloss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()
            # torch.save(self.attention.state_dict(), os.path.join('/home/luoqian/revised_cliport', 'pretrained_ae_checkpoints/pretrained_ae.pth'))
            # torch.save(self.attention.state_dict(), os.path.join(self.cfg['train']['train_dir'], 'pretrained_ae_checkpoints/pretrained_ae.pth'))
            #print('afja')


    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'affordance_checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()

        batch_size = batch['batch_size']
        lang_goals = batch['lang_goals']
        # found_objs_words = batch['found_objs_words']
        found_objs_imgs = batch['found_objs_imgs']
        labels = batch['labels']

        # Get training losses.

        max_obj_num = 20

        token_type_index, position_index, emb, batch_size, relabel = self.process_data(20, batch_size, lang_goals, found_objs_imgs, labels)
        labels = torch.tensor(labels).cuda()
        # print('labels', labels.shape)

        total_loss = 0
        loss = self.attn_training_step(max_obj_num, token_type_index, position_index, emb, batch_size, relabel, backprop=False)
        total_loss += loss
        # total_loss /= self.val_repeats

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(total_loss)

        return dict(
            val_affordance_loss=total_loss,
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch+1)

        mean_train_total_loss = np.mean([v['affordanceloss'].item() for v in all_outputs])

        self.log('tr/affordanceloss', mean_train_total_loss)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_affordance_loss'].item() for v in all_outputs])

        self.log('vl/val_affordance_loss', mean_val_total_loss)

        print("val loss: {:.2f}".format(mean_val_total_loss))

        return dict(
            val_affordance_loss=mean_val_total_loss,
        )

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)