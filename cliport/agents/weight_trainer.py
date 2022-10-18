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

from cliport.models.simple_lang_fusion import SimpleLingFusion

class WeightTrainerAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']

        self.pix_size = 0.003125
        # self.in_shape = (320, 160, 6)
        self.in_shape = (320, 160, 8)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr'])
            # 'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }
        self.criterion = nn.L1Loss(reduction='mean')
        # print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def _build_model(self):
        # stream_one_fcn = 'plain_resnet'
        # stream_two_fcn = 'clip_lingunet'
        self.attention = SimpleLingFusion(
            output_dim=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        # self.transport = TwoStreamTransportLangFusion(
        #     stream_fcn=(stream_one_fcn, stream_two_fcn),
        #     in_shape=self.in_shape,
        #     n_rotations=self.n_rotations,
        #     crop_size=self.crop_size,
        #     preprocess=utils.preprocess,
        #     cfg=self.cfg,
        #     device=self.device_type,
        # )

    def attn_forward(self, inp, softmax=True):
        inp_color = inp['inp_color']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_color, lang_goal)
        return out

    def attn_training_step(self, colorname, label, frame, backprop=True, compute_err=False):
        inp_color = colorname
        lang_goal = frame['lang_goal']

        inp = {'inp_color': inp_color, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        #if backprop == False:
        # print('out', 'label', out, label)
        return self.attn_criterion(backprop, out, label)

    # def trans_forward(self, inp, softmax=True):
    #     inp_img = inp['inp_img']
    #     p0 = inp['p0']
    #     lang_goal = inp['lang_goal']

    #     out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax)
    #     return out

    # def transport_training_step(self, frame, backprop=True, compute_err=False):
    #     inp_img = frame['img']
    #     p0 = frame['p0']
    #     p1, p1_theta = frame['p1'], frame['p1_theta']
    #     lang_goal = frame['lang_goal']

    #     inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}
    #     out = self.trans_forward(inp, softmax=False)
    #     err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
    #     return loss, err

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
        # Get label.
        # theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        # theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        # inp_img = inp['inp_img']
        # label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        # label = np.zeros(label_size)
        # label[p[0], p[1], theta_i] = 1
        # label = label.transpose((2, 0, 1))
        # label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(np.array([label])).to(dtype=torch.float, device=out.device)

        # Get loss.
        # loss = self.cross_entropy_with_logits(out, label)
        
        loss = self.criterion(out, label)
        # print('out, label, loss', out, label, loss)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        # err = {}
        # if compute_err:
        #     pick_conf = self.attn_forward(inp)
        #     pick_conf = pick_conf.detach().cpu().numpy()
        #     argmax = np.argmax(pick_conf)
        #     argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        #     p0_pix = argmax[:2]
        #     p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        #     err = {
        #         'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
        #         'theta': np.absolute((theta - p0_theta) % np.pi)
        #     }
        return loss


    # def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
    #     itheta = theta / (2 * np.pi / self.transport.n_rotations)
    #     itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

    #     # Get one-hot pixel label map.
    #     inp_img = inp['inp_img']
    #     label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
    #     label = np.zeros(label_size)
    #     label[q[0], q[1], itheta] = 1

    #     # Get loss.
    #     label = label.transpose((2, 0, 1))
    #     label = label.reshape(1, np.prod(label.shape))
    #     label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
    #     output = output.reshape(1, np.prod(output.shape))
    #     loss = self.cross_entropy_with_logits(output, label)
    #     if backprop:
    #         transport_optim = self._optimizers['trans']
    #         self.manual_backward(loss, transport_optim)
    #         transport_optim.step()
    #         transport_optim.zero_grad()
 
    #     # Pixel and Rotation error (not used anywhere).
    #     err = {}
    #     if compute_err:
    #         place_conf = self.trans_forward(inp)
    #         place_conf = place_conf.permute(1, 2, 0)
    #         place_conf = place_conf.detach().cpu().numpy()
    #         argmax = np.argmax(place_conf)
    #         argmax = np.unravel_index(argmax, shape=place_conf.shape)
    #         p1_pix = argmax[:2]
    #         p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

    #         err = {
    #             'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
    #             'theta': np.absolute((theta - p1_theta) % np.pi)
    #         }
    #     self.transport.iters += 1
    #     return err, loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        # self.transport.train()

        frame, _ = batch

        # Get training losses.
        for colorname, label in frame['img'].items():
            step = self.total_steps + 1

            loss0 = self.attn_training_step(colorname, label, frame, compute_err=True)
        # if isinstance(self.transport, Attention):
        #     loss1, err1 = self.attn_training_step(frame, compute_err=True)
        # else:
        #     loss1, err1 = self.transport_training_step(frame, compute_err=True)
            total_loss = loss0
            self.log('tr/weightloss', total_loss)
            self.total_steps = step

            self.trainer.train_loop.running_loss.append(total_loss)

            self.check_save_iteration()

        return dict(
            weightloss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        # if (global_step + 1) in self.save_steps:
        #     self.trainer.run_evaluation()
        #     val_loss = self.trainer.callback_metrics['val_loss']
        #     steps = f'{global_step + 1:05d}'
        #     filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
        #     checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        #     ckpt_path = os.path.join(checkpoint_path, filename)
        #     self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()
            torch.save(self.attention.state_dict(), os.path.join(self.cfg['train']['train_dir'], 'weight_checkpoints/weights.pth'))
            #print('afja')


    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'weight_checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()

        total_loss = 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            for colorname, label in frame['img'].items():
                loss = self.attn_training_step(colorname, label, frame, backprop=False, compute_err=True)
                total_loss += loss
        total_loss /= self.val_repeats

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(total_loss)

        return dict(
            val_weight_loss=total_loss,
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch+1)

        mean_train_total_loss = np.mean([v['weightloss'].item() for v in all_outputs])

        self.log('tr/weightloss', mean_train_total_loss)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_weight_loss'].item() for v in all_outputs])

        self.log('vl/val_weight_loss', mean_val_total_loss)

        print("val loss: {:.2f}".format(mean_val_total_loss))

        return dict(
            val_weight_loss=mean_val_total_loss,
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