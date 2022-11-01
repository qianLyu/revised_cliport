import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import build_model, load_clip, tokenize

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat, FusionConcat, FusionMult, DotAttn

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import cliport.models as models
import os
import numpy as np

from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.clip_lingunet_lat import CLIPLingUNetLat
import clip
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import copy
import random

class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out

class CLIP_AE(CLIPLingUNetLat):
    """ CLIP RN50 with U-Net skip connections without language """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _build_decoder(self):
        # self.input_dim = 1024
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

        self.bottleneck_in = nn.Sequential(nn.Linear(2048*50, 256), nn.ReLU(True),)
        self.bottleneck_out = nn.Sequential(nn.Linear(256, 2048*50), nn.ReLU(True),)
        # self.bottleneck_out = nn.Sequential(nn.Linear(256, 2048*100), nn.ReLU(True),)
        self.im = None

        # self.conv2 = nn.Sequential(
        #     # head

        #     ConvBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),

        #     ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),

        #     ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),

        #     # conv2
        #     ConvBlock(64, [16, 16, self.output_dim], kernel_size=3, stride=1,
        #               final_relu=False, batchnorm=self.batchnorm),
        #     IdentityBlock(self.output_dim, [16, 16, self.output_dim], kernel_size=3, stride=1,
        #                   final_relu=False, batchnorm=self.batchnorm),
        # )

    def encoder_forward(self, x):
        # x = self.preprocess(x, dist='clip')

        # x = x[:,:3]  # select RGB
        # x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        # x = x[:,:3]  # select RGB
        x, self.im = self.encode_image(x)
        x = x.to(in_type)# [bz, 2048, 10, 5]
        x = x.reshape(-1, 2048*50)
        # print('x', x.shape)
        x = self.bottleneck_in(x)
        return x

    def decoder_forward(self, x):
        x = self.bottleneck_out(x)

        # x = x.reshape(-1, 256, 40, 20)
        # x = self.conv2(x)


        x = x.reshape(-1, 2048, 10, 5)

        x = self.conv1(x)
        x = self.up1(x, self.im[-2])
        x = self.up2(x, self.im[-3])
        x = self.up3(x, self.im[-4])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        # x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        x = F.interpolate(x, size=(320, 160), mode='bilinear')
        return x

class ResNet_Autoencoder(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(ResNet_Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_shape[-1]
        self.output_dim = output_dim
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        # self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.preprocess = preprocess

        self._make_layers()

    def _make_layers(self):
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(self.input_dim, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.batchnorm else nn.Identity(),
            nn.ReLU(True),

            # fcn
            ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(64, [128, 128, 128], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(128, [256, 256, 256], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(256, [512, 512, 512], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),

        )

        # bottleneck
        self.bottleneck_in = nn.Sequential(nn.Linear(128*10*5, 1024), nn.ReLU(True),)
                                            # nn.Linear(200, 20), nn.ReLU(True),
                                            # nn.Linear(20, 2), nn.ReLU(True),)

        self.bottleneck_out = nn.Sequential(nn.Linear(1024, 128*10*5), nn.ReLU(True),)
                                            # nn.Linear(20, 200), nn.ReLU(True),
                                            # nn.Linear(200, 800), nn.ReLU(True),)

        self.conv2 = nn.Sequential(
            # head

            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),

            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),

            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),

            # conv2
            ConvBlock(64, [16, 16, self.output_dim], kernel_size=3, stride=1,
                      final_relu=False, batchnorm=self.batchnorm),
            IdentityBlock(self.output_dim, [16, 16, self.output_dim], kernel_size=3, stride=1,
                          final_relu=False, batchnorm=self.batchnorm),
        )


    # def encode_text(self, l):
    #     with torch.no_grad():
    #         inputs = self.tokenizer(l, return_tensors='pt')
    #         input_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
    #         text_embeddings = self.text_encoder(input_ids, attention_mask)
    #         text_encodings = text_embeddings.last_hidden_state.mean(1)
    #     text_feat = self.text_fc(text_encodings)
    #     text_mask = torch.ones_like(input_ids) # [1, max_token_len]
    #     return text_feat, text_embeddings.last_hidden_state, text_mask

    def encoder_forward(self, x):
        # x = self.preprocess(x, dist='transporter')
        # encode language
        # l_enc, l_emb, l_mask = self.encode_text(l)
        # l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        # l_input = l_input.to(dtype=x.dtype)

        y = self.conv1(x) # [batch_size, 32, 40, 20]
        # print('y', y.shape)
        # y = torch.flatten(y, start_dim=1)
        # # y = torch.reshape(y, (-1, 512, 800))
        # out = self.bottleneck_in(y)
        # out = torch.reshape(out, (-1, 1024))

        # print('e_out', out.shape)

        return y

    def decoder_forward(self, x):
        # x = self.preprocess(x, dist='transporter')
        # encode language
        # l_enc, l_emb, l_mask = self.encode_text(l)
        # l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        # l_input = l_input.to(dtype=x.dtype)

        # y = self.bottleneck_out(x)  # [batch_size, 512, 40 * 20]
        # y = torch.reshape(y, (-1, 64, 10, 5)) #32*40*20

        out = self.conv2(x)

        # print('d_out', out.shape)

        return out

        # x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        # x = self.decoder1(x)

        # x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        # x = self.decoder2(x)

        # x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        # x = self.decoder3(x)

        # out = self.conv2(x)

        # return out


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device) #"ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)


class SSPTransformer(nn.Module):
    """ Fuse instruction and objects"""

    def __init__(self, img_size, img_in, img_out, output_dim, cfg, device, preprocess, num_attention_heads=32, encoder_hidden_dim=16, \
                    encoder_dropout=0.1, encoder_activation="relu", encoder_num_layers=8):  # [8/32, 16, 8]
        super(SSPTransformer, self).__init__()
        # self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess
        self.token_type_embeddings = torch.nn.Embedding(2, 8)
        self.position_embeddings = torch.nn.Embedding(32, 8)

        encoder_layers = TransformerEncoderLayer(512, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)
        self._load_clip()

        self.in_shape = self.get_in_shape((320, 160, 1))
        # self.xnet = models.ResNet43_8s(self.in_shape, 1, self.cfg, self.device, utils.preprocess)
        # self.autoencoder = ResNet_Autoencoder(self.in_shape, 1, self.cfg, self.device, utils.preprocess)
        # self.autoencoder = CLIP_AE(self.in_shape, 1, self.cfg, self.device, utils.preprocess)
        self.text_encoder = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True),
                                            nn.Linear(512, 240), nn.ReLU(True),)
        self.imgs_encoder = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True),
                                            nn.Linear(512, 240), nn.ReLU(True),)
                                            # nn.Linear(256, 240), nn.ReLU(True),)
        self.s_decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True),
                                        nn.Linear(1024, 512), nn.ReLU(True),
                                        nn.Linear(512, 1), nn.ReLU(True),)
        self.sp_decoder = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True),
                                        nn.Linear(1024, 512), nn.ReLU(True),
                                        nn.Linear(512, 1), nn.ReLU(True),)
        self.mult_fusion = FusionMult(input_dim=1024)
        self.pos_emb = nn.Sequential(nn.Linear(2, 256), nn.ReLU(True),)
        self.sp_a = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True),
                                        nn.Linear(1024, 512), nn.ReLU(True),
                                        nn.Linear(512, 1), nn.ReLU(True),)
        # self.mult_fusion = DotAttn()
        # self._build_img_encoder(img_size, img_out)
        # self._build_decoder()
        # self.clip_model, self.preprocess = clip.load("RN50", device=device) # Load any model
        # self.clip_model = self.clip_model.eval() # Inference Only

    def build_trans(self, num_attention_heads=8, encoder_hidden_dim=16, \
                    encoder_dropout=0.1, encoder_activation="relu", encoder_num_layers=8):
        encoder_layers = TransformerEncoderLayer(256, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)

    def get_in_shape(self, in_shape):
        padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(padding, axis=1)
        in_shape = tuple(in_shape)
        return in_shape

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model

    def img_encoder_forward(self, x):
        out = self.imag_encoder(x)
        return out

        # self.decoder = nn.Sequential(nn.Linear(1024, 256),
        #                                            nn.LayerNorm(256),
        #                                            nn.ReLU(),
        #                                            nn.Linear(256, 128),
        #                                            nn.LayerNorm(128),
        #                                            nn.ReLU(),
        #                                            nn.Linear(128, 1))

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, total_length, token_type_index, position_index, lang_goals, emb, batch_size, labels, label_sp, pos):
        pretrain = False
        # emb_shape: [batch_size, 12, 320, 160, 3]

        # with torch.no_grad():
        #     for i in range(batch_size):
        #         # l_enc_l, l_emb_l, l_mask_l = self.encode_text(lang_goals[i])  #lang_goals[j])
        #         # lang_g = ["a yellow block", "a blue block", "a red block"]
        #         # l_enc_l = clip.tokenize(lang_g).to(device)
        #         # print('l_enc_l', l_enc_l)
        #         words = [j for j in lang_goals[i].split()] 

        #         img = []
        #         for j in range(12):
        #             # plt.imshow(emb[i][j])
        #             # plt.savefig(f'./{j}.jpg')
        #             # plt.close()

        #             image = preprocess(Image.fromarray(emb[i][j])).unsqueeze(0).to(device)
        #             image_features = model.encode_image(image)  # [1, 1024]
        #             img.append(image_features)

        #             # logits_per_image, logits_per_text = model(image, l_enc_l)
        #             # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        #             # print(probs)
        #             # print(fff)
        #             # print('b', image_features.shape)
        #         img_emb = torch.cat(img, dim=0).float() # [12, 1024]

        #         imgs.append(img_emb)

        #     langx = lang_goals[0]

        #     # langx = ''.join([words[8], words[9], words[10]])
        #     langx = clip.tokenize(langx).to(device)
        #     text_langx = model.encode_text(langx).float()
        #     # langx = ''.join([words[8], words[9], words[10]])
        #     # text_langx, _, _ = self.encode_text(langx)
        #     # text_langx = text_langx.float().cuda()
        
        # text_langx = self.text_encoder(text_langx)
        # text_xx = torch.cat([text_langx for i in range(batch_size*12)], dim=0).cuda()

        # # imgs [bz, 12, 1, 512]

        # text_emb = torch.stack(lang, dim=0).float() # [batch_size, 38, 1024]
        # text_emb = text_emb.reshape(batch_size*38, 1024) # [batch_size*38, 1024]
        # text_emb = self.text_encoder(text_emb).reshape(batch_size, 38, 256) # [batch_size, 1, 256]
        # imgs_emb = torch.stack(imgs, dim=0).float() # [batch_size, 12, 512]
        # imgs_emb = imgs_emb.reshape(batch_size*12, 1024) # [batch_size*12, 256]

        # imgs_emb = self.imgs_encoder(imgs_emb)
        # # imgs_emb = F.cosine_similarity(text_xx, imgs_emb, dim=-1)
        # # print('imgs_emb', imgs_emb.shape)
        # pos = torch.tensor(pos).cuda().reshape(batch_size*12, 2).float()
        # pos = self.pos_emb(pos)

        # # imgs_emb = self.mult_fusion(text_xx, imgs_emb)
        # # imgs_emb = self.mult_fusion(pos, imgs_emb)
        # imgs_emb = imgs_emb.reshape(batch_size, 12, 256) # [batch_size, 12, 256]
        # input = torch.cat([text_emb, imgs_emb], dim=1)
        # latent_input = input.transpose(1, 0)  # [50, batch_size, 256]
        # encode = self.encoder(latent_input) # [50, batch_size, 256]
        # encode = latent_input
        # decoder_in = encode[38:50].transpose(1,0) # [batch_size, 12, 256]
        # decoder_in = decoder_in.reshape(batch_size*12, 256)
        # decoder_out = self.decoder(decoder_in).reshape(batch_size, 12)
        # decoder_out = F.softmax(decoder_out, dim=-1)
        # decoder_out = decoder_out.reshape(batch_size, 12, 1).float()


        # labels = np.array(labels)
        # # labels = np.where(labels > 1, 0, labels)
        # # print(labels)
        # labels = torch.tensor(labels).cuda()
        # labels = labels.reshape(batch_size, 12).float()
        # labels = F.softmax(labels, dim=-1)
        # labels = labels.reshape(batch_size, 12, 1).float()

        ############
        texts = []
        imgs = []
        loc_dict = ['left', 'right', 'top', 'bottom', 'middle']
        #loc_dict = ['left', 'right', 'top', 'bottom', 'bottom']
        labels_s = [[] for i in range(batch_size)]
        labels_sp = [[] for i in range(batch_size)]
        # new_lang = [[] for i in range(batch_size)]
        new_pos = [[] for i in range(batch_size)]
  
        blocks = [[] for i in range(batch_size)]
        bowls = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for a, j in enumerate(labels[i]):
                if j == 1:
                    blocks[i].append(a)
                if j == 2:
                    bowls[i].append(a)

        def find_target(objs, positions, ins):
            loc = []
            for i in objs:
                loc.append([i, positions[i]])
            if ins == 'left':
                loc.sort(key=lambda x: x[1][0])
                return loc[0][0]
            if ins == 'middle':
                loc.sort(key=lambda x: x[1][0])
                return loc[1][0]
            if ins == 'right':
                loc.sort(key=lambda x: x[1][0])
                return loc[2][0]
            if ins == 'top':
                loc.sort(key=lambda x: x[1][1])
                return loc[0][0]
            if ins == 'bottom':
                loc.sort(key=lambda x: x[1][1])
                return loc[2][0]

        softmax_index = [[] for i in range(batch_size)] #[bz, 25, 2, 3]
        softmax_label = [[] for i in range(batch_size)] #[bz, 25, 2, 3]

        with torch.no_grad():
            for i in range(batch_size):
                words = [j for j in lang_goals[i].split()] 
                img = []
                for j in range(12):
                    image = preprocess(Image.fromarray(emb[i][j])).unsqueeze(0).to(device)
                    image_features = model.encode_image(image)  # [1, 1024]
                    img.append(image_features)
                img = torch.cat(img, dim=0).float() # [12, 1024]
                imgs.append(torch.stack([img for x in range(25)], dim=0))
                new_lang = []
                for a in range(5):
                    for b in range(5):
                        new_word = copy.deepcopy(words)
                        new_word[6] = 'center'
                        new_word[8] = loc_dict[a]
                        new_word[12] = loc_dict[b]
                        # new_lang.append(' '.join(new_word))
                        new_lang.append(new_word)
                        # print('new_lang', new_lang)
                        id_block = find_target(blocks[i], pos[i], loc_dict[a])
                        id_bowl = find_target(bowls[i], pos[i], loc_dict[b])

                        # print('new_lang', blocks[i], pos[i], id_block)
                        # print(fff)
                        # posx = []
                        # for i in range(3):
                        #     posx.append(pos[0][blocks[i]])  # = pos[0][i for i in blocks]
                        new_pos[i].append([[xx[0]/320, xx[1]/160] for xx in pos[i]])
                        # new_label.append([0 for i in range(12)])
                        # new_label.append([0 for i in blocks])
                        labels_s[i].append([0 for i in range(12)])
                        labels_sp[i].append([0 for i in range(12)])
                        for j in blocks[i]:
                            labels_s[i][-1][j] = 1
                        for j in bowls[i]:
                            labels_s[i][-1][j] = 1
                        labels_sp[i][-1][id_block] = 1
                        labels_sp[i][-1][id_bowl] = 1

                        # softmax_block = [item for item in blocks[i]]
                        # softmax_bowl = [item for item in bowls[i]]
                        # softmax_index[i].append([softmax_block, softmax_bowl])
                        # softmax_label[i].append([[labels_sp[i][-1][x] for x in softmax_index[i][-1][0]], [labels_sp[i][-1][x] for x in softmax_index[i][-1][1]]])
                
                # print('new_lang', new_lang)
                # print(fff)

                text_langx = []
                with torch.no_grad():
                    for k in range(25):
                        langx = clip.tokenize(new_lang[k]).to(device)
                        langx = model.encode_text(langx).float().cuda()   #[15, 1024]
                        text_langx.append(langx)
                texts.append(torch.stack(text_langx, dim=0))

        imgs = torch.stack(imgs, dim=0) # [bz, 25, 12, 1024]
        # imgs_emb = torch.tensor(new_imgs).cuda().reshape(25, 12, 1024).float()
        # imgs = torch.stack(imgs, dim=0) # [bz, 12, 1024]
        
        # text_xx = torch.stack([text_langx for i in range(3)], dim=1).cuda() #[25, 12, 1024]
        # text_emb = text_xx.reshape(25*3, 1024)
        texts = torch.stack(texts, dim=0) #[bz, 25, 15, 1024]
        texts = texts.reshape(batch_size*25*15, 1024)
        text_emb = self.text_encoder(texts) # [25*15, 248]
        text_emb = text_emb.reshape(batch_size, 25, 15, 240).cuda()
        padding = torch.zeros(batch_size, 25, 5, 240).cuda()
        text_emb = torch.cat([text_emb, padding], dim=2)  #[bz, 25, 20, 240]
        pad = torch.zeros(batch_size, 25, 20, 256).cuda()
        text_emb = torch.cat([text_emb, pad], dim=3)  #[bz, 25, 20, 496]

        imgs_emb = imgs.reshape(batch_size*25*12, 1024) # [batch_size*12, 256]
        imgs_emb = self.imgs_encoder(imgs_emb) # [25*12, 240]
        pos = torch.tensor(new_pos).cuda().reshape(batch_size*25*12, 2).float()
        pos = self.pos_emb(pos) # [bz*25*12, 256]
        pos = pos.reshape(batch_size*25*12, 256)
        # imgs_emb = torch.cat([imgs_emb, pos], dim=1)
        # imgs_emb = imgs_emb.reshape(25, 12, 240)
        # imgs_emb = self.mult_fusion(imgs_emb, pos)
        imgs_emb = torch.cat([imgs_emb, pos], dim=1)
        imgs_emb = imgs_emb.reshape(batch_size, 25, 12, 496)
        input = torch.cat([text_emb, imgs_emb], dim=2) # [batch_size, 25, 32, 496]

        position_index = [[[] for i in range(25)] for j in range(batch_size)]
        for x in range(batch_size):
            for i in range(25):
                for j in range(20):
                    position_index[x][i].append(j)
                for k in range(12):
                    position_index[x][i].append(20+k)
                # for k in range(12):
                #     position_index[i].append(20+k)
        position_index = torch.LongTensor(position_index).cuda()
        position_embed = self.position_embeddings(position_index) # [batch_size, 25, 32, 8]

        type_index = [[[] for i in range(25)] for j in range(batch_size)]
        for x in range(batch_size):
            for i in range(25):
                for j in range(20):
                    type_index[x][i].append(0)
                for k in range(12):
                    type_index[x][i].append(1)
                # for k in range(12):
                #     type_index[i].append(2)
        type_index = torch.LongTensor(type_index).cuda()
        type_embed = self.token_type_embeddings(type_index) # [batch_size, 25, 32, 8]

        input = torch.cat([input, position_embed, type_embed], dim=-1) # [batch_size, 25, 32, 256]
        input = input.reshape(batch_size*25, 32, 512)
        latent_input = input.transpose(1, 0)  # [32, batch_size*25, 256]
        encode = self.encoder(latent_input) # [32, batch_size*25, 256]
        # encode = latent_input
        out = encode[20:32].transpose(1,0) # [batch_size*25, 12, 256]
        # out = out.reshape(25, 12, 2, 128)
        semantic_out = out # [:,:,0,:] # [25, 12, 128]
        semantic_out = semantic_out.reshape(batch_size*25*12, 512)
        # semantic_out = self.mult_fusion(semantic_out, pos)

        # softmax_index = np.array(softmax_index).reshape(batch_size*25, 2, 3)        #[bz, 25, 2, 3]
        # softmax_label = np.array(softmax_label).reshape(batch_size*25, 2, 3)      

        sp_out = out # [batch_size*25, 12, 256]

        sp_out = sp_out.reshape(batch_size*25*12, 512)
        # sp_out = self.sp_a(sp_out)
        # sp_out = self.mult_fusion(sp_out, pos)
        # sp_out = self.sp_b(sp_out)
        # sp_out = sp_out.reshape(batch_size*25, 12, 1)

        # qian_out = sp_out
        # qian_out = self.sp_a(qian_out)
        # qian_out = qian_out.reshape(batch_size*25, 12, 1)
        # sp_softmax = []   #[[[] for j in range(2)] for i in range(batch_size*25)]
        # for i in range(batch_size*25):
        #     aa = []
        #     for j in range(2):
        #         a = []
        #         for k in range(3):
        #             a.append(qian_out[i][softmax_index[i][j][k]])
        #             # print('fas', sp_out[i][softmax_index[i][j][k]].shape)
        #             # print(fasd)
        #         a = torch.stack(a, dim=0)
        #         aa.append(a)
        #     aa = torch.stack(aa, dim=0)
        #     sp_softmax.append(aa)
        # sp_softmax = torch.stack(sp_softmax, dim=0)
        # sp_softmax = sp_softmax.reshape(batch_size*25*2, 3)
        # sp_softmax_f = F.softmax(sp_softmax, dim=-1)
        # sp_softmax_f = sp_softmax_f.reshape(batch_size*25*2, 3, 1).float()
        # labels_sof = torch.tensor(softmax_label).cuda()
        # labels_sof = labels_sof.reshape(batch_size*25*2, 3).float()
        # labels_sof = F.softmax(labels_sof, dim=-1)
        # labels_sof = labels_sof.reshape(batch_size*25*2, 3, 1).float()

        # print('sp_out', sp_softmax_f[0])
        # print('labels_sp', labels_sof[0])
        # imgs_emb = self.mult_fusion(text_emb, imgs_emb)
        # imgs_emb = self.mult_fusion(pos, imgs_emb)

        semantic_out = self.s_decoder(semantic_out).reshape(batch_size*25, 12)
        semantic_out_f = F.softmax(semantic_out, dim=-1)
        semantic_out_f = semantic_out_f.reshape(batch_size, 25, 12, 1).float()
        sp_out_f = self.sp_decoder(sp_out).reshape(batch_size*25, 12)
        # sp_out_f = self.mult_fusion(semantic_out, sp_out_f)
        #sp_out_f = sp_out.reshape(batch_size*25, 12)
        sp_out_f = F.softmax(sp_out_f, dim=-1)
        sp_out_f = sp_out_f.reshape(batch_size, 25, 12, 1).float()

        labels_s = np.array(labels_s)
        labels_sp = np.array(labels_sp)
        # labels = np.where(labels > 1, 1, labels)
        # labels_sp = np.array(label_sp)
        # labels = labels + labels_sp
        # print(labels)

        # labels_s = labels_sp #labels_s + labels_sp
        labels_s = torch.tensor(labels_s).cuda()
        labels_s = labels_s.reshape(batch_size*25, 12).float()
        labels_s = F.softmax(labels_s, dim=-1)
        labels_s = labels_s.reshape(batch_size, 25, 12, 1).float()
        print('semantic_out', semantic_out_f[0][0])
        print('labels_s', labels_s[0][0])

        labels_sp = torch.tensor(labels_sp).cuda()
        labels_sp = labels_sp.reshape(batch_size*25, 12).float()
        labels_sp = F.softmax(labels_sp, dim=-1)
        labels_sp = labels_sp.reshape(batch_size, 25, 12, 1).float()
        print('sp_out', sp_out_f[0][20])
        print('labels_sp', labels_sp[0][20])

        # for j in range(12):
        #     plt.subplot(3, 4, (j+1))
        #     plt.imshow(emb[0][j])
        #     print(new_pos[0][0][j])

        # plt.show()
        # plt.savefig('./visualization1.jpg')
        # plt.close()            
        # print(fa)
        # print('sp_out', sp_softmax_f[0])
        # print('labels_sp', labels_sof[0])

        return semantic_out_f, labels_s, sp_out_f, labels_sp # sp_softmax_f, labels_sof #, sp_out_f, labels_sp
        #return semantic_out_f, labels_s, sp_softmax_f, labels_sof #, sp_out_f, labels_sp

        ############

        # labels = np.array(labels)
        # labels = np.where(labels > 1, 1, labels)
        # # labels_sp = np.array(label_sp)
        # # labels = labels + labels_sp
        # # print(labels)
        # labels = torch.tensor(labels).cuda()
        # labels = labels.reshape(batch_size, 12).float()
        # labels = F.softmax(labels, dim=-1)
        # labels = labels.reshape(batch_size, 12, 1).float()
        # print('decoder_out', decoder_out)
        # print('labels', labels)
        # return decoder_out, labels    




        # nc = torch.zeros_like(emb)
        # x = torch.cat([emb, nc], dim=1)
        # input = x[:,:,0,:,:].reshape(batch_size*6, 1, 320, 160)
        # input = torch.cat([input, input, input], dim=1)
        # label = x[:,:,1,:,:].reshape(batch_size, 6, 320, 160)
        # label = torch.cat([label, label, label], dim=1)

        # if pretrain:
        #     latent = self.autoencoder.encoder_forward(input) # [batch_size*6, 1024]
        #     prediction = self.autoencoder.decoder_forward(latent)
        #     prediction = torch.reshape(prediction, ((batch_size, 6, 320, 160)))

        # else:
        #     # with torch.no_grad():
        #     latent = self.autoencoder.encoder_forward(input) # [batch_size*6, 256]
        #     latent = latent.reshape(batch_size, 6, 256)
        #     latent_input = torch.cat([text_emb, latent], dim=1)
        #     latent_input = latent_input.transpose(1, 0)  # [7, batch_size, 256]
        #     encode = self.encoder(latent_input) # [7, batch_size, 256]
        #     decoder_in = encode[1:7].transpose(1,0) # [batch_size, 6, 256]

        #     prediction = decoder_in.reshape(batch_size*6, 256)
        #     prediction = self.autoencoder.decoder_forward(prediction)
        #     prediction = torch.reshape(prediction, ((batch_size, 6, 320, 160)))

        #     # visualization
        # with torch.no_grad():    
        #     predictions = prediction.clone()
        #     predictions0 = predictions[0][0].detach().cpu()
        #     predictions0 = np.asarray(predictions0)
        #     # predictions0 = np.ma.masked_where(predictions0 < 0, predictions0)
        #     predictions0 = np.where(predictions0 > 0, predictions0, 0) 
        #     predictions1 = predictions[0][1].detach().cpu()
        #     predictions1 = np.asarray(predictions1)
        #     predictions1 = np.where(predictions1 > 0, predictions1, 0) 
        #     predictions2 = predictions[0][2].detach().cpu()
        #     predictions2 = np.asarray(predictions2)      
        #     predictions2 = np.where(predictions2 > 0, predictions2, 0) 

        #     # label = self.autoencoder.encoder_forward(label) # [batch_size*6, 1024]
        #     # label = label.reshape(batch_size, 6, 1024)
        #     truth = label.clone()
        #     # truth = truth.reshape(batch_size*6, 1024)
        #     # truth = self.autoencoder.decoder_forward(truth)
        #     # truth = torch.reshape(truth, ((batch_size, 6, 320, 160)))

        #     truth0 = truth[0][0].detach().cpu()
        #     truth0 = np.asarray(truth0)
        #     truth0 = np.where(truth0 > 0, truth0, 0) 
        #     truth1 = truth[0][1].detach().cpu()
        #     truth1 = np.asarray(truth1)
        #     truth1 = np.where(truth1 > 0, truth1, 0) 
        #     truth2 = truth[0][2].detach().cpu()
        #     truth2 = np.asarray(truth2)   
        #     truth2 = np.where(truth2 > 0, truth2, 0)   

        #     plt.subplot(2, 3, 1)
        #     plt.imshow(predictions0)
        #     plt.subplot(2, 3, 4)
        #     plt.imshow(truth0)
        #     plt.subplot(2, 3, 2)
        #     plt.imshow(predictions1)
        #     plt.subplot(2, 3, 5)
        #     plt.imshow(truth1)
        #     plt.subplot(2, 3, 3)
        #     plt.imshow(predictions2)
        #     plt.subplot(2, 3, 6)
        #     plt.imshow(truth2)
        #     plt.show()
        #     plt.savefig('./visualization1.jpg')
        #     plt.close()

        # with torch.no_grad():    
        #     predictions = prediction.clone()
        #     predictions0 = predictions[1][0].detach().cpu()
        #     predictions0 = np.asarray(predictions0)
        #     predictions0 = np.where(predictions0 > 0, predictions0, 0) 
        #     predictions1 = predictions[1][1].detach().cpu()
        #     predictions1 = np.asarray(predictions1)
        #     predictions1 = np.where(predictions1 > 0, predictions1, 0) 
        #     predictions2 = predictions[1][2].detach().cpu()
        #     predictions2 = np.asarray(predictions2)      
        #     predictions2 = np.where(predictions2 > 0, predictions2, 0) 

        #     # label = self.autoencoder.encoder_forward(label) # [batch_size*6, 1024]
        #     # label = label.reshape(batch_size, 6, 1024)
        #     truth = label.clone()
        #     # truth = truth.reshape(batch_size*6, 1024)
        #     # truth = self.autoencoder.decoder_forward(truth)
        #     # truth = torch.reshape(truth, ((batch_size, 6, 320, 160)))

        #     truth0 = truth[1][0].detach().cpu()
        #     truth0 = np.asarray(truth0)
        #     truth0 = np.where(truth0 > 0, truth0, 0) 
        #     truth1 = truth[1][1].detach().cpu()
        #     truth1 = np.asarray(truth1)
        #     truth1 = np.where(truth1 > 0, truth1, 0) 
        #     truth2 = truth[1][2].detach().cpu()
        #     truth2 = np.asarray(truth2)      
        #     truth2 = np.where(truth2 > 0, truth2, 0) 

        #     plt.subplot(2, 3, 1)
        #     plt.imshow(predictions0)
        #     plt.subplot(2, 3, 4)
        #     plt.imshow(truth0)
        #     plt.subplot(2, 3, 2)
        #     plt.imshow(predictions1)
        #     plt.subplot(2, 3, 5)
        #     plt.imshow(truth1)
        #     plt.subplot(2, 3, 3)
        #     plt.imshow(predictions2)
        #     plt.subplot(2, 3, 6)
        #     plt.imshow(truth2)
        #     plt.show()
        #     plt.savefig('./visualization2.jpg')
        #     plt.close()

        # with torch.no_grad():    
        #     predictions = prediction.clone()
        #     predictions0 = predictions[2][0].detach().cpu()
        #     predictions0 = np.asarray(predictions0)
        #     predictions0 = np.where(predictions0 > 0, predictions0, 0) 
        #     predictions1 = predictions[2][1].detach().cpu()
        #     predictions1 = np.asarray(predictions1)
        #     predictions1 = np.where(predictions1 > 0, predictions1, 0) 
        #     predictions2 = predictions[2][2].detach().cpu()
        #     predictions2 = np.asarray(predictions2)     
        #     predictions2 = np.where(predictions2 > 0, predictions2, 0)  

        #     # label = self.autoencoder.encoder_forward(label) # [batch_size*6, 1024]
        #     # label = label.reshape(batch_size, 6, 1024)
        #     truth = label.clone()
        #     # truth = truth.reshape(batch_size*6, 1024)
        #     # truth = self.autoencoder.decoder_forward(truth)
        #     # truth = torch.reshape(truth, ((batch_size, 6, 320, 160)))

        #     truth0 = truth[2][0].detach().cpu()
        #     truth0 = np.asarray(truth0)
        #     truth0 = np.where(truth0 > 0, truth0, 0) 
        #     truth1 = truth[2][1].detach().cpu()
        #     truth1 = np.asarray(truth1)
        #     truth1 = np.where(truth1 > 0, truth1, 0) 
        #     truth2 = truth[2][2].detach().cpu()
        #     truth2 = np.asarray(truth2)      
        #     truth2 = np.where(truth2 > 0, truth2, 0) 

        #     plt.subplot(2, 3, 1)
        #     plt.imshow(predictions0)
        #     plt.subplot(2, 3, 4)
        #     plt.imshow(truth0)
        #     plt.subplot(2, 3, 2)
        #     plt.imshow(predictions1)
        #     plt.subplot(2, 3, 5)
        #     plt.imshow(truth1)
        #     plt.subplot(2, 3, 3)
        #     plt.imshow(predictions2)
        #     plt.subplot(2, 3, 6)
        #     plt.imshow(truth2)
        #     plt.show()
        #     plt.savefig('./visualization3.jpg')
        #     plt.close()

        # return prediction, label            




    # def forward(self, total_length, token_type_index, position_index, emb, emb_decoder):
        # predictions = torch.mean(predictions, dim=0)

        # predictions = torch.reshape(predictions, ((batch_size, 320, 160)))


        # position_embed = self.position_embeddings(position_index)
        # token_type_embed = self.token_type_embeddings(token_type_index)
        # print('afsd', position_embed.shape, token_type_embed.shape)

        # sequence_encode = torch.cat([sentence, x], dim=1)
        # input_raw = torch.cat([emb, position_embed, token_type_embed], dim=-1)
        # # print('gffd', input_raw.shape)
        # padding = torch.zeros((10, (total_length-len(token_type_index[0])), 1040), device='cuda')
        # # print('padding', padding.shape)
        # input_all = torch.cat([input_raw, padding], dim=1)
        # # print('input_all', input_all.shape) # [10, 41, 1040]

        # x = torch.flatten(emb, start_dim=2)
        # y = self.img_encoder_forward(x)

        # nc = torch.zeros_like(emb)
        # input = torch.cat([emb, nc], dim=1)

        # input_raw = y
        # # print('gffd', input_raw.shape)
        # padding = torch.zeros((1, (6-len(emb[0])), 128), device='cuda')
        # # print('padding', padding.shape)
        # input_all = torch.cat([input_raw, padding], dim=1)
        # # print('input_all', input_all.shape) # [128, 6, 1024]


        # #########################
        # # sequence_encode: [batch size, sequence_length, encoder input dimension]
        # # input to transformer needs to have dimenion [sequence_length, batch size, encoder input dimension]
        # input_all = input_all.transpose(1, 0)

        # # encode: [sequence_length, batch_size, embedding_size]
        # # encode = self.encoder(input_all)
        # # print('encode', encode.shape)

        # encode = input_all

        # max_obj_num = (total_length - 1) // 2
        # e = []
        # # for i in range(max_obj_num):
        # #     e.append(encode[2*i+1])
        # # for i in range(5):
        # #     e.append(encode[i+1])
        # e.append(encode[0])
        # emb_decoder_0 = torch.stack(e, dim=0) #[20, 10, 1040]
        # emb_decoder_0 = emb_decoder_0.transpose(1, 0) # [128, 1, 1024]
        # ####################################


        # emb_decoder_0 = self.decoder(emb_decoder_0).reshape((10, 20, 320, 160))
        # padding_decoder = torch.zeros((10, (max_obj_num-len(emb_decoder[0])), 1040), device='cuda')
        # emb_decoder_1 = torch.cat([emb_decoder, padding_decoder], dim=1) # [10, 20, 1040]

        # emb_decoder_all = torch.cat([emb_decoder_0, emb_decoder_1], dim=2) # [10, 20, 2064]

        # emb_decoder_all = torch.flatten(emb_decoder_all, start_dim=1, end_dim=-1) # [10, 20*2064]
        ########################
        # b = emb[0][0].cpu().numpy()
        # emb_decoder_0 = np.asarray([b, b, b])
        # emb_decoder_0.transpose(1,2,0)
        # emb_decoder_0.permute(1, 2, 0)

        # predictions = self.decoder(emb_decoder_0) # [10, 320*160]
        # print('predictions', predictions.shape)
        # predictions = torch.reshape(predictions, ((5, 320, 160)))
        # predictions = torch.mean(predictions,dim=0,keepdim=True)
        # predictions = nn.Softmax(dim=1)(predictions)
        # predictions = torch.argmax(predictions, dim=1)
        # print('predictions', predictions.shape)

        # input = (emb_decoder_0 * emb_decoder_1).permute(0, 2, 3, 1) 
        # predictions = self.transporternet(input)
        # print('pre', predictions.shape)



        # # emb_shape: [batch_size, 1, 320, 160]
        # x = []
    
        # imgs = emb.cpu().numpy()
        # for i, img in enumerate(imgs):
        #     img = Image.fromarray(img[0])
        #     with torch.no_grad():
        #         image = self.preprocess(img).unsqueeze(0).to('cuda')
        #         x.append(self.clip_model.encode_image(image).to('cuda')) # [1, 1024]

        # input = torch.cat([emb, emb, emb], dim=1)
        # x = self.autoencoder.encoder_forward(input)
        # predictions = self.autoencoder.decoder_forward(x)
        # print('predictions', predictions.shape)

        # # emb_shape: [batch_size, 1, 320, 160]
        # x = self.autoencoder.encoder_forward(emb)
        # # x = x.reshape(-1, 32, 2) #32*40*20
        # predictions = self.autoencoder.decoder_forward(x)
        # predictions = torch.reshape(predictions, ((-1, 320, 160)))

        # return prediction