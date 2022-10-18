import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import build_model, load_clip, tokenize

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat, FusionConcat, FusionMult

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

        self.bottleneck_in = nn.Sequential(nn.Linear(2048*50, 1024), nn.ReLU(True),)
        self.bottleneck_out = nn.Sequential(nn.Linear(1024, 2048*50), nn.ReLU(True),)
        self.im = None

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


class SSPTransformer(nn.Module):
    """ Fuse instruction and objects"""

    def __init__(self, img_size, img_in, img_out, output_dim, cfg, device, preprocess, num_attention_heads=8, encoder_hidden_dim=160, \
                    encoder_dropout=0.1, encoder_activation="relu", encoder_num_layers=8):
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
        self.position_embeddings = torch.nn.Embedding(21, 8)

        encoder_layers = TransformerEncoderLayer(1024, num_attention_heads,
                                                 encoder_hidden_dim, encoder_dropout, encoder_activation)
        self.encoder = TransformerEncoder(encoder_layers, encoder_num_layers)
        self._load_clip()

        self.in_shape = self.get_in_shape((320, 160, 1))
        # self.xnet = models.ResNet43_8s(self.in_shape, 1, self.cfg, self.device, utils.preprocess)
        # self.autoencoder = ResNet_Autoencoder(self.in_shape, 1, self.cfg, self.device, utils.preprocess)
        self.autoencoder = CLIP_AE(self.in_shape, 1, self.cfg, self.device, utils.preprocess)
        # self._build_img_encoder(img_size, img_out)
        # self._build_decoder()
        # self.clip_model, self.preprocess = clip.load("RN50", device=device) # Load any model
        # self.clip_model = self.clip_model.eval() # Inference Only

    def build_trans(self, num_attention_heads=8, encoder_hidden_dim=16, \
                    encoder_dropout=0.1, encoder_activation="relu", encoder_num_layers=8):
        encoder_layers = TransformerEncoderLayer(1024, num_attention_heads,
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

    def forward(self, total_length, token_type_index, position_index, emb, batch_size):
        batch_size = 4
        # emb_shape: [batch_size, 3, 2, 320, 160]

        l_enc, l_emb, l_mask = self.encode_text('left')  #lang_goals[j])
        text_emb = torch.stack([l_enc for i in range(batch_size)], dim=0) # [batch_size, 1, 1024]

        nc = torch.zeros_like(emb)
        x = torch.cat([emb, nc], dim=1)
        input = x[:,:,0,:,:].reshape(batch_size*6, 1, 320, 160)
        input = torch.cat([input, input, input], dim=1)
        label = x[:,:,1,:,:].reshape(batch_size, 6, 320, 160)
        # label = torch.cat([label, label, label], dim=1)
        with torch.no_grad():
            latent = self.autoencoder.encoder_forward(input) # [batch_size*6, 1024]
            latent = latent.reshape(batch_size, 6, 1024)
            latent_input = torch.cat([text_emb, latent], dim=1)
            latent_input = latent_input.transpose(1, 0)  # [7, batch_size, 1024]
        encode = self.encoder(latent_input) # [7, batch_size, 1024]
        decoder_in = encode[1:7].transpose(1,0) # [batch_size, 6, 1024]

        prediction = decoder_in.reshape(batch_size*6, 1024)
        prediction = self.autoencoder.decoder_forward(prediction)
        prediction = torch.reshape(prediction, ((batch_size, 6, 320, 160)))

        # visualization
        with torch.no_grad():    
            predictions = prediction.clone()
            predictions0 = predictions[0][0].detach().cpu()
            predictions0 = np.asarray(predictions0)
            predictions1 = predictions[0][1].detach().cpu()
            predictions1 = np.asarray(predictions1)
            predictions2 = predictions[0][2].detach().cpu()
            predictions2 = np.asarray(predictions2)      

            # label = self.autoencoder.encoder_forward(label) # [batch_size*6, 1024]
            # label = label.reshape(batch_size, 6, 1024)
            truth = label.clone()
            # truth = truth.reshape(batch_size*6, 1024)
            # truth = self.autoencoder.decoder_forward(truth)
            # truth = torch.reshape(truth, ((batch_size, 6, 320, 160)))

            truth0 = truth[0][0].detach().cpu()
            truth0 = np.asarray(truth0)
            truth1 = truth[0][1].detach().cpu()
            truth1 = np.asarray(truth1)
            truth2 = truth[0][2].detach().cpu()
            truth2 = np.asarray(truth2)      

            plt.subplot(2, 3, 1)
            plt.imshow(predictions0)
            plt.subplot(2, 3, 4)
            plt.imshow(truth0)
            plt.subplot(2, 3, 2)
            plt.imshow(predictions1)
            plt.subplot(2, 3, 5)
            plt.imshow(truth1)
            plt.subplot(2, 3, 3)
            plt.imshow(predictions2)
            plt.subplot(2, 3, 6)
            plt.imshow(truth2)
            plt.show()
            plt.savefig('./visualization.jpg')
            plt.close()

        return prediction, label


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