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

import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device) #"ViT-B/32", device=device)

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
        self.batchnorm = None #self.cfg['train']['batchnorm']
        self.lang_fusion_type = None #self.cfg['train']['lang_fusion_type']
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

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, total_length, token_type_index, position_index, lang_goals, emb, batch_size, labels, labels_sp, pos):
        pretrain = False
        # emb_shape: [batch_size, 12, 50, 50, 3]

        texts = []
        imgs = []
        loc_dict = ['left', 'right', 'top', 'bottom', 'middle']
        labels_sp = [[] for i in range(batch_size)]
        # new_lang = [[] for i in range(batch_size)]
        new_pos = [[] for i in range(batch_size)]

        num_of_object = 12

        with torch.no_grad():
            for i in range(batch_size):
                words = [j for j in lang_goals[i].split()] 
                img = []
                for j in range(num_of_object):
                    image = preprocess(Image.fromarray(emb[i][j])).unsqueeze(0).to(device)
                    image_features = model.encode_image(image)  # [1, 1024]
                    img.append(image_features)
                img = torch.cat(img, dim=0).float() # [num_of_object, 1024]
                imgs.append(img)
                # for a in range(5):
                #     for b in range(5):
                #         new_word = copy.deepcopy(words)
                #         new_word[8] = loc_dict[a]
                #         new_word[12] = loc_dict[b]
                #         # new_lang.append(' '.join(new_word))
                #         new_lang.append(new_word)
                #         # print('new_lang', new_lang)
                #         id_block = find_target(blocks[i], pos[i], loc_dict[a])
                #         id_bowl = find_target(bowls[i], pos[i], loc_dict[b])

                #         new_pos[i].append([[xx[0]/320, xx[1]/160] for xx in pos[i]])

                #         labels_sp[i].append([0 for i in range(12)])

                #         labels_sp[i][-1][id_block] = 1
                #         labels_sp[i][-1][id_bowl] = 1

                lang = clip.tokenize(words).to(device)
                lang = model.encode_text(lang).float().cuda()   #[15, 1024]
                texts.append(lang)

        imgs = torch.stack(imgs, dim=0) # [bz, num_of_object, 1024]

        texts = torch.stack(texts, dim=0) #[bz, 15, 1024]
        texts = texts.reshape(batch_size*15, 1024)
        text_emb = self.text_encoder(texts) # [batch_size*15, 240]
        text_emb = text_emb.reshape(batch_size, 15, 240).cuda()
        padding_0 = torch.zeros(batch_size, 5, 240).cuda()
        text_emb = torch.cat([text_emb, padding_0], dim=1)  #[bz, 20, 240]
        padding_1 = torch.zeros(batch_size, 20, 256).cuda()
        text_emb = torch.cat([text_emb, padding_1], dim=3)  #[bz, 20, 496]

        imgs_emb = imgs.reshape(batch_size*num_of_object, 1024) # [batch_size*num_of_object, 1024]
        imgs_emb = self.imgs_encoder(imgs_emb) # [batch_size*num_of_object, 240]
        pos = torch.tensor(new_pos).cuda().reshape(batch_size*num_of_object, 2).float()
        pos = self.pos_emb(pos) # [bz*num_of_object, 256]
        pos = pos.reshape(batch_size*num_of_object, 256)

        imgs_emb = torch.cat([imgs_emb, pos], dim=1)
        imgs_emb = imgs_emb.reshape(batch_size, num_of_object, 496)
        input = torch.cat([text_emb, imgs_emb], dim=1) # [batch_size, 20+num_of_object, 496]

        position_index = [[] for x in range(batch_size)]
        for x in range(batch_size):
            for j in range(20):
                position_index[x].append(j)
            for k in range(num_of_object):
                position_index[x].append(20+k)

        position_index = torch.LongTensor(position_index).cuda()
        position_embed = self.position_embeddings(position_index) # [batch_size, 32, 8]

        type_index = [[] for x in range(batch_size)]
        for x in range(batch_size):
            for j in range(20):
                type_index[x].append(0)
            for k in range(num_of_object):
                type_index[x].append(1)

        type_index = torch.LongTensor(type_index).cuda()
        type_embed = self.token_type_embeddings(type_index) # [batch_size, 32, 8]

        input = torch.cat([input, position_embed, type_embed], dim=-1) # [batch_size, 32, 256]
        input = input.reshape(batch_size, 32, 512)
        latent_input = input.transpose(1, 0)  # [32, batch_size, 256]
        encode = self.encoder(latent_input) # [32, batch_size, 256]
        # encode = latent_input
        out = encode[20:32].transpose(1,0) # [batch_size, 12, 256]


        sp_out = out # [batch_size, 12, 256]
        sp_out = sp_out.reshape(batch_size*12, 512)
        sp_out_f = self.sp_decoder(sp_out).reshape(batch_size, 12)
        sp_out_f = F.softmax(sp_out_f, dim=-1)
        sp_out_f = sp_out_f.reshape(batch_size, 12, 1).float()

        labels_sp = np.array(labels_sp)

        labels_sp = torch.tensor(labels_sp).cuda()
        labels_sp = labels_sp.reshape(batch_size, 12).float()
        labels_sp = F.softmax(labels_sp, dim=-1)
        labels_sp = labels_sp.reshape(batch_size, 12, 1).float()

        print('sp_out', sp_out_f)
        print('labels_sp', labels_sp)

        return sp_out_f, labels_sp 


if __name__ == '__main__':

    # load model
    attention = SSPTransformer(
        img_size = 320*160, 
        img_in = 2048*7*7,
        img_out = 1024,
        output_dim=1,
        preprocess=utils.preprocess,
        cfg=None,
        device=device,
    )
    attention.load_state_dict(torch.load(os.path.join('/home/luoqian/revised_cliport', 'pretrained_transformer/pretrained_0.pth')))

    attention.eval()

    total_length = None
    token_type_index = None
    position_index = None
    lang_goals = ["put a on b", "put a on b", ...] # len: batch_size
    emb = [batch_size, num_of_objects, 50, 50, 3] # rgb img [50, 50, 3]]
    batch_size =
    labels = None
    labels_sp = [[0, 0, 1, ...], [0, 0, 1, ...]] #[batch_size, num_of_objects]
    pos = [[[0.1, 0.9], [0.2, 0.7]], [], ..] #[batch_size, num_of_objects, 2]
    # 注：pos第一维小代表left，pos第二维小代表top

    loss0, loss1 = attention.forward(total_length, token_type_index, position_index, lang_goals, emb, batch_size, labels, labels_sp, pos)

