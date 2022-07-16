import pdb
import copy

import torchvision

import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv

import math


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResnetCustom(nn.Module):

    def __init__(self, in_channels=1):
        super(ResnetCustom, self).__init__()

        # bring resnet
        self.model = torchvision.models.resnet18(pretrained=False)

        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # your case
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)


class SLRModelMF(nn.Module):
    def __init__(self, num_classes, c2d_type, conv_type, use_bn=False, tm_type='BiLSTM',
                 hidden_size=1024, gloss_dict=None, loss_weights=None,
                 use_temporal_attn=False, temporal_embedd_dim=512, temporal_n_heads=4,
                 use_spatial_attn=False, spatial_embedd_dim=512, spatial_n_heads=4):
        super(SLRModelMF, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d_1ch = getattr(models, c2d_type)(pretrained=False)
        self.conv2d_1ch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2d_1ch.fc = Identity()
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)

        self.conv1d_key = TemporalConv(input_size=512,
                                       hidden_size=512,
                                       conv_type=conv_type,
                                       use_bn=use_bn,
                                       num_classes=num_classes)

        self.conv1d_type_1_block1 = TemporalConv(input_size=512,
                                                 hidden_size=hidden_size,
                                                 conv_type=1,
                                                 use_bn=use_bn,
                                                 num_classes=num_classes)

        self.conv1d_type_1_block2 = TemporalConv(input_size=1024,
                                                 hidden_size=hidden_size,
                                                 conv_type=1,
                                                 use_bn=use_bn,
                                                 num_classes=num_classes)

        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')

        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size * 2,
                                          hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        self.use_temporal_attn = use_temporal_attn

        if self.use_temporal_attn:
            self.temporal_attn = MultiHeadedAttention(temporal_n_heads, temporal_embedd_dim, 0.3)
            self.temporal_attn_key = MultiHeadedAttention(temporal_n_heads, 512, 0.3)
            print('Using Temporal Attention layer', use_temporal_attn)
        else:
            self.temporal_attn = None
            self.temporal_attn_key = None
            print('Temporal Attention layer not Used', use_temporal_attn)

        self.use_spatial_attn = use_spatial_attn

        if self.use_spatial_attn:
            self.spatial_attn = nn.MultiheadAttention(spatial_embedd_dim, spatial_n_heads)
            print('Using Spatial Attention layer', use_spatial_attn)
        else:
            self.spatial_attn = None
            print('Spatial Attention layer not Used', use_spatial_attn)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # print(x.shape)
        x = self.conv2d(x)
        # print(x.shape)
        # print('masked_bn')
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def masked_bn_kp(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # print(x.shape)
        x = self.conv2d_1ch(x)
        # print(x.shape)
        # print('masked_bn')
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, key_x, len_x, label=None, label_lgt=None):
        # print(x.shape)
        # print(len_x.shape)
        # print(key_x.shape)

        batch, temp, placeholder, point, axis = key_x.shape
        inputs_kp = key_x.reshape(batch * temp, placeholder, point, axis)
        # # print(inputs_kp.shape)
        keypoints = self.masked_bn_kp(inputs_kp, len_x)
        # # print(keypoints.shape)
        keypoints = keypoints.reshape(batch, temp, -1).transpose(1, 2)
        # # print(keypoints.shape)

        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            # print(inputs.shape)
            framewise = self.masked_bn(inputs, len_x)
            # print(framewise.shape)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            # print(framewise.shape)

        else:
            # frame-wise features
            framewise = x

        if self.use_spatial_attn:
            framewise = torch.reshape(framewise, (framewise.shape[0], framewise.shape[2], framewise.shape[1]))
            keypoints = torch.reshape(keypoints, (keypoints.shape[0], keypoints.shape[2], keypoints.shape[1]))

            framewise_spatial_attn_out, _ = self.spatial_attn(framewise, framewise, framewise)
            keypoints_spatial_attn_out, _ = self.spatial_attn(keypoints, keypoints, keypoints)

            framewise_spatial_attn_out = torch.reshape(framewise_spatial_attn_out,
                                                       (framewise.shape[0], framewise.shape[2], framewise.shape[1]))
            keypoints_spatial_attn_out = torch.reshape(keypoints_spatial_attn_out,
                                                       (keypoints.shape[0], keypoints.shape[2], keypoints.shape[1]))

            conv1d_outputs = self.conv1d_type_1_block1(framewise_spatial_attn_out, len_x)
            conv1d_outputs_key = self.conv1d_type_1_block1(keypoints_spatial_attn_out, len_x)
        else:
            conv1d_outputs = self.conv1d(framewise, len_x)
            conv1d_outputs_key = self.conv1d(keypoints, len_x)

        if self.use_temporal_attn:
            # conv1d_outputs = self.conv1d_type_1_block1(framewise, len_x)
            # conv1d_outputs_key = self.conv1d_type_1_block1(keypoints, len_x)

            # x: T, B, C
            block_1 = conv1d_outputs['visual_feat']
            # x_key: T, B, C
            block_1_key = conv1d_outputs_key['visual_feat']

            block_1 = self.temporal_attn(block_1, block_1, block_1)
            block_1_key = self.temporal_attn(block_1_key, block_1_key, block_1_key)

            block_1 = torch.reshape(block_1, (block_1.shape[1], block_1.shape[2], block_1.shape[0]))
            block_1_key = torch.reshape(block_1_key, (block_1_key.shape[1], block_1_key.shape[2], block_1_key.shape[0]))

            lgt = conv1d_outputs['feat_len']

            block_2 = self.conv1d_type_1_block2(block_1, lgt)
            block_2_key = self.conv1d_type_1_block2(block_1_key, lgt)

            # x: T, B, C
            x = block_2['visual_feat']
            # x_key: T, B, C
            x_key = block_2_key['visual_feat']

            x = self.temporal_attn(x, x, x)
            x_key = self.temporal_attn(x_key, x_key, x_key)

            lgt = block_2['feat_len']
        else:
            conv1d_outputs = self.conv1d(framewise, len_x)
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']

        # concat
        x_cat = torch.cat([x, x_key], 2)

        tm_outputs = self.temporal_model(x_cat, lgt)

        outputs = self.classifier(tm_outputs['predictions'])

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        # key_pred = None if self.training \
        #     else self.decoder.decode(conv1d_outputs_key['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'KeyCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["key_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()

            elif k == 'SeqFullFrameCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["seq_ff_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()

            elif k == 'SeqKeyCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["seq_key_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()

            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)

            elif k == 'DistKey':
                loss += weight * self.loss['distillation'](ret_dict["key_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss

# A helper function for producing N identical layers
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Self-Attention mechanism
def ScaledDotProductAttention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    # src_mask=(batch, 1, 1, max_seq) #NOTE: this is like the tutorials but it is weird!
    # trg_mask = (batch, 1, max_seq, max_seq)
    # score=(batch, n_heads, Seq, Seq)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)

    return output  # (Batch, n_heads, Seq, d_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.3):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()

        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        # d_k = dim of key for one head
        self.d_k = n_units // n_heads

        # This requires the number of n_heads to evenly divide n_units.
        # NOTE: nb of n_units (hidden_size) must be a multiple of 6 (n_heads)
        assert n_units % n_heads == 0
        # n_units represent total of units for all the heads

        self.n_units = n_units
        self.n_heads = n_heads

        self.linears = clones(nn.Linear(n_units, n_units), 4)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = ScaledDotProductAttention(query, key, value, mask=mask,
                                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_heads * self.d_k)

        z = self.linears[-1](x)

        # (batch_size, seq_len, self.n_units)
        return z
