from __future__ import absolute_import
from __future__ import division

import torch
import copy
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet50, Bottleneck
from .hacnn import SoftBlock, SoftHardBlock
import torchvision


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss_type='xent', **kwargs):
        super(ResNet50, self).__init__()
        self.loss_type = loss_type
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if self.loss_type == 'xent':
            if self.training:
                y = self.classifier(f)
                return [y]
            else:
                feat = torch.div(f, f.norm(dim=1, keepdim=True))
                return feat
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2', 'xent_triplet_sqrt', 'xent_triplet_squa']:
            feat = torch.div(f, f.norm(dim=1, keepdim=True))
            if self.training:
                y = self.classifier(f)
                return [y], feat
            else:
                return feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class MGN(nn.Module):
    def __init__(self, num_classes, loss_type='xent', **kwargs):
        super(MGN, self).__init__()
        self.loss_type = loss_type
        self.dimension_branch = 512
        # self.dimension_branch = 1024
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            # nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # res_conv2
            resnet.layer2,      # res_conv3
            resnet.layer3[0],   # res_conv4_1
        )

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_g_conv5 = resnet.layer4
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(16, 16))

        reduction_512 = nn.Sequential(nn.Conv2d(2048, self.dimension_branch, 1, bias=False),
                                      nn.BatchNorm2d(self.dimension_branch), nn.ReLU())
        self.reduction_1 = copy.deepcopy(reduction_512)
        self.reduction_2 = copy.deepcopy(reduction_512)

        self.fc_id_512_1 = nn.Linear(self.dimension_branch, num_classes)
        self.fc_id_512_2 = nn.Linear(self.dimension_branch, num_classes)
        # self.fc_id_512_1 = nn.Linear(2048, num_classes)
        # self.fc_id_512_2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.p1(x)
        p2 = self.p2(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)

        fg_p1 = self.reduction_1(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_2(zg_p2).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_512_1(fg_p1)
        l_p2 = self.fc_id_512_2(fg_p2)
        # l_p1 = self.fc_id_512_1(zg_p1.squeeze(dim=3).squeeze(dim=2))
        # l_p2 = self.fc_id_512_2(zg_p2.squeeze(dim=3).squeeze(dim=2))


        if self.loss_type in ['xent']:
            if self.training:
                feat_clfy = [l_p1, l_p2]
                return feat_clfy
            else:
                # feat_embed = torch.cat([fg_p1, fg_p2], dim=1)
                # feat_embed = torch.div(feat_embed, feat_embed.norm(dim=1, keepdim=True))
                # return feat_embed

                # fg_p1 = torch.div(fg_p1, fg_p1.norm(dim=1, keepdim=True))
                # fg_p2 = torch.div(fg_p2, fg_p2.norm(dim=1, keepdim=True))
                feat_global = torch.cat([fg_p1, fg_p2], dim=1)
                feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
                return feat_global
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2', 'xent_triplet_sqrt', 'xent_triplet_squa']:
            # # feat_clfy = torch.cat([l_p1, l_p2], dim=0)
            # feat_clfy = [l_p1, l_p2]
            # # feat_clfy = l_p1
            # feat_global = torch.cat([fg_p1, fg_p2], dim=1)
            # # feat_global = fg_p1
            # feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            # # feat_local = torch.cat([fz_p1, fz_p2, fz_p3, fz_p4], dim=1)
            # # feat_local = torch.div(feat_local, feat_local.norm(dim=1, keepdim=True))
            # if self.training:
            #     return feat_clfy, feat_global
            # else:
            #     return feat_global

            # feat_clfy = [l_p1, l_p2]
            # fg_p1 = torch.div(fg_p1, fg_p1.norm(dim=1, keepdim=True))
            # fg_p2 = torch.div(fg_p2, fg_p2.norm(dim=1, keepdim=True))
            # feat_global = [fg_p1, fg_p2]
            # if self.training:
            #     return feat_clfy, feat_global
            # else:
            #     feat_global = torch.cat([fg_p1, fg_p2], dim=1)
            #     return feat_global

            # feat_clfy = [l_p1, l_p2]
            # feat_global = [fg_p1, fg_p2]
            # if self.training:
            #     return feat_clfy, feat_global
            # else:
            #     # fg_p1 = torch.div(fg_p1, fg_p1.norm(dim=1, keepdim=True))
            #     # fg_p2 = torch.div(fg_p2, fg_p2.norm(dim=1, keepdim=True))
            #     feat_global = torch.cat([fg_p1, fg_p2], dim=1)
            #     feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            #     return feat_global

            feat_clfy = [l_p1, l_p2]
            feat_global = torch.cat([fg_p1, fg_p2], dim=1)
            feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            if self.training:
                # fg_p1 = torch.div(fg_p1, fg_p1.norm(dim=1, keepdim=True))
                # fg_p2 = torch.div(fg_p2, fg_p2.norm(dim=1, keepdim=True))
                # feat_global = [fg_p1, fg_p2]
                return feat_clfy, feat_global
            else:
                # feat_global = torch.cat([fg_p1, fg_p2], dim=1)
                # feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
                return feat_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class OriginMGN(nn.Module):
    """
    @ARTICLE{2018arXiv180401438W,
        author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
        title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1804.01438},
        primaryClass = "cs.CV",
        keywords = {Computer Science - Computer Vision and Pattern Recognition},
        year = 2018,
        month = apr,
        adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    """
    def __init__(self, num_classes, loss_type='xent', **kwargs):
        super(OriginMGN, self).__init__()
        self.loss_type = loss_type
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # res_conv2
            resnet.layer2,      # res_conv3
            resnet.layer3[0],   # res_conv4_1
        )

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        # res_conv5 global
        res_g_conv5 = resnet.layer4
        # res_conv5 part
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        # mgn part-1 global
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        # mgn part-2
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        # mgn part-3
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        # global max pooling
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        # conv1 reduce
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # fc softmax loss
        self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_1 = nn.Linear(2048, num_classes)
        self.fc_id_2048_2 = nn.Linear(2048, num_classes)
        self.fc_id_256_1_0 = nn.Linear(256, num_classes)
        self.fc_id_256_1_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_0 = nn.Linear(256, num_classes)
        self.fc_id_256_2_1 = nn.Linear(256, num_classes)
        self.fc_id_256_2_2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)  # z_g^G
        zg_p2 = self.maxpool_zg_p2(p2)  # z_g^P2
        zg_p3 = self.maxpool_zg_p3(p3)  # z_g^P3

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]  # z_p0^P2
        z1_p2 = zp2[:, :, 1:2, :]  # z_p1^P2

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]  # z_p0^P3
        z1_p3 = zp3[:, :, 1:2, :]  # z_p1^P3
        z2_p3 = zp3[:, :, 2:3, :]  # z_p2^P3

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)  # f_g^G, L_triplet^G
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)  # f_g^P2, L_triplet^P2
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)  # f_g^P3, L_triplet^P3
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)  # f_p0^P2
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)  # f_p1^P2
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)  # f_p0^P3
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)  # f_p1^P3
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)  # f_p2^P3

        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))  # L_softmax^G
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P2
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))  # L_softmax^P3
        l0_p2 = self.fc_id_256_1_0(f0_p2)  # L_softmax0^P2
        l1_p2 = self.fc_id_256_1_1(f1_p2)  # L_softmax1^P2
        l0_p3 = self.fc_id_256_2_0(f0_p3)  # L_softmax0^P3
        l1_p3 = self.fc_id_256_2_1(f1_p3)  # L_softmax1^P3
        l2_p3 = self.fc_id_256_2_2(f2_p3)  # L_softmax2^P3


        if self.loss_type in ['xent_triplet', 'xent_tripletv2', 'xent_triplet_sqrt', 'xent_triplet_squa']:
            if self.training:
                feat_clfy = [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]
                feat = torch.cat([fg_p1, fg_p2, fg_p3], dim=1)
                feat = torch.div(feat, feat.norm(dim=1, keepdim=True))
                return feat_clfy, feat
            else:
                feat = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
                feat = torch.div(feat, feat.norm(dim=1, keepdim=True))
                return feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class MGNB4(nn.Module):
    def __init__(self, num_classes, loss_type='xent', **kwargs):
        super(MGNB4, self).__init__()
        self.loss_type = loss_type
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # res_conv2
            resnet.layer2,      # res_conv3
            resnet.layer3[0],   # res_conv4_1
        )

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_conv5 = resnet.layer4

        self.b1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))
        self.b2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))
        self.b3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))
        self.b4 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))

        self.maxpool_b1 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_b2 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_b3 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_b4 = nn.MaxPool2d(kernel_size=(8, 8))

        reduction_512 = nn.Sequential(nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())
        self.reduction_1 = copy.deepcopy(reduction_512)
        self.reduction_2 = copy.deepcopy(reduction_512)
        self.reduction_3 = copy.deepcopy(reduction_512)
        self.reduction_4 = copy.deepcopy(reduction_512)

        self.fc_id_512_1 = nn.Linear(512, num_classes)
        self.fc_id_512_2 = nn.Linear(512, num_classes)
        self.fc_id_512_3 = nn.Linear(512, num_classes)
        self.fc_id_512_4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        pb1 = self.maxpool_b1(b1)
        pb2 = self.maxpool_b2(b2)
        pb3 = self.maxpool_b3(b3)
        pb4 = self.maxpool_b4(b4)

        f_b1 = self.reduction_1(pb1).squeeze(dim=3).squeeze(dim=2)
        f_b2 = self.reduction_2(pb2).squeeze(dim=3).squeeze(dim=2)
        f_b3 = self.reduction_3(pb3).squeeze(dim=3).squeeze(dim=2)
        f_b4 = self.reduction_4(pb4).squeeze(dim=3).squeeze(dim=2)

        cf_b1 = self.fc_id_512_1(f_b1)
        cf_b2 = self.fc_id_512_2(f_b2)
        cf_b3 = self.fc_id_512_3(f_b3)
        cf_b4 = self.fc_id_512_4(f_b4)


        if self.loss_type in ['xent']:
            if self.training:
                feat_clfy = [cf_b1, cf_b2, cf_b3, cf_b4]
                return feat_clfy
            else:
                feat_global = torch.cat([f_b1, f_b2, f_b3, f_b4], dim=1)
                feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
                return feat_global
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2']:
            feat_clfy = [cf_b1, cf_b2, cf_b3, cf_b4]
            feat_global = torch.cat([f_b1, f_b2, f_b3, f_b4], dim=1)
            feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            if self.training:
                return feat_clfy, feat_global
            else:
                return feat_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class MGNB2(nn.Module):
    def __init__(self, num_classes, loss_type='xent', **kwargs):
        super(MGNB2, self).__init__()
        self.loss_type = loss_type
        self.dimension_branch = 1024
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # res_conv2
            resnet.layer2,      # res_conv3
            resnet.layer3[0],   # res_conv4_1
        )

        # res_conv4x
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_conv5 = resnet.layer4

        self.b1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))
        self.b2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_conv5))

        self.maxpool_b1 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_b2 = nn.MaxPool2d(kernel_size=(8, 8))

        reduction_512 = nn.Sequential(nn.Conv2d(2048, self.dimension_branch, 1, bias=False),
                                      nn.BatchNorm2d(self.dimension_branch), nn.ReLU())
        self.reduction_1 = copy.deepcopy(reduction_512)
        self.reduction_2 = copy.deepcopy(reduction_512)

        self.fc_id_512_1 = nn.Linear(self.dimension_branch, num_classes)
        self.fc_id_512_2 = nn.Linear(self.dimension_branch, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        b1 = self.b1(x)
        b2 = self.b2(x)

        pb1 = self.maxpool_b1(b1)
        pb2 = self.maxpool_b2(b2)

        f_b1 = self.reduction_1(pb1).squeeze(dim=3).squeeze(dim=2)
        f_b2 = self.reduction_2(pb2).squeeze(dim=3).squeeze(dim=2)

        cf_b1 = self.fc_id_512_1(f_b1)
        cf_b2 = self.fc_id_512_2(f_b2)


        if self.loss_type in ['xent']:
            if self.training:
                feat_clfy = [cf_b1, cf_b2]
                return feat_clfy
            else:
                feat_global = torch.cat([f_b1, f_b2], dim=1)
                feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
                return feat_global
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2', 'xent_triplet_sqrt', 'xent_triplet_squa']:
            feat_clfy = [cf_b1, cf_b2]
            feat_global = torch.cat([f_b1, f_b2], dim=1)
            feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            if self.training:
                return feat_clfy, feat_global
            else:
                return feat_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class ResSoAttn(nn.Module):
    def __init__(self, num_classes, loss_type='xent', nchannels=[128, 256, 384], branch_feat_dim=682, **kwargs):
        super(ResSoAttn, self).__init__()
        self.loss_type = loss_type
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # res_conv2
            resnet.layer2,      # res_conv3
        )

        self.habk1 = nn.Sequential(SoftBlock(nchannels=nchannels, input_channel=512, feat_dim=branch_feat_dim),
                                   nn.Dropout(p=0.5, inplace=True))
        self.habk2 = nn.Sequential(SoftBlock(nchannels=nchannels, input_channel=512, feat_dim=branch_feat_dim),
                                   nn.Dropout(p=0.5, inplace=True))
        self.habk3 = nn.Sequential(SoftBlock(nchannels=nchannels, input_channel=512, feat_dim=branch_feat_dim),
                                   nn.Dropout(p=0.5, inplace=True))

        self.fc_id_1 = nn.Linear(branch_feat_dim, num_classes)
        self.fc_id_2 = nn.Linear(branch_feat_dim, num_classes)
        self.fc_id_3 = nn.Linear(branch_feat_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        f_b1 = self.habk1(x)
        f_b2 = self.habk2(x)
        f_b3 = self.habk3(x)

        cf_b1 = self.fc_id_1(f_b1)
        cf_b2 = self.fc_id_2(f_b2)
        cf_b3 = self.fc_id_3(f_b3)


        if self.loss_type in ['xent']:
            if self.training:
                feat_clfy = [cf_b1, cf_b2, cf_b3]
                return feat_clfy
            else:
                feat_global = torch.cat([f_b1, f_b2, f_b3], dim=1)
                feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
                return feat_global
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2']:
            feat_clfy = [cf_b1, cf_b2, cf_b3]
            feat_global = torch.cat([f_b1, f_b2, f_b3], dim=1)
            feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            if self.training:
                return feat_clfy, feat_global
            else:
                return feat_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class ResSoHaAttn(nn.Module):
    def __init__(self, num_classes, loss_type='xent', nchannels=[128, 256, 384], branch_feat_dim=682, **kwargs):
        super(ResSoHaAttn, self).__init__()
        self.loss_type = loss_type
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,      # res_conv2
            resnet.layer2,      # res_conv3
        )

        self.habk1 = SoftHardBlock(nchannels=nchannels, input_channel=512, feat_dim=branch_feat_dim)
        self.habk2 = SoftHardBlock(nchannels=nchannels, input_channel=512, feat_dim=branch_feat_dim)
        self.habk3 = SoftHardBlock(nchannels=nchannels, input_channel=512, feat_dim=branch_feat_dim)

        self.fc_id_1 = nn.Linear(branch_feat_dim, num_classes)
        self.fc_id_2 = nn.Linear(branch_feat_dim, num_classes)
        self.fc_id_3 = nn.Linear(branch_feat_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        fg_b1, fl_b1 = self.habk1(x)
        fg_b2, fl_b2 = self.habk2(x)
        fg_b3, fl_b3 = self.habk3(x)
        f_b1 = torch.cat([fg_b1, fl_b1], dim=1)
        f_b2 = torch.cat([fg_b2, fl_b2], dim=1)
        f_b3 = torch.cat([fg_b3, fl_b3], dim=1)

        cf_b1 = self.fc_id_1(f_b1)
        cf_b2 = self.fc_id_2(f_b2)
        cf_b3 = self.fc_id_3(f_b3)


        if self.loss_type in ['xent']:
            if self.training:
                feat_clfy = [cf_b1, cf_b2, cf_b3]
                return feat_clfy
            else:
                feat = torch.cat([f_b1, f_b2, f_b3], dim=1)
                feat = torch.div(feat, feat.norm(dim=1, keepdim=True))
                return feat
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2']:
            feat_clfy = [cf_b1, cf_b2, cf_b3]
            # feat_global = torch.cat([fg_b1, fg_b2, fg_b3], dim=1)
            # feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            feat = torch.cat([f_b1, f_b2, f_b3], dim=1)
            feat = torch.div(feat, feat.norm(dim=1, keepdim=True))
            if self.training:
                # return feat_clfy, feat_global
                return feat_clfy, feat
            else:
                # feat = torch.cat([f_b1, f_b2, f_b3], dim=1)
                # feat = torch.div(feat, feat.norm(dim=1, keepdim=True))
                return feat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


class Resv2SoAttn(nn.Module):
    def __init__(self, num_classes, loss_type='xent', nchannels=[256, 384, 512], branch_feat_dim=682, **kwargs):
        super(Resv2SoAttn, self).__init__()
        self.loss_type = loss_type
        self.inplanes = 16
        self.layer1 = self.make_layer(Bottleneck, 16, 3, stride=1)
        self.layer2 = self.make_layer(Bottleneck, 32, 4, stride=2)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.layer1,
            self.layer2,
        )

        self.habk1 = nn.Sequential(SoftBlock(nchannels=nchannels, input_channel=128, feat_dim=branch_feat_dim),
                                   nn.Dropout(p=0.5, inplace=True))
        self.habk2 = nn.Sequential(SoftBlock(nchannels=nchannels, input_channel=128, feat_dim=branch_feat_dim),
                                   nn.Dropout(p=0.5, inplace=True))
        self.habk3 = nn.Sequential(SoftBlock(nchannels=nchannels, input_channel=128, feat_dim=branch_feat_dim),
                                   nn.Dropout(p=0.5, inplace=True))

        self.fc_id_1 = nn.Linear(branch_feat_dim, num_classes)
        self.fc_id_2 = nn.Linear(branch_feat_dim, num_classes)
        self.fc_id_3 = nn.Linear(branch_feat_dim, num_classes)

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        f_b1 = self.habk1(x)
        f_b2 = self.habk2(x)
        f_b3 = self.habk3(x)

        cf_b1 = self.fc_id_1(f_b1)
        cf_b2 = self.fc_id_2(f_b2)
        cf_b3 = self.fc_id_3(f_b3)


        if self.loss_type in ['xent']:
            if self.training:
                feat_clfy = [cf_b1, cf_b2, cf_b3]
                return feat_clfy
            else:
                feat_global = torch.cat([f_b1, f_b2, f_b3], dim=1)
                feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
                return feat_global
        elif self.loss_type in ['xent_triplet', 'xent_tripletv2']:
            feat_clfy = [cf_b1, cf_b2, cf_b3]
            feat_global = torch.cat([f_b1, f_b2, f_b3], dim=1)
            feat_global = torch.div(feat_global, feat_global.norm(dim=1, keepdim=True))
            if self.training:
                return feat_clfy, feat_global
            else:
                return feat_global
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss_type))


