# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.twins import twins_svt_small,SELayer

from timm.models.resnet import Bottleneck,tv_resnet50




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name == 'twins_svt_small':
            self.base = twins_svt_small(pretrained=True)
            self.in_planes = 512
            # resnet = tv_resnet50(pretrained=True)
            # resnetPost1 = resnet.layer3
            # resnetPost2 = resnet.layer4
            # resnetPost1.load_state_dict(resnet.layer3.state_dict())
            # resnetPost2.load_state_dict(resnet.layer4.state_dict())
            # self.globalBranch = copy.deepcopy(resnetPost1)
            # self.localBranch = nn.Sequential(copy.deepcopy(resnetPost1),
            #                                  copy.deepcopy(resnetPost2))



        if pretrain_choice == 'imagenet' and model_name != 'twins_svt_small':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.avggap = nn.AdaptiveAvgPool2d(1)
        self.maxgap = nn.AdaptiveMaxPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        # self.gap = nn.AdaptiveAvgPool2d(1)



        # self.f_1_avggap = nn.AdaptiveAvgPool2d(1)
        # self.f_1_maxgap = nn.AdaptiveMaxPool2d(1)
        # self.f_2_avggap = nn.AdaptiveAvgPool2d(1)
        # self.f_2_maxgap = nn.AdaptiveMaxPool2d(1)
        # self.f_3_avggap = nn.AdaptiveAvgPool2d(1)
        # self.f_3_maxgap = nn.AdaptiveMaxPool2d(1)
        # self.f_4_avggap = nn.AdaptiveAvgPool2d(1)
        # self.f_4_maxgap = nn.AdaptiveMaxPool2d(1)

        # reductiong = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        # reductionl = nn.Sequential(nn.Conv2d(2048, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU())
        #
        # # self._init_reduction(reductiong)
        # self._init_reduction(reductionl)
        # # reductiong.apply(_init_reduction)
        # # reductionl.apply(_init_reduction)
        # # self._init_reduction(reduction)
        # # self.reduction_0 = copy.deepcopy(reductiong)
        # self.reduction_1 = copy.deepcopy(reductionl)
        # self.reduction_2 = copy.deepcopy(reductionl)
        # self.reduction_3 = copy.deepcopy(reductionl)
        # self.reduction_4 = copy.deepcopy(reductionl)
        # se = SELayer(channel=512)
        self.b1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            # se,
            nn.ReLU()
        )
        self._init_reduction(self.b1)
        # self.b2 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(512),
        #     # se,
        #     nn.ReLU()
        # )
        # self._init_reduction(self.b2)
        # self.b3 = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(256),
        #     # se,
        #     nn.ReLU()
        # )
        # self._init_reduction(self.b3)
        # self.b4 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(128),
        #     # se,
        #     nn.ReLU()
        # )





        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.in_planes = self.in_planes
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier_f1 = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier_f2 = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier_f3 = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier_f4 = nn.Linear(self.in_planes, self.num_classes)

            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            # self.in_planes = self.in_planes//8
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

            self.bottleneck_f1 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_f1.bias.requires_grad_(False)  # no shift
            self.classifier_f1 = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck_f1.apply(weights_init_kaiming)
            self.classifier_f1.apply(weights_init_classifier)

            self.bottleneck_f2 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_f2.bias.requires_grad_(False)  # no shift
            self.classifier_f2 = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck_f2.apply(weights_init_kaiming)
            self.classifier_f2.apply(weights_init_classifier)
            # #
            # self.bottleneck_f3 = nn.BatchNorm1d(self.in_planes)
            # self.bottleneck_f3.bias.requires_grad_(False)  # no shift
            # self.classifier_f3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            #
            # self.bottleneck_f3.apply(weights_init_kaiming)
            # self.classifier_f3.apply(weights_init_classifier)
            #
            # self.bottleneck_f4 = nn.BatchNorm1d(self.in_planes)
            # self.bottleneck_f4.bias.requires_grad_(False)  # no shift
            # self.classifier_f4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            #
            # self.bottleneck_f4.apply(weights_init_kaiming)
            # self.classifier_f4.apply(weights_init_classifier)

    ##single
    # def forward(self, x):
    #     feat_map = self.base(x)
    #     # print(feat_map.shape)
    #     # global_feat = self.gap(feat_map)  # (b, 2048, 1, 1)
    #     # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
    #     global_feat = feat_map.view(feat_map.shape[0], -1)  # flatten to (bs, 2048)
    #
    #     if self.neck == 'no':
    #         feat = global_feat
    #     elif self.neck == 'bnneck':
    #         feat = self.bottleneck(global_feat)  # normalize for angular softmax
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, global_feat  # global feature for triplet loss
    #     else:
    #         if self.neck_feat == 'after':
    #             # print("Test with feature after BN")
    #             return feat
    #         else:
    #             # print("Test with feature before BN")
    #             return global_feat

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)


    # ##multi
    # def forward(self, x):
    #     feat_map = self.base(x)    ###b,512,7,7
    #     global_feat = self.maxgap(feat_map).squeeze(dim=3).squeeze(dim=2)  ##b,1024
    #     # global_feat = self.reduction_0(global_feat)
    #
    #
    #     ###local branch
    #     local_f1_map = feat_map[:,:,0:3,:]  ##b,512,1,2
    #     local_f2_map = feat_map[:,:,2:5,:]  ##b,512,1,2
    #     local_f3_map = feat_map[:,:,4:7,:]  ##b,512,1,2
    #
    #
    #     local_f1_final = self.avggap(local_f1_map).squeeze(dim=3).squeeze(dim=2)  ## b,512
    #     local_f2_final = self.avggap(local_f2_map).squeeze(dim=3).squeeze(dim=2)
    #     local_f3_final = self.avggap(local_f3_map).squeeze(dim=3).squeeze(dim=2)
    #
    #
    #     if self.neck == 'no':
    #         feat = global_feat
    #         feat_f1 = local_f1_final
    #         feat_f2 = local_f2_final
    #         feat_f3 = local_f3_final
    #
    #
    #     elif self.neck == 'bnneck':
    #         feat = self.bottleneck(global_feat)  # normalize for angular softmax
    #
    #         feat_f1 = self.bottleneck_f1(local_f1_final)
    #         feat_f2 = self.bottleneck_f2(local_f2_final)
    #         feat_f3 = self.bottleneck_f3(local_f3_final)
    #
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #         cls_f1 = self.classifier_f1(feat_f1)
    #         cls_f2 = self.classifier_f2(feat_f2)
    #         cls_f3 = self.classifier_f3(feat_f3)
    #
    #
    #         # return cls_score, global_feat  # global feature for triplet loss
    #         return [cls_score,cls_f1,cls_f2,cls_f3], [global_feat,local_f1_final,local_f2_final,local_f3_final] # global feature for triplet loss
    #     else:
    #         if self.neck_feat == 'after':
    #             # print("Test with feature after BN")
    #             final_feat = torch.cat([feat,feat_f1,feat_f2,feat_f3],dim=1)   #b,3072
    #             return final_feat
    #         else:
    #             # print("Test with feature before BN")
    #             final_feat = torch.cat([global_feat,local_f1_final,local_f2_final,local_f3_final],dim=1)
    #             return final_feat

    # #multi block3
    # def forward(self, x):
    #     feat_mapBlock3 , feat_map = self.base(x)  ###b,512,7,7
    #     global_feat = self.avggap(feat_map).squeeze(dim=3).squeeze(dim=2)  ##b,1024
    #
    #
    #     ###local branch
    #     local_f1_map = feat_mapBlock3[:, 0:64, :, :]  ##b,64,14,14
    #     local_f2_map = feat_mapBlock3[:, 64:128, :, :]  ##b,64,14,14
    #     local_f3_map = feat_mapBlock3[:, 128:192, :, :]  ##b,64,14,14
    #     local_f4_map = feat_mapBlock3[:, 192:256, :, :]  ##b,64,14,14
    #
    #     local_f1_final = self.avggap(self.b1(local_f1_map)).squeeze(dim=3).squeeze(dim=2)  ## b,512
    #     local_f2_final = self.avggap(self.b2(local_f2_map)).squeeze(dim=3).squeeze(dim=2)
    #     local_f3_final = self.avggap(self.b3(local_f3_map)).squeeze(dim=3).squeeze(dim=2)
    #     local_f4_final = self.avggap(self.b4(local_f4_map)).squeeze(dim=3).squeeze(dim=2)
    #
    #     feat_local_final = torch.cat([local_f1_final, local_f2_final, local_f3_final, local_f4_final], dim=1)
    #
    #     if self.neck == 'no':
    #         feat = global_feat
    #         feat_local = feat_local_final
    #
    #     elif self.neck == 'bnneck':
    #         feat = self.bottleneck(global_feat)  # normalize for angular softmax
    #         feat_local = self.bottleneck_f1(feat_local_final)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #         cls_f1 = self.classifier_f1(feat_local)
    #
    #         # return cls_score, global_feat  # global feature for triplet loss
    #         return [cls_score, cls_f1], [global_feat, feat_local_final]  # global feature for triplet loss
    #     else:
    #         if self.neck_feat == 'after':
    #             # print("Test with feature after BN")
    #             final_feat = torch.cat([feat, feat_local], dim=1)  # b,3072
    #             return final_feat
    #         else:
    #             # print("Test with feature before BN")
    #             final_feat = torch.cat([global_feat, feat_local], dim=1)
    #             return final_feat

    # multi block3 block2
    def forward(self, x):
        feat_mapBlock3, feat_map = self.base(x)  ###b,512,7,7

        global_feat = self.avggap(feat_map).squeeze(dim=3).squeeze(dim=2)  ##b,512

        global_feat_blk3 = self.b1(feat_mapBlock3)  ## b,512,12,12

        ###local branch
        local_f1_map = global_feat_blk3[:, :, 0:6, :]  ##b,512,6,12
        local_f2_map = global_feat_blk3[:, :, 6:12, :]  ##b,512,6,12



        # local_f1_final = self.avggap(self.b1(local_f1_map)).squeeze(dim=3).squeeze(dim=2)  ## b,512
        # local_f2_final = self.avggap(self.b2(local_f2_map)).squeeze(dim=3).squeeze(dim=2)
        local_f1_final = self.avggap(local_f1_map).squeeze(dim=3).squeeze(dim=2)  ## b,512
        local_f2_final = self.avggap(local_f2_map).squeeze(dim=3).squeeze(dim=2)


        # feat_local_final = torch.cat([local_f1_final,local_f2_final],dim=1)

        if self.neck == 'no':
            feat = global_feat
            # feat_local = feat_local_final
            feat_local1 = local_f1_final
            feat_local2 = local_f2_final

        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            feat_local1 = self.bottleneck_f1(local_f1_final)
            feat_local2 = self.bottleneck_f2(local_f2_final)

        if self.training:
            cls_score = self.classifier(feat)
            cls_f1 = self.classifier_f1(feat_local1)
            cls_f2 = self.classifier_f2(feat_local2)

            # return cls_score, global_feat  # global feature for triplet loss
            return [cls_score, cls_f1,cls_f2], [global_feat, local_f1_final,local_f2_final]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                final_feat = torch.cat([feat, feat_local1,feat_local2], dim=1)  # b,3072
                return final_feat
            else:
                # print("Test with feature before BN")
                final_feat = torch.cat([global_feat, local_f1_final,local_f2_final], dim=1)
                return final_feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
