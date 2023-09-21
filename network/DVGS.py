import timm
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from model.Resnet import Resnet
from model.ViT import ViT
from utils import matrix_visualize


class DVGS(nn.Module):
    def __init__(self, args):
        super(DVGS, self).__init__()

        self.args = args

        ##### pretrained model version, default is pretrained on ImageNet21K
        # Others variants of ViT can be used as well
        '''
        1 --- 'vit_small_patch16_224'
        2 --- 'vit_base_patch16_224'
        3 --- 'vit_large_patch16_224',
        4 --- 'vit_large_patch32_224'
        5 --- 'vit_deit_base_patch16_224'
        6 --- 'deit_base_distilled_patch16_224',
        '''
        model_name = "vit_large_patch16_224_in21k"
        self.v_encoder = ViT(model_name, pretrained=True)
        attr_num = args.attr_num
        v_embedding_dim = 1024
        region_num = 196

        #### dynamic region selection
        self.DRS = DynamicRegionSelection(v_embedding_dim, region_num, bias=False)

        #### dynamic attribute selection
        self.DAS = DynamicRegionSelection(v_embedding_dim, attr_num, bias=True)

        #### attribute classifier
        self.attribute_classifier = nn.Linear(v_embedding_dim, attr_num, bias=False)

    def forward(self, x):
        args = self.args

        region_feature, global_feature = self.v_encoder(x)

        refined_region_feature = self.DRS(region_feature, global_feature)

        region_predicted_prototype = self.attribute_classifier(refined_region_feature)
        global_predicted_prototype = self.attribute_classifier(global_feature)

        class_prototype = args.attr_mats
        refined_prototype = self.DAS(refined_region_feature, global_feature, class_prototype)

        package = {'region_predicted_prototype': region_predicted_prototype,
                   'global_predicted_prototype': global_predicted_prototype,
                   'refined_prototype': refined_prototype}

        return package


class DynamicRegionSelection(nn.Module):
    def __init__(self, v_embedding_dim, region_num, bias=False):
        super(DynamicRegionSelection, self).__init__()
        self.fc = nn.Linear(v_embedding_dim, region_num, bias=bias)

    def forward(self, region_feature, global_feature):
        region_weight = self.fc(global_feature)
        refined_region_feature = torch.einsum('bir, bi -> bir', region_feature, region_weight)
        refined_region_feature = refined_region_feature.mean(dim=1)
        return refined_region_feature


class DynamicAttributeSelection(nn.Module):
    def __init__(self, v_embedding, attr_num, bias=True):
        super(DynamicAttributeSelection, self).__init__()
        self.relu = nn.ReLU
        self.softmax = nn.Softmax
        self.fc1 = nn.Linear(v_embedding, v_embedding // 2, bias=bias)
        self.fc2 = nn.Linear(v_embedding // 2, attr_num, bias=bias)

    def forward(self, refined_region_feature, global_feature, class_prototype):
        fused_feature = refined_region_feature.mean(dim=1) + global_feature
        attr_weight = self.fc(fused_feature)
        attr_weight = self.softmax(attr_weight)
        attr_weight = attr_weight.unsqueeze(dim=1).repeat(1, class_prototype.shape[0], 1)
        class_prototype = class_prototype.unsqueeze(dim=0).repeat(global_feature.shape[0], 1, 1)
        refined_prototype = torch.einsum('bir, bir -> bir', attr_weight, class_prototype)
        return refined_prototype
