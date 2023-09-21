import timm
import torch
from torch import nn


class ViT(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224_in21k", pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # Change the head depending of the dataset used
        self.vit.head = nn.Identity()

    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        global_feature = x[:, 0]
        region_feature = x[:, 1:]

        return region_feature, global_feature
