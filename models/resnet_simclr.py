import torch.nn as nn
from models.transformer import SpectralModel
from torchvision import models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, args):
        super(ResNetSimCLR, self).__init__()
        self.model_dict = {"transformer": SpectralModel(in_channels=1, wav_min=args.wav_min, wav_max=args.wav_max,
                                                         emb_size=args.emb_size, nhead=args.nhead, num_layers=args.num_layers,
                                                         dim_feedforward=args.dim_feedforward, dropout=args.dropout, use_mean_pool=args.use_mean_pool,
                                                         output_dim=out_dim),
                            "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x, *args, **kwargs):
        return self.backbone(x, *args, **kwargs)
