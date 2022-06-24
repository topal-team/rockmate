import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from .utils import *

## Copied and adapted from the torchvision package from conda
## torchvision-cpu           0.3.0             py36_cuNone_1

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Sequential):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.add_module('features', features)
        self.add_module('avgpool', nn.AdaptiveAvgPool2d((7, 7)))
        self.add_module('flatten', Flatten())
        self.add_module('classifier', nn.Sequential(
            ReLUatEnd(nn.Linear(512 * 7 * 7, 4096)),
            nn.Dropout(),
            ReLUatEnd(nn.Linear(4096, 4096)),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ))
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if batch_norm:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), BatchNorm2dAndReLU(v)]
            else:
                layers += [ReLUatEnd(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))]
            in_channels = v
    return nn.Sequential(*layers)

# Keep track of which indices were chaged in the make_layers Sequential modules
# Compared to the original pretrained version
def changed_indices(cfg, batch_norm = False):
    removed = []
    updated = []
    current_index = 0
    for v in cfg:
        if v == 'M':
            current_index += 1
        else:
            if batch_norm:
                removed.append(current_index + 2)
                current_index += 3
            else:
                updated.append(current_index)
                removed.append(current_index + 1)
                current_index += 2
    return updated, removed

# Renumber keys in a state_dict corresponding to a nn.Sequential module
# To ensure that we can use the pretrained version
def renumber_keys(prefix, scheme, state_dict):
    for key in list(state_dict.keys()):
        if key.startswith(prefix):
            try:
                start = key.index('.', len(prefix))
                end   = key.index('.', start+1)
                number = int(key[start+1:end])
            except ValueError as e:
                continue
            new_number = scheme(number)
            if new_number != number:
                new_key = key[:start+1] + str(new_number) + key[end:]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

# Specifif renumbering scheme to take into account the removal of some indices
def remove_scheme(removed_indices):
    def func(x):
        assert x not in removed_indices
        return x - sum(1 for y in removed_indices if y < x)
    return func

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        updated, removed = changed_indices(cfgs[cfg], batch_norm=batch_norm)
        include_string_in_keys('.module', ['classifier.0', 'classifier.3'], state_dict)
        renumber_keys('classifier', remove_scheme([1, 4]), state_dict)
        include_string_in_keys('.module', ['features.{}'.format(x) for x in updated], state_dict)
        renumber_keys('features', remove_scheme(removed), state_dict)
        model.load_state_dict(state_dict)

    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
