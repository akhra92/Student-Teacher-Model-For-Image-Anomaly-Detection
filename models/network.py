import torch
from torch import nn
from torchvision.models import vgg16
from pathlib import Path


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target_layer=11):
        # Emit one feature per "block boundary" — i.e., after each MaxPool and
        # after each ReLU not immediately followed by a MaxPool. This keeps
        # student/teacher outputs aligned position-by-position regardless of
        # whether BatchNorm is enabled or how cfg is laid out.
        result = []
        layers = list(self.features)
        for i, layer in enumerate(layers):
            x = layer(x)
            if i == target_layer:
                self.activation = x
                if x.requires_grad:
                    x.register_hook(self.activations_hook)
            is_relu = isinstance(layer, nn.ReLU)
            is_pool = isinstance(layer, nn.MaxPool2d)
            next_is_pool = (i + 1 < len(layers)) and isinstance(layers[i + 1], nn.MaxPool2d)
            if is_pool or (is_relu and not next_is_pool):
                result.append(x)
        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation


def make_layers(cfg, use_bias, batch_norm=False):
    layers = []
    in_channels = 3
    outputs = []
    for i in range(len(cfg)):
        if cfg[i] == 'O':
            outputs.append(nn.Sequential(*layers))
        elif cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1, bias=use_bias)
            torch.nn.init.xavier_uniform_(conv2d.weight)
            if batch_norm and cfg[i + 1] != 'M':
                layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[i]
    return nn.Sequential(*layers)


def make_arch(idx, cfg, use_bias, batch_norm=False):
    return VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))

class Vgg16(torch.nn.Module):
    def __init__(self, pretrain):
        super(Vgg16, self).__init__()
        features = list(vgg16('vgg16-397923af.pth').features)

        if not pretrain:
            for ind, f in enumerate(features):
                # nn.init.xavier_normal_(f)
                if type(f) is torch.nn.modules.conv.Conv2d:
                    torch.nn.init.xavier_uniform(f.weight)
                    print("Initialized", ind, f)
                else:
                    print("Bypassed", ind, f)
            # print("Pre-trained Network loaded")
        self.features = nn.ModuleList(features).eval()
        self.output = []

    def forward(self, x):
        # Same role-based emission as the student VGG: one feature after each
        # MaxPool and after each ReLU not immediately followed by a MaxPool.
        # Produces 13 outputs identical in shape to the student at indices
        # 3, 6, 9, 12 (the layers used by the loss).
        output = []
        layers = list(self.features)[:31]
        for i, layer in enumerate(layers):
            x = layer(x)
            is_relu = isinstance(layer, nn.ReLU)
            is_pool = isinstance(layer, nn.MaxPool2d)
            next_is_pool = (i + 1 < len(layers)) and isinstance(layers[i + 1], nn.MaxPool2d)
            if is_pool or (is_relu and not next_is_pool):
                output.append(x)
        return output

def get_networks(config, load_checkpoint=False):
    equal_network_size = config['equal_network_size']
    pretrain = config['pretrain']
    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    normal_class = config['normal_class']
    use_bias = config['use_bias']
    cfg = {
        'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'B': [8, 8, 'M', 8, 128, 'M', 8, 8, 256, 'M', 8, 8, 512, 'M', 8, 8, 512, 'M'],
    }

    if equal_network_size:
        config_type = 'A'
    else:
        config_type = 'B'

    vgg = Vgg16(pretrain).cuda()
    model = make_arch(config_type, cfg, use_bias, True).cuda()

    for j, item in enumerate(nn.ModuleList(model.features)):
        print('layer : {} {}'.format(j, item))

    if load_checkpoint:
        last_checkpoint = config['last_checkpoint']
        checkpoint_path = "./outputs/{}/{}/checkpoints/".format(experiment_name, dataset_name)
        model.load_state_dict(
            torch.load('{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))
        if not pretrain:
            vgg.load_state_dict(
                torch.load('{}Source_{}_random_vgg.pth'.format(checkpoint_path, normal_class)))
    elif not pretrain:
        checkpoint_path = "./outputs/{}/{}/checkpoints/".format(experiment_name, dataset_name)
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        torch.save(vgg.state_dict(), '{}Source_{}_random_vgg.pth'.format(checkpoint_path, normal_class))
        print("Source Checkpoint saved!")

    return vgg, model
