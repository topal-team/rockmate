import torch
from .models import resnet
from .models import inception
from .models import densenet
from .models import vgg
from .models.utils import AtomicSequential

class Dimension:
    def __init__(self, str):
        if 'x' in str: 
            self.x, self.y = map(int, str.split('x'))
        else: 
            self.x = int(str)
            self.y = self.x


class ForkConv(torch.nn.Module):
    def __init__(self):
        super(ForkConv, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3)
        self.conv2 = torch.nn.Conv1d(1, 1, 3)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return (a, b)

class BiConv(torch.nn.Module):
    def __init__(self):
        super(BiConv, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 3)
        self.conv2 = torch.nn.Conv1d(1, 1, 3)

    def forward(self, xs):
        a = self.conv1(xs[0])
        b = self.conv2(xs[1])
        return (a, b)

class Join(torch.nn.Module):
    def __init__(self):
        super(Join, self).__init__()

    def forward(self, xs):
        return xs[0] + xs[1]
    
    

class Network:
    allowedNames = ["biseq", "seq", "resnet", "inception", "densenet", "vgg" ]

    resnets = { 18: resnet.resnet18, 34: resnet.resnet34,
                50: resnet.resnet50, 101: resnet.resnet101, 152: resnet.resnet152,
                200: resnet.resnet200, 1001: resnet.resnet1001 }

    densenets = { 121: densenet.densenet121, 161: densenet.densenet161,
                  169: densenet.densenet169, 201: densenet.densenet201 }

    vggnets = { 11: vgg.vgg11_bn, 13: vgg.vgg13_bn, 16: vgg.vgg16_bn, 19: vgg.vgg19_bn}
    

    networks = { "resnet": (resnets, {}), "densenet": (densenets, {"drop_rate": 0.25}), "vgg": (vggnets, {}) }
    
    def __init__(self, descr):
        parts = descr.split(':')
        self.name = parts[0]
        self.length = int(parts[1]) if len(parts) > 1 else 18
        self.size = Dimension(parts[2]) if len(parts) > 2 else None
        if self.name not in Network.allowedNames:
            raise ValueError("Unknown network name {}. Allowed names: {}".format(self.name, Network.allowedNames))

    def make_module_from_number(self, functions, kwargs): 
        if not self.size: self.size = Dimension('224')
        self.shape = (3, self.size.x, self.size.y)
        try:
                 module = functions[self.length](pretrained=False, **kwargs)
        except KeyError:
            print("Unknown {} number {}. Known values: {}".format(self.name, self.length, list(functions.keys())))
            exit(-1)
        return module
        
    def make_module(self, device=None):
        if self.name == "seq": 
            if not self.size: self.size = 100
            else: self.size = self.size.x
            self.shape = (1, self.size)
            module = torch.nn.Sequential()
            for idx in range(self.length):
                if idx % 2 == 0:
                    module.add_module(str(idx), AtomicSequential(torch.nn.Conv1d(1, 1, 3), torch.nn.Dropout(p=0.1, inplace=False)))
                else: 
                    module.add_module(str(idx), torch.nn.Conv1d(1, 1, 50))
        elif self.name == "biseq": 
            if not self.size: self.size = 100
            else: self.size = self.size.x
            self.shape = (1, self.size)
            module = torch.nn.Sequential()
            module.add_module("fork", ForkConv())
            for idx in range(self.length - 2):
                module.add_module(str(idx), BiConv())
            module.add_module("join", Join())
        elif self.name == "inception":
            if not self.size: self.size = Dimension('299')
            self.shape = (3, self.size.x, self.size.y)
            module = inception.Inception3(aux_logits=False)
        else:
            try: 
                (functions, kwargs) = self.networks[self.name]
            except KeyError:
                print("Unknown network name {}. Allowed Names: {}".format(self.name, allowedNames))
                exit(-1)
            module = self.make_module_from_number(functions, kwargs)

        if device:
            module.to(device=device)

        for (n, p) in module.named_parameters():
            p.grad = torch.zeros_like(p)
            
        return module
            
