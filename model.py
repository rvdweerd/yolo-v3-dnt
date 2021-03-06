import torch
import torch.nn as nn
import numpy as np

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        # kwargs will be kernel size, stride
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act
    
    def forward(self, x):
        # for scale predictions we don't use batch normalization and activation
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1) 
                ) 
             ] # 2 CNNBlocks should be inside a sequential
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size =1)
            # each achor box we need: 1 node for each of the predicted classes, 5 (bounding box p_c,x,y,w,h)
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        return (
            # split vector of all bounding boxes to dim1 = which of the three bounding boxes
            self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        # scale 1: (N x (3 anchor boxes) x (13x13 grid) x (5+num_classes))
        # scale 2: .. 26x26
        # scale 3: .. 52x52

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = [] # one output for each scale prediction
        route_connections = [] # where we concatenate the channels (skip channels)
        for layer in self.layers:
            if isinstance(layer,ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x) # !! CHECK AND FIX

            if isinstance(layer, ResBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1) # channels are concatenated with last routeconnection
                route_connections.pop()
        return outputs

    def numTrainableParameters(self):
        total = 0
        for name, p in self.layers.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("\nTotal number of parameters: {}\n".format(total))
        #assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                            in_channels,
                        out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = 1 if kernel_size == 3 else 0
                    )
                )
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResBlock(in_channels, num_repeats = num_repeats
                    )
                )
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module  == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 # concatenation after upscaling
        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416 # yolov1 448, yolov3 416 (multi-scale training)
    model = YOLOv3(num_classes=num_classes)
    model.numTrainableParameters()
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5) # IMAGE_SIZE // 32 = 13
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5) # 26
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)   # 52
    print("Success!")