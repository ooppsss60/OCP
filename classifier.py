import torch
import os
import h5py
import numpy as np
from PIL import Image
import torch.nn as nn

mean = 0.3291
std = 0.1260
classes = {0: 'good', 1: 'color', 2: 'cut', 3: 'hole', 4: 'thread', 5: 'metal_contamination'}
n_classes = 6
cfg = {
    'simple_1': [16, 'M', 64, 'M', 512, 512, 'M'],
    'simple_regr_chan': [32, 'M', 64, 'M', 256, 256, 'M', 128, 64, n_classes]
}
path_weights = 'weights/simpnet_regr_chan_weights_v2_22_7.pth'


class Net(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.features = self.feature_extractor(cfg[name])
        self.classifier = nn.AvgPool2d(kernel_size=8, stride=1)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features).squeeze()

    def feature_extractor(self, cfg):
        layers = list()
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def classify(path_set: set):

    simple_net = Net("simple_regr_chan")
    simple_net = simple_net.double()
    simple_net.load_state_dict(torch.load(path_weights, map_location=torch.device('cpu')))
    preds = list()
    for path in path_set:
        pic = Image.open(path)
        pix = np.array(pic) / 255
        inp = (torch.tensor(pix[None, None, :, :])-mean)/std
        simple_net.eval()
        with torch.no_grad():
            outputs = simple_net(inp)
            result = {
                'prob': torch.softmax(outputs, dim=-1).tolist(),
                'pred': int(torch.argmax(outputs))
            }
            preds.append(result)
    return preds


def unpack_h5():
    f = h5py.File('OCP/test64.h5', 'r')
    dset = f['images']
    dir = 'static/images/'
    os.makedirs(dir, exist_ok=True)

    for i in range(dset.shape[0]):
        data = np.array(dset[i, :, :, 0])
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
        img = Image.fromarray(rescaled)
        img.save(f"static/images/{i}.jpg", "JPEG")
        print(i)

