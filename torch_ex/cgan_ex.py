from torch import nn
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn. Linear(in_feat, out_feat)]
            if normalize:
            layers.append(nn.BatchNormld(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
        *block(opt.latent_dim + opt. n_classes, 128., normalize=False),
        *block(128, 256),
        *block(256, 512),
        *block(512, 1024),
        nn. Linear(1024, int(np.prod(img_shape))),
        nn.Tanh()
        )
class Discriminator(nn.Module):
    def __init__(self):
        super (Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(
            nn. Linear(opt.n_classes + int(np.prod(img_shape)), 512).,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )
