from torch import nn


class LIIF(nn.Module):
    def __init__(self):
        super(LIIF, self).__init__()


class VINR(nn.Module):
    def __init__(self, encoder, liif, modulator, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.liif = liif
        self.modulator = modulator
        self.mapper = mapper

    def forward(self, frames, t):
        encoded = self.encoder(frames)
        continuous_feature = self.liif(encoded)

        # TODO: Namalization?

        mod_params = self.modulator(continuous_feature)
        rgb = self.mapper(t.unsqueeze(-1), mod_params).permute(0, 3, 1, 2)
        return rgb
