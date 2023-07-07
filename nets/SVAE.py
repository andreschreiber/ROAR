import torch
import torch.nn as nn

# SVAE from https://github.com/tianchenji/PAAD

class SVAE(nn.Module):
    """ SVAE model from PAAD paper """
    def __init__(self, device, latent_size=32):
        """ SVAE
        
        :param device: device to use
        :param latent_size: latent feature size
        """
        super().__init__()

        self.latent_size = latent_size
        encoder_layer_sizes = [1081, 128]
        decoder_layer_sizes = [128, 1081]

        self.device = device

        self.encoder = Encoder(encoder_layer_sizes, self.latent_size)
        self.decoder = Decoder(decoder_layer_sizes, self.latent_size)

    def forward(self, x):
        """ Compute SVAE features and reconstruction
        
        :param x: input (shape: [B, 1081])
        :returns: reconstruction, means, and log variances
        """
        batch_size = x.size(0)
        means, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means
        recon_x = self.decoder(z)
        return recon_x, means, log_var

class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        """ Encoder for SVAE
        
        :param layer_sizes: input layer sizes
        :param latent_size: latent feature size
        """
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        """ Encodes x with SVAE encoder
        
        :param x: input
        :returns: means and log variances
        """
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        """ Decoder for SVAE
        
        :param layer_sizes: layer sizes
        :param latent_size: latent feature size
        """
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([latent_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        """ Decode z with SVAE decoder
        
        :param z: encoding
        :returns: reconstruction
        """
        x = self.MLP(z)
        return x
