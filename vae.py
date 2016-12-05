"""Variational autoencoder for mxnet

Code adapted from
- https://github.com/fastforwardlabs/vae-tf
- https://github.com/dmlc/mxnet/blob/master/example/autoencoder/autoencoder.py
"""

import mxnet as mx

class VAE():
    """Variational Autoencoder
    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (https://arxiv.org/abs/1312.6114)
    """

    DEFAULTS = {
        "nonlinearity": "elu",
        "squashing": "sigmoid",
        "dropout": 0.0,  # Fraction of the input that gets dropped out at training time
    }

    def __init__(self,
                 architecture,
                 d_hyperparams={},
                 saved_model=None,
                 save_graph_def=True,
                 log_dir="./log"):
        """(Re)build a symmetric VAE model with given:
         * architecture (list of nodes per encoder layer); e.g.
           [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
           & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]
         * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)

    def testing_model(self):
        z = mx.sym.Variable('data')
        decoder = self._build_decoder(z)

        return decoder

    def training_model(self):
        data = mx.sym.Variable('data')
        z_mean, z_log_sigma, kl_divergence = self._build_encoder(data)
        z = self._build_z(z_mean, z_log_sigma)
        decoder = self._build_decoder(z)

        cross_entropy = VAE._cross_entropy(decoder, data)
        cross_entropy = mx.sym.MakeLoss(cross_entropy)

        group = mx.symbol.Group([mx.sym.BlockGrad(z_mean),
                                 mx.sym.BlockGrad(z_log_sigma),
                                 mx.sym.BlockGrad(decoder),
                                 kl_divergence,
                                 cross_entropy])

        return group

    def _build_encoder(self, data):
        N = len(self.architecture)
        x = data
        for i in range(1, N-1):
            x = mx.sym.FullyConnected(
                name='encoder_{0}'.format(i),
                data=x,
                num_hidden=self.architecture[i])
            x = mx.sym.LeakyReLU(data=x, act_type=self.nonlinearity)
            x = mx.sym.Dropout(data=x, p=self.dropout)

        z_mean = mx.sym.FullyConnected(
            name="z_mean",
            data=x,
            num_hidden=self.architecture[-1])

        z_log_sigma = mx.sym.FullyConnected(
            name="z_log_sigma",
            data=x,
            num_hidden=self.architecture[-1])

        kl_divergence = VAE._kullback_leibler(z_mean, z_log_sigma)
        kl_divergence = mx.sym.MakeLoss(kl_divergence)

        return (z_mean, z_log_sigma, kl_divergence)

    def _build_z(self, z_mean, z_log_sigma):
        # broadcast_mul automatically broadcasts the normal symbol to the right shape
        return z_mean + mx.sym.broadcast_mul(
            mx.sym.normal(shape=(1, self.architecture[-1])), mx.sym.exp(z_log_sigma))

    def _build_decoder(self, z):
        N = len(self.architecture)
        x = z
        for i in reversed(range(1, N-1)):
            x = mx.sym.FullyConnected(
                name='decoder_{0}'.format(i),
                data=x,
                num_hidden=self.architecture[i])
            x = mx.sym.LeakyReLU(data=x, act_type=self.nonlinearity)
            x = mx.sym.Dropout(data=x, p=self.dropout)

        x = mx.sym.FullyConnected(
            name="output",
            data=x,
            num_hidden=self.architecture[0])
        x = mx.sym.Activation(data=x, act_type=self.squashing)

        return x

    @staticmethod
    def _cross_entropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        return -mx.sym.sum(actual * mx.sym.log(obs + offset) +
                              (1 - actual) * mx.sym.log(1 - obs + offset), axis=1)

    @staticmethod
    def _kullback_leibler(mu, log_sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        # 0.5 ( Tr[Σ] + <μ,μ> - k - log|Σ| ) = -0.5 \sum ( 1 + 2logσ - μ² - σ² )
        return -0.5 * mx.sym.sum(1 + 2*log_sigma - mu*mu - mx.sym.exp(2 * log_sigma), axis=1)
