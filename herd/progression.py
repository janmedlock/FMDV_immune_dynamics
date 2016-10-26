from scipy import stats

from . import rv


class gen(rv.RV):
    '''
    Gamma-distributed latent period with rate
    latent_mean and shape latent_shape.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.latent_mean = parameters.latent_mean
        self.latent_shape = parameters.latent_shape
        distn = stats.gamma(self.latent_shape,
                            scale = self.latent_mean / self.latent_shape)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('latent_mean', 'latent_shape'))
