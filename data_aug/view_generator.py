import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x, wl=None):
        """
        Genera `n_views` transformaciones sobre (x, wl). Si wl es None, asume transformación independiente de wl.
        """
        if wl is None:
            # solo intensidades
            return [self.base_transform(x) for _ in range(self.n_views)]
        # intensidades y wl retornados como tuplas (x_i, wl_i)
        views = [self.base_transform(x, wl) for _ in range(self.n_views)]
        xs, wls = zip(*views)
        return list(xs), list(wls)
