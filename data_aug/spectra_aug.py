import torch
import random
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

class AddGaussianNoise(object):
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class RandomScale(object):
    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*self.scale_range)
        return x * factor


class RandomShift(object):
    def __init__(self, shift_range: tuple = (-0.05, 0.05)):
        self.shift_range = shift_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = random.uniform(*self.shift_range)
        return x + shift


class RandomSavitzkyGolaySmoothing(object):
    """
    Aplica un suavizado aleatorio usando el filtro de Savitzky–Golay.
    """
    def __init__(
        self,
        window_length_range: tuple = (5, 15),
        polyorder_range: tuple = (2, 4),
        mode: str = 'interp'
    ):
        """
        :param window_length_range: tupla (mín, máx) de longitudes de ventana (deben ser impares)
        :param polyorder_range: tupla (mín, máx) de órdenes de polinomio (polyorder < window_length)
        :param mode: modo de tratamiento de los bordes, por defecto 'interp' (igual que SciPy)
        """
        self.win_min, self.win_max = window_length_range
        self.poly_min, self.poly_max = polyorder_range
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: Tensor 1D (L,) o (1, L)
        # Extraemos vector y pasamos a numpy
        arr = x.detach().cpu().numpy().ravel()

        # Escogemos una ventana impar al azar entre win_min y win_max
        # Ajustamos para que siempre sea impar
        wl = random.randrange(self.win_min, self.win_max + 1)
        if wl % 2 == 0:
            wl += 1
            if wl > self.win_max:
                wl -= 2

        # Escogemos un polyorder válido (< wl)
        po_min = min(self.poly_min, wl-1)
        po_max = min(self.poly_max, wl-1)
        polyorder = random.randint(po_min, po_max)

        # Aplicamos filtro Savitzky–Golay
        smooth = savgol_filter(arr, window_length=wl, polyorder=polyorder, mode=self.mode)

        # Volvemos a tensor, mismo dispositivo y forma
        smooth_t = torch.from_numpy(smooth).to(x.device)
        return smooth_t.view_as(x)


class SpectralDerivative(object):
    def __init__(self, order: int = 1):
        assert order in (1, 2), "order must be 1 or 2"
        self.order = order

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # uses torch.gradient (returns a tuple for each dim; here 1D)
        deriv = torch.gradient(x, dim=0)[0]
        if self.order == 2:
            deriv = torch.gradient(deriv, dim=0)[0]
        return deriv


class PCAJitter(object):
    def __init__(self, pca: PCA, jitter_std: float = 0.1, n_components: int = None):
        """
        pca: a fitted sklearn.decomposition.PCA object
        jitter_std: standard deviation of noise added to the top components
        n_components: how many PCA components to jitter (default: all)
        """
        self.pca = pca
        self.jitter_std = jitter_std
        self.n_components = n_components or pca.n_components_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # convert to numpy, transform to PC space
        x_np = x.cpu().numpy().reshape(1, -1)
        coeffs = self.pca.transform(x_np)
        # add Gaussian noise to the top components
        noise = np.random.randn(1, self.n_components) * self.jitter_std
        coeffs[:, :self.n_components] += noise
        # invert back to original space
        x_aug = self.pca.inverse_transform(coeffs)
        return torch.from_numpy(x_aug.flatten()).to(x.device)


class EMSA(object):
    def __init__(
        self,
        degree: int = 1,
        a0_std: float = 0.01,
        a1_std: float = 0.01,
        baseline_std: float = 0.01
    ):
        """
        degree: polynomial degree for baseline distortions
        a0_std: std of the additive constant term
        a1_std: std of the multiplicative scaling around 1
        baseline_std: std for each polynomial coefficient
        """
        self.degree = degree
        self.a0_std = a0_std
        self.a1_std = a1_std
        self.baseline_std = baseline_std
        self._basis = None

    def _init_basis(self, length: int, device):
        # create polynomial basis on [-1,1]
        grid = torch.linspace(-1, 1, steps=length, device=device)
        self._basis = [grid**d for d in range(1, self.degree + 1)]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._basis is None or self._basis[0].shape[-1] != x.shape[-1]:
            self._init_basis(x.shape[-1], x.device)

        # random additive & multiplicative terms
        a0 = 1.0 + torch.randn(1, device=x.device) * self.a0_std
        a1 = 1.0 + torch.randn(1, device=x.device) * self.a1_std

        y = a0 + a1 * x
        # add random polynomial baseline
        for p in self._basis:
            c = torch.randn(1, device=x.device) * self.baseline_std
            y = y + c * p
        return y

class SpectralAugment(object):
    """
    Agrega ruido, escala, shift, Savitzky–Golay, derivadas, PCA‐jitter y EMSA
    con probabilidades configurables.
    """
    def __init__(
        self,
        p_noise: float = 0.5,
        p_scale: float = 0.5,
        p_shift: float = 0.5,
        p_savgol: float = 0.3,
        p_derivative: float = 0.3,
        p_pca: float = 0.3,
        p_emsa: float = 0.3,
        # Parámetros para PCA‐jitter
        pca: PCA = None,
        pca_jitter_std: float = 0.1,
        pca_n_components: int = None,
        # Parámetros para EMSA
        emsa_degree: int = 1,
        emsa_a0_std: float = 0.01,
        emsa_a1_std: float = 0.01,
        emsa_baseline_std: float = 0.01,
        # Parámetro para derivada
        derivative_order: int = 1,
        # Parámetros para Savitzky–Golay
        sg_window_range: tuple = (5, 15),
        sg_polyorder_range: tuple = (2, 4),
        sg_mode: str = 'interp'
    ):
        # Inicializa cada transformador
        self.noise = AddGaussianNoise(std=0.01)
        self.scale = RandomScale()
        self.shift = RandomShift()
        self.savgol = RandomSavitzkyGolaySmoothing(
            window_length_range=sg_window_range,
            polyorder_range=sg_polyorder_range,
            mode=sg_mode
        )
        self.derivative = SpectralDerivative(order=derivative_order)
        self.pca_jitter = (
            PCAJitter(pca, jitter_std=pca_jitter_std, n_components=pca_n_components)
            if pca is not None else None
        )
        self.emsa = EMSA(
            degree=emsa_degree,
            a0_std=emsa_a0_std,
            a1_std=emsa_a1_std,
            baseline_std=emsa_baseline_std
        )

        # Probabilidades de aplicación
        self.probs = {
            'noise':     p_noise,
            'scale':     p_scale,
            'shift':     p_shift,
            'savgol':    p_savgol,
            'derivative':p_derivative,
            'pca':       p_pca  if self.pca_jitter is not None else 0.0,
            'emsa':      p_emsa
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Aplica cada transformador según su probabilidad
        if random.random() < self.probs['noise']:
            x = self.noise(x)
        if random.random() < self.probs['scale']:
            x = self.scale(x)
        if random.random() < self.probs['shift']:
            x = self.shift(x)
        if random.random() < self.probs['savgol']:
            x = self.savgol(x)
        if random.random() < self.probs['derivative']:
            x = self.derivative(x)
        if self.pca_jitter is not None and random.random() < self.probs['pca']:
            x = self.pca_jitter(x)
        if random.random() < self.probs['emsa']:
            x = self.emsa(x)
        return x

####################################################################################


# import torch
# import random
# from torchvision.transforms import functional as F

# class AddGaussianNoise(object):
#     def __init__(self, std=0.01):
#         self.std = std

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         return x + torch.randn_like(x) * self.std

# class RandomScale(object):
#     def __init__(self, scale_range=(0.9, 1.1)):
#         self.scale_range = scale_range

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         factor = random.uniform(*self.scale_range)
#         return x * factor

# class RandomShift(object):
#     def __init__(self, shift_range=(-0.05, 0.05)):
#         self.shift_range = shift_range

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         shift = random.uniform(*self.shift_range)
#         return x + shift

# class RandomSmoothing(object):
#     def __init__(self, kernel_size=5):
#         self.kernel_size = kernel_size
#         self.pad = kernel_size // 2
#         self.kernel = torch.ones(1,1,kernel_size) / kernel_size

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (1, L)
#         x_unsq = x.unsqueeze(0)  # (1,1,L)
#         kernel = self.kernel.to(x.device)
#         sm = torch.nn.functional.conv1d(x_unsq, kernel, padding=self.pad)
#         return sm.squeeze(0)

# class SpectralAugment(object):
#     def __init__(self, p_noise=0.5, p_scale=0.5, p_shift=0.5, p_smooth=0.3):
#         self.noise = AddGaussianNoise()
#         self.scale = RandomScale()
#         self.shift = RandomShift()
#         self.smooth = RandomSmoothing()
#         self.probs = {'noise': p_noise, 'scale': p_scale, 'shift': p_shift, 'smooth': p_smooth}

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         if random.random() < self.probs['noise']:
#             x = self.noise(x)
#         if random.random() < self.probs['scale']:
#             x = self.scale(x)
#         if random.random() < self.probs['shift']:
#             x = self.shift(x)
#         if random.random() < self.probs['smooth']:
#             x = self.smooth(x)
#         return x