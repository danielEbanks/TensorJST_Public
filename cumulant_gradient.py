import tensorly as tl
# This should work with any backend
# tl.set_backend('pytorch')
from tensorly.cp_tensor import cp_mode_dot
import tensorly.tenalg as tnl
from tensorly.tenalg.core_tenalg import tensor_dot, batched_tensor_dot, outer, inner

def cumulant_gradient(phi, y_batch, alpha=1, theta=1):
    """Computes the average gradient for a batch of whitened samples
    phi : (n_features, rank)
        factor to be optimized
    y_batch : (n_samples, n_features)
        each row is one whitened sample
    Returns
    -------
    phi_gradient : gradient of the loss with respect to Phi
        of shape (n_features, rank)
    """
    gradient = 3*(1 + theta)*tl.dot(phi, tl.dot(phi.T, phi)**2)
    gradient -= 3*(1 + alpha)*(2 + alpha)/(2*y_batch.shape[0])*tl.dot(y_batch.T, tl.dot(y_batch, phi)**2)
    return gradient