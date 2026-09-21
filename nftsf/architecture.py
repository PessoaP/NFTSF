import normflows as nf


def create_nfm(device, latent_size, context_size, K=6, hidden_units=64, hidden_layers_list=(1,2), tail_bound=30):
    """
    Constructs a conditional normalizing flow model that learns the
    conditional density p(x_future | x_past).

    A normalizing flow models a complex target distribution by applying a
    sequence of invertible, differentiable transformations (flow layers)
    to a simple base distribution.  Given base density q_0(z) (here a
    diagonal Gaussian) and composed bijection f = f_1 o f_2 o ... o f_L,
    the model density is obtained via the change-of-variables formula:
        p(x) = q_0(f^{-1}(x)) * |det(df^{-1}/dx)|

    Architecture components:

    1. Base distribution (prior):
        q_0 = DiagGaussian(latent_size)
       A factorized Gaussian N(0, I) in R^{n_future}. This is the
       distribution in the latent space that gets transformed into the
       complex forecast distribution.

    2. Flow layers (repeated K times):
       Each block consists of:

       a) Autoregressive Rational Quadratic Spline (RQS) transforms:
          These are neural spline flows (Durkan et al., 2019) that
          parameterize each conditional x_i | x_{<i} via a monotonic
          rational-quadratic spline. The spline knot positions and
          derivatives are predicted by an autoregressive neural network
          conditioned on x_past (the context). The `tail_bound` parameter
          defines the region [-B, B] where the spline is active; outside
          this region the transform is the identity (linear tails).
          Multiple RQS layers with different network depths
          (hidden_layers_list) are stacked per block.

       b) LU Linear Permute:
          A learnable linear layer parameterized via its LU decomposition,
          which permutes and mixes dimensions between autoregressive layers.
          This is essential because autoregressive transforms only model
          dependencies in one ordering; permutations ensure all
          inter-dimensional dependencies are captured across layers.

    3. Conditional normalizing flow:
       The context vector x_past is fed to each spline layer's conditioner
       network, enabling the flow to produce different forecast
       distributions for different observed histories.

    The total number of flow layers is K * (len(hidden_layers_list) + 1).

    Args:
        device: torch device for the model
        latent_size: dimensionality of the target space (n_future)
        context_size: dimensionality of the conditioning input (n_past)
        K: number of flow blocks (default 4)
        hidden_units: width of hidden layers in the spline conditioner networks
        hidden_layers_list: tuple of network depths to use per block
                           (e.g., (1,2) creates one 1-layer and one 2-layer RQS per block)
        tail_bound: spline boundary B; the RQS is active on [-B, B]

    Returns:
        Conditional normalizing flow model on the specified device.
    """
    prior = nf.distributions.DiagGaussian(latent_size, trainable=False)
    flows = []
    for _ in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, num_context_channels=context_size, tail_bound=tail_bound) for hidden_layers in hidden_layers_list]
        flows += [nf.flows.LULinearPermute(latent_size)]
    nfm = nf.ConditionalNormalizingFlow(q0=prior, flows=flows)
    return nfm.to(device)
