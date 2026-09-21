"""Single entry point for building an NFTSF model from just its sizes."""

from .architecture import create_nfm
from .architecture_encoder import create_nfm_encoder

# Below this context length, the raw (standardized) past is short enough to
# feed directly as context to every flow layer. At or above it, a context
# encoder compresses the past into a fixed-size vector first (cheaper, and
# what every experiment in the paper uses for long contexts).
ENCODER_CONTEXT_THRESHOLD = 100


def build_nftsf(
    device,
    n_past,
    n_future,
    flow_blocks=6,
    hidden_units=64,
    hidden_layers_list=(1, 2),
    tail_bound=30.0,
    encoder_type="cnn",
    encoder_hidden=128,
    encoder_layers=2,
    context_dim=64,
):
    """Build an NFTSF model sized for (n_past, n_future).

    Picks the base conditional flow (`create_nfm`) for short contexts
    (`n_past < 100`) or the context-encoder flow (`create_nfm_encoder`,
    CNN by default) for longer ones — matching what `train.py` and
    `forecast.py` use, so you don't have to choose the variant yourself.
    """
    if n_past < ENCODER_CONTEXT_THRESHOLD:
        return create_nfm(
            device=device, latent_size=n_future, context_size=n_past,
            K=flow_blocks, hidden_units=hidden_units,
            hidden_layers_list=hidden_layers_list, tail_bound=tail_bound,
        )
    return create_nfm_encoder(
        device=device, n_past=n_past, n_future=n_future, past_dim=1,
        encoder=encoder_type, encoder_hidden=encoder_hidden,
        encoder_layers=encoder_layers, context_dim=context_dim,
        K=flow_blocks, hidden_units=hidden_units,
        hidden_layers_list=hidden_layers_list, tail_bound=tail_bound,
    )
