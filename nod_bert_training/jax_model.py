import transformers
from transformers.models.auto import FlaxAutoModelForMaskedLM
from transformers import FlaxBertModel
import jax
import optax
import flax
from flax.training.common_utils import onehot
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from numpy.typing import ArrayLike
from typing import Dict, Callable, Any, List

RngState = Any


def create_model() -> FlaxBertModel:
    return FlaxBertModel.from_pretrained("bert-base-cased")


Batch = Dict[str, ArrayLike]
TrainStepMetrics = Dict[str, Any]
TrainStep = Callable[
    [TrainState, Batch, RngState], List[TrainState, TrainStepMetrics, RngState]
]


def make_train_step_fn(decay_lr_schedule_fn: optax.Schedule) -> TrainStep:
    def train_step(state: TrainState, batch: Dict[str, ArrayLike], rng: RngState):
        dropout_rng, new_rng = jax.random.split(rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]

            # compute loss, ignore padded input tokens
            label_mask = jnp.where(labels > 0, 1.0, 0.0)
            loss = (
                optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
                * label_mask
            )

            # take average
            loss = loss.sum()
            num_labels = label_mask.sum()

            return loss, num_labels

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, num_labels), grad = grad_fn(state.params)
        num_labels = jax.lax.psum(num_labels, "batch")

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        # true grad = total grad / total samples
        grad = jax.lax.psum(grad, "batch")
        grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss, "learning_rate": decay_lr_schedule_fn(state.step)}

        return new_state, metrics, new_rng

    return train_step


def make_train_step_data_parallel(train_step: TrainStep) -> TrainStep:
    return jax.pmap(train_step, "batch", donate_argnums=(0,))
