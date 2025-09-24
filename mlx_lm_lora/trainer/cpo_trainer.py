from pathlib import Path
from tqdm import tqdm
import time

from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from mlx_lm.tuner.callbacks import TrainingCallback

from .sft_trainer import grad_checkpoint
from .dpo_trainer import DPOTrainingArgs as CPOTrainingArgs

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def get_token_scores(model, x, mask):
    inputs, targets = x[:, :-1], x[:, 1:]
    logits = model(inputs).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, targets) * mask[:, :-1]

def compute_score(scores, mask, loss_type):
    token_count = mask.sum(-1)
    return scores.sum(-1) / token_count if loss_type == "ipo" else scores.sum(-1)


def cpo_loss(
    policy_chosen_score: mx.array,
    policy_rejected_score: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    loss_type: str = "sigmoid",
):
    # Preference logits
    logits = (policy_chosen_score - policy_rejected_score)

    # Loss calculation
    if loss_type == "sigmoid":
        losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(
            mx.zeros_like(policy_chosen_score),
            policy_rejected_score - policy_chosen_score,
        )
        losses = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Token counts and rewards
    num_chosen_tokens = chosen_masks.sum(-1)
    num_rejected_tokens = rejected_masks.sum(-1)
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    chosen_reward = beta * mx.mean(policy_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    # Metrics
    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_rejected_logps": mx.mean(policy_rejected_score),
        "policy_chosen_logps": mx.mean(policy_chosen_score / num_chosen_tokens),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
    }

    mx.clear_cache()
    return mx.mean(losses), reward, num_tokens, metrics


def iterate_cpo_batches(dataset, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["chosen"]))

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = (
            np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        )
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]

            # Get and process lengths
            chosen_lengths = [len(x["chosen"]) for x in batch]
            rejected_lengths = [len(x["rejected"]) for x in batch]
            max_length = min(
                max(max(chosen_lengths), max(rejected_lengths)), max_seq_length
            )

            # Dynamic padding based on batch content
            max_length_in_batch = max_length

            chosen_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)
            rejected_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            chosen_masks = np.zeros(
                (batch_size // step, max_length_in_batch), np.float32
            )
            rejected_masks = np.zeros(
                (batch_size // step, max_length_in_batch), np.float32
            )

            for j in range(batch_size // step):
                chosen_length = min(chosen_lengths[j], max_seq_length)
                rejected_length = min(rejected_lengths[j], max_seq_length)

                chosen_arr[j, :chosen_length] = batch[j]["chosen"][:chosen_length]
                rejected_arr[j, :rejected_length] = batch[j]["rejected"][
                    :rejected_length
                ]

                chosen_masks[j, :chosen_length] = 1.0
                rejected_masks[j, :rejected_length] = 1.0

            yield mx.array(chosen_arr), mx.array(rejected_arr), mx.array(
                chosen_masks
            ), mx.array(rejected_masks)

        if not train:
            break



def evaluate_cpo(
    model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length,
    loss_type,
    loss_fn: callable = cpo_loss,
):
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_cpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        chosen, rejected, chosen_masks, rejected_masks = batch

        policy_chosen_scores = get_token_scores(model, chosen, chosen_masks)
        policy_rejected_scores = get_token_scores(model, rejected, rejected_masks)

        policy_chosen_score = compute_score(policy_chosen_scores, chosen_masks, loss_type)
        policy_rejected_score = compute_score(policy_rejected_scores, rejected_masks, loss_type)

        loss_value, reward, toks, metrics = loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            loss_type=loss_type,
            beta=beta,
            delta=delta,
        )
        all_losses += loss_value * toks
        all_rewards += reward
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        mx.eval(all_losses, all_rewards, ntokens)
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()

    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_cpo(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args: CPOTrainingArgs = CPOTrainingArgs(),
    loss_fn: callable = cpo_loss,
    training_callback: TrainingCallback = None,
):
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    print(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        chosen, rejected, chosen_masks, rejected_masks = batch

        policy_chosen_scores = get_token_scores(model, chosen, chosen_masks)
        policy_rejected_scores = get_token_scores(model, rejected, rejected_masks)

        policy_chosen_score = compute_score(policy_chosen_scores, chosen_masks, args.loss_type)
        policy_rejected_score = compute_score(policy_rejected_scores, rejected_masks, args.loss_type)

        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            policy_chosen_score, policy_rejected_score, chosen_masks=chosen_masks, rejected_masks=rejected_masks
        )
        
        if (it + 1) % args.gradient_accumulation_steps == 0:
            grad = average_gradients(grad)
            optimizer.update(model, grad)

        return (lvalue / args.gradient_accumulation_steps), reward, toks, metrics

    def loss_wrapper(policy_chosen_score, policy_rejected_score, chosen_masks, rejected_masks):
        return loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.loss_type,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    losses = 0
    rewards = mx.zeros((2,))
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "accuracies": 0,
        "margins": 0,
        "policy_rejected_logps": 0,
        "policy_chosen_logps": 0,
        "rejected_logits_mean": 0,
        "chosen_logits_mean": 0,
    }

    start = time.perf_counter()
    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        batch = next(iterate_cpo_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ))

        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_cpo(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=args.beta,
                delta=args.delta,
                loss_type=args.loss_type,
                loss_fn=loss_fn,
            )
            val_time = time.perf_counter() - stop
            if rank == 0:
                tqdm.write(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val chosen reward {val_rewards[0]:.3f}, "
                    f"Val rejected reward {val_rewards[1]:.3f}, "
                    f"Val accuracy {val_metrics['accuracies']:.3f}, "
                    f"Val margin {val_metrics['margins']:.3f}, "
                    f"Val took {val_time:.3f}s",
                )

            if training_callback is not None:
                training_callback.on_val_loss_report(
                    {
                        "iteration": it,
                        "val_loss": val_loss,
                        "val_chosen_reward": val_rewards[0],
                        "val_rejected_reward": val_rewards[1],
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "val_time": val_time,
                    }
                )

            start = time.perf_counter()

        lvalue, reward, toks, metrics = step(batch)
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, rewards, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = mx.distributed.all_sum(rewards).tolist()
            train_rewards = [r / (steps * world_size) for r in train_rewards]
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                pbar.set_postfix({
                    'loss': f"{train_loss:.3f}",
                    'it/s': f"{it_sec:.3f}",
                })
                tqdm.write(
                    f"\nIter {it}: "
                    f"loss {train_loss:.3f}, "
                    f"chosen_r {train_rewards[0]:.3f}, "
                    f"rejected_r {train_rewards[1]:.3f}, "
                    f"acc {avg_metrics['accuracies']:.3f}, "
                    f"margin {avg_metrics['margins']:.3f}, "
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            rewards = mx.zeros((2,))
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
