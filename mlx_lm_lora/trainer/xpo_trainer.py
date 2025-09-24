from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import time

from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from mlx_lm.tuner.callbacks import TrainingCallback

from .online_dpo_trainer import OnlineDPOTrainingArgs, generate_for_online_dpo, compute_score
from .judge import LLMPairwiseJudge, HumanPairwiseJudge
from .dpo_trainer import get_token_scores
from .sft_trainer import grad_checkpoint

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class XPOTrainingArgs(OnlineDPOTrainingArgs):
    alpha: list[float] = field(
        default=lambda: [1e-5],
        metadata={
            "help": "Weight of the XPO loss term. If a list of floats is provided then the alpha is selected for each new epoch and the last alpha is used for the rest of the epochs."
        }
    )


def get_current_alpha(step: int, total_steps: int, alpha_schedule: list[float]) -> float:
    if len(alpha_schedule) == 1:
        return alpha_schedule[0]

    step_size = total_steps // len(alpha_schedule)
    index = min(step // step_size, len(alpha_schedule) - 1)
    return alpha_schedule[index]


def xpo_loss(
    policy_chosen_score: mx.array,
    policy_rejected_score: mx.array,
    reference_chosen_score: mx.array,
    reference_rejected_score: mx.array,
    chosen_masks: mx.array,
    rejected_masks: mx.array,
    beta: float,
    delta: float,
    loss_type: str = "sigmoid",
    alpha: float = 0.0,  # Add alpha parameter
):
    # Preference logits
    logits = (policy_chosen_score - policy_rejected_score) - \
             (reference_chosen_score - reference_rejected_score)

    # Standard DPO Loss calculation
    if loss_type == "sigmoid":
        dpo_losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        dpo_losses = nn.relu(1 - beta * logits)
    elif loss_type == "ipo":
        dpo_losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(
            mx.zeros_like(policy_chosen_score),
            reference_chosen_score - policy_chosen_score,
        )
        dpo_losses = -(nn.log_sigmoid(beta * logits) - delta * penalty)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # XPO Exploration Bonus
    if alpha > 0:
        # Compute KL divergence between policy and reference for exploration
        # KL(π || π_ref) = log(π) - log(π_ref)
        chosen_kl = policy_chosen_score - reference_chosen_score
        rejected_kl = policy_rejected_score - reference_rejected_score
        
        # Exploration bonus encourages deviation from reference model
        exploration_bonus = alpha * (chosen_kl + rejected_kl)
        
        # Combine DPO loss with exploration bonus
        losses = dpo_losses - exploration_bonus
    else:
        losses = dpo_losses

    # Token counts and rewards
    num_chosen_tokens = chosen_masks.sum(-1) if hasattr(chosen_masks, 'sum') else chosen_masks
    num_rejected_tokens = rejected_masks.sum(-1) if hasattr(rejected_masks, 'sum') else rejected_masks
    num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

    chosen_reward = beta * mx.mean(policy_chosen_score - reference_chosen_score)
    rejected_reward = beta * mx.mean(policy_rejected_score - reference_rejected_score)
    reward = mx.stack([chosen_reward, rejected_reward])

    # Metrics
    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_rejected_logps": mx.mean(policy_rejected_score / num_rejected_tokens),
        "policy_chosen_logps": mx.mean(policy_chosen_score / num_chosen_tokens),
        "rejected_logits_mean": mx.mean(policy_rejected_score),
        "chosen_logits_mean": mx.mean(policy_chosen_score),
        "exploration_bonus": 0,
        "chosen_kl": 0,
        "rejected_kl": 0,
    }
    
    # Add XPO-specific metrics
    if alpha > 0:
        metrics["exploration_bonus"] = mx.mean(exploration_bonus)
        metrics["chosen_kl"] = mx.mean(chosen_kl)
        metrics["rejected_kl"] = mx.mean(rejected_kl)

    mx.clear_cache()
    return mx.mean(losses), reward, num_tokens, metrics


def iterate_online_dpo_batches(dataset, batch_size, max_seq_length, train=False):
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["prompt"]))

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
            prompts = [x["prompt"] for x in batch]
            prompt_text = [x["prompt_text"] for x in batch]

            yield prompts, prompt_text
        if not train:
            break


def evaluate_xpo(
    model,
    ref_model,
    dataset,
    batch_size,
    num_batches,
    beta: float,
    delta: float,
    max_seq_length,
    loss_type,
    judge_config,
    alpha: float,
    loss_fn: callable = xpo_loss,
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    tokenizer=None,
    max_tokens: int = 512
):
    all_losses = 0
    all_rewards = mx.zeros((2,))
    all_metrics = None
    ntokens = 0
    
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)
    
    for _, batch in zip(
        index_iterator,
        iterate_online_dpo_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        prompts, prompt_texts = batch
        
        completions = generate_for_online_dpo(model, tokenizer, prompts, max_tokens=max_tokens)
        
        if judge_model == "human":
            judger = HumanPairwiseJudge()
            judged = judger.judge(prompt_texts, completions=completions)
        else:
            judger = LLMPairwiseJudge(model=judge_model, tokenizer=judge_tokenizer, system_prompt=judge_config.get("system_prompt", None))
            judged = judger.judge(prompt_texts, completions=completions)
        
        chosen = []
        rejected = []
        for i, (prompt_text, completion_pair, judgment) in enumerate(zip(prompt_texts, completions, judged)):
            if judgment == 0:
                chosen.append(prompt_text + completion_pair[0])
                rejected.append(prompt_text + completion_pair[1])
            else:
                chosen.append(prompt_text + completion_pair[1])
                rejected.append(prompt_text + completion_pair[0])
        
        chosen_tokens = [mx.array(tokenizer.encode(text)) for text in chosen]
        rejected_tokens = [mx.array(tokenizer.encode(text)) for text in rejected]
        
        chosen_masks = [mx.ones(len(tokens)) for tokens in chosen_tokens]
        rejected_masks = [mx.ones(len(tokens)) for tokens in rejected_tokens]
        
        # Fix the get_token_scores calls - convert to proper batch format
        policy_chosen_scores = []
        policy_rejected_scores = []
        
        for tokens, mask in zip(chosen_tokens, chosen_masks):
            batch_tokens = tokens.reshape(1, -1)  # Shape: (1, seq_len)
            batch_mask = mask.reshape(1, -1)      # Shape: (1, seq_len)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_chosen_scores.append(score)
            
        for tokens, mask in zip(rejected_tokens, rejected_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_rejected_scores.append(score)
        
        policy_chosen_score = mx.array([compute_score(score, mask, loss_type) for score, mask in zip(policy_chosen_scores, chosen_masks)])
        policy_rejected_score = mx.array([compute_score(score, mask, loss_type) for score, mask in zip(policy_rejected_scores, rejected_masks)])
        
        if ref_model is None:
            reference_chosen_logprobs = mx.zeros_like(policy_chosen_score)
            reference_rejected_logprobs = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_scores = []
            ref_rejected_scores = []
            
            for tokens, mask in zip(chosen_tokens, chosen_masks):
                batch_tokens = tokens.reshape(1, -1)
                batch_mask = mask.reshape(1, -1)
                score = mx.stop_gradient(get_token_scores(ref_model, batch_tokens, batch_mask))
                ref_chosen_scores.append(score)
                
            for tokens, mask in zip(rejected_tokens, rejected_masks):
                batch_tokens = tokens.reshape(1, -1)
                batch_mask = mask.reshape(1, -1)
                score = mx.stop_gradient(get_token_scores(ref_model, batch_tokens, batch_mask))
                ref_rejected_scores.append(score)
                
            reference_chosen_logprobs = mx.array([compute_score(score, mask, loss_type) for score, mask in zip(ref_chosen_scores, chosen_masks)])
            reference_rejected_logprobs = mx.array([compute_score(score, mask, loss_type) for score, mask in zip(ref_rejected_scores, rejected_masks)])
        
        # Convert masks to token counts
        chosen_mask_counts = mx.array([mask.sum() for mask in chosen_masks])
        rejected_mask_counts = mx.array([mask.sum() for mask in rejected_masks])
        
        # Compute loss
        loss_value, reward, toks, metrics = loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_logprobs,
            reference_rejected_score=reference_rejected_logprobs,
            chosen_masks=chosen_mask_counts,
            rejected_masks=rejected_mask_counts,
            loss_type=loss_type,
            beta=beta,
            delta=delta,
            alpha=alpha,
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
    
    # Distributed reduction
    all_losses = mx.distributed.all_sum(all_losses)
    all_rewards = mx.distributed.all_sum(all_rewards)
    ntokens = mx.distributed.all_sum(ntokens)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}
    
    # Compute averages
    avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_rewards = (all_rewards / ntokens).tolist()
    avg_loss = (all_losses / ntokens).item()
    
    return avg_loss, avg_rewards, ntokens, avg_metrics


def train_xpo(
    model,
    ref_model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    judge_config,
    args: XPOTrainingArgs = XPOTrainingArgs(),
    judge_model: mx.array = None,
    judge_tokenizer: mx.array = None,
    loss_fn: callable = xpo_loss,
    training_callback: TrainingCallback = None,
):
    tqdm.write(f"Starting XPO training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        tqdm.write(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch, current_alpha):
        prompts, prompt_texts = batch
        
        # Generate completions for each prompt
        completions = generate_for_online_dpo(model, tokenizer, prompts, max_tokens=args.max_completion_length)
        
        # Judge the completions
        if judge_model == "human":
            judger = HumanPairwiseJudge()
            judged = judger.judge(prompt_texts, completions=completions)
        else:
            judger = LLMPairwiseJudge(model=judge_model, tokenizer=judge_tokenizer, system_prompt=judge_config.get("system_prompt", None))
            judged = judger.judge(prompt_texts, completions=completions)
        
        # Process judged results to create chosen/rejected pairs
        chosen = []
        rejected = []
        for i, (prompt_text, completion_pair, judgment) in enumerate(zip(prompt_texts, completions, judged)):
            if judgment == 0:  # First completion is preferred
                chosen.append(prompt_text + completion_pair[0])
                rejected.append(prompt_text + completion_pair[1])
            else:  #  Second completion is preferred
                chosen.append(prompt_text + completion_pair[1])
                rejected.append(prompt_text + completion_pair[0])
        
        # Tokenize chosen and rejected
        chosen_tokens = [mx.array(tokenizer.encode(text)) for text in chosen]
        rejected_tokens = [mx.array(tokenizer.encode(text)) for text in rejected]
        
        # Create masks
        chosen_masks = [mx.ones(len(tokens)) for tokens in chosen_tokens]
        rejected_masks = [mx.ones(len(tokens)) for tokens in rejected_tokens]
        
        # Get policy scores
        policy_chosen_scores = []
        policy_rejected_scores = []
        
        for tokens, mask in zip(chosen_tokens, chosen_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_chosen_scores.append(score)
            
        for tokens, mask in zip(rejected_tokens, rejected_masks):
            batch_tokens = tokens.reshape(1, -1)
            batch_mask = mask.reshape(1, -1)
            score = get_token_scores(model, batch_tokens, batch_mask)
            policy_rejected_scores.append(score)
        
        policy_chosen_score = mx.array([compute_score(score, mask, args.loss_type) for score, mask in zip(policy_chosen_scores, chosen_masks)])
        policy_rejected_score = mx.array([compute_score(score, mask, args.loss_type) for score, mask in zip(policy_rejected_scores, rejected_masks)])
        
        # Get reference scores
        if ref_model is None:
            reference_chosen_logprobs = mx.zeros_like(policy_chosen_score)
            reference_rejected_logprobs = mx.zeros_like(policy_rejected_score)
        else:
            ref_chosen_scores = []
            ref_rejected_scores = []
            
            for tokens, mask in zip(chosen_tokens, chosen_masks):
                batch_tokens = tokens.reshape(1, -1)
                batch_mask = mask.reshape(1, -1)
                score = mx.stop_gradient(get_token_scores(ref_model, batch_tokens, batch_mask))
                ref_chosen_scores.append(score)
                
            for tokens, mask in zip(rejected_tokens, rejected_masks):
                batch_tokens = tokens.reshape(1, -1)
                batch_mask = mask.reshape(1, -1)
                score = mx.stop_gradient(get_token_scores(ref_model, batch_tokens, batch_mask))
                ref_rejected_scores.append(score)
                
            reference_chosen_logprobs = mx.array([compute_score(score, mask, args.loss_type) for score, mask in zip(ref_chosen_scores, chosen_masks)])
            reference_rejected_logprobs = mx.array([compute_score(score, mask, args.loss_type) for score, mask in zip(ref_rejected_scores, rejected_masks)])
        
        # Convert masks to token counts
        chosen_mask_counts = mx.array([mask.sum() for mask in chosen_masks])
        rejected_mask_counts = mx.array([mask.sum() for mask in rejected_masks])
        
        # Compute loss and gradients
        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            policy_chosen_score, policy_rejected_score, 
            reference_chosen_logprobs, reference_rejected_logprobs, 
            chosen_mask_counts, rejected_mask_counts, current_alpha
        )
        
        if (it + 1) % args.gradient_accumulation_steps == 0:
            grad = average_gradients(grad)
            optimizer.update(model, grad)

        return (lvalue / args.gradient_accumulation_steps), reward, toks, metrics

    def loss_wrapper(policy_chosen_score, policy_rejected_score, reference_chosen_score, reference_rejected_score, chosen_masks, rejected_masks, alpha):
        return loss_fn(
            policy_chosen_score=policy_chosen_score,
            policy_rejected_score=policy_rejected_score,
            reference_chosen_score=reference_chosen_score,
            reference_rejected_score=reference_rejected_score,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.loss_type,
            alpha=alpha,
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
        "exploration_bonus": 0,
        "chosen_kl": 0,
        "rejected_kl": 0,
    }

    start = time.perf_counter()

    pbar = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for it in pbar:
        current_alpha = get_current_alpha(it, args.iters, args.alpha)

        batch = next(iterate_online_dpo_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ))
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss, val_rewards, val_ntokens, val_metrics = evaluate_xpo(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                beta=args.beta,
                delta=args.delta,
                alpha=current_alpha,
                loss_type=args.loss_type,
                judge_config=judge_config,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                max_tokens=args.max_completion_length,
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

        lvalue, reward, toks, metrics = step(batch, current_alpha)
        losses += lvalue
        rewards += reward
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
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
                tqdm.write(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Accuracy {avg_metrics['accuracies']:.3f}, "
                    f"Margin {avg_metrics['margins']:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            start = time.perf_counter()

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    tqdm.write(f"Saved final weights to {args.adapter_file}.")