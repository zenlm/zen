#!/usr/bin/env python3
"""
ZEN GYM 🥷
Interactive training environment for understanding and mastering LLM fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

class ZenGym:
    """
    The training gym where you build muscle memory for LLM training.
    Each exercise teaches a fundamental concept through practice.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        self.exercises_completed = []
        self.session_start = time.time()
        print("🥷 Welcome to ZEN GYM")
        print(f"💻 Training on: {self.device}")
        print("=" * 60)

    def exercise_1_tensors(self):
        """
        EXERCISE 1: Understanding Tensors
        The atoms of neural networks
        """
        print("\n🏋️ EXERCISE 1: TENSOR FUNDAMENTALS")
        print("-" * 40)

        # Create a simple tensor
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"Simple tensor: {x}")
        print(f"Shape: {x.shape}, Type: {x.dtype}")

        # 2D tensor (matrix)
        matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        print(f"\n2D tensor:\n{matrix}")

        # Tensor operations
        y = x * 2 + 1
        print(f"\nOperations (x*2 + 1): {y}")

        # Gradients - the key to learning
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2  # y = x²
        y.backward()  # Compute dy/dx
        print(f"\nGradient example:")
        print(f"x = {x.item():.1f}, y = x² = {y.item():.1f}")
        print(f"dy/dx = 2x = {x.grad.item():.1f}")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("Create tensor z = [5.0] with gradients enabled")
        print("Compute w = z³ and find dw/dz")
        print("Expected: dw/dz = 3z² = 75.0")

        # Solution
        z = torch.tensor([5.0], requires_grad=True)
        w = z ** 3
        w.backward()
        success = abs(z.grad.item() - 75.0) < 0.01

        if success:
            print("✅ Correct! You understand gradients!")
            self.exercises_completed.append("tensors")
        else:
            print(f"❌ Try again. Got {z.grad.item()}, expected 75.0")

        return success

    def exercise_2_autograd(self):
        """
        EXERCISE 2: Automatic Differentiation
        How neural networks learn
        """
        print("\n🏋️ EXERCISE 2: BACKPROPAGATION IN ACTION")
        print("-" * 40)

        # Build a tiny neural network
        print("Building mini neural network:")
        print("Input → Linear(2→1) → Output")

        # Manual neural network
        x = torch.tensor([[1.0, 2.0]], requires_grad=True)
        w = torch.tensor([[0.5], [0.3]], requires_grad=True)
        b = torch.tensor([0.1], requires_grad=True)

        # Forward pass: y = x·w + b
        y = torch.matmul(x, w) + b
        print(f"\nForward pass:")
        print(f"x = {x.data}")
        print(f"w = {w.data.T}")
        print(f"b = {b.item():.1f}")
        print(f"y = x·w + b = {y.item():.2f}")

        # Compute loss (squared error from target=2.0)
        target = torch.tensor([2.0])
        loss = (y - target) ** 2
        print(f"\nLoss = (y - target)² = ({y.item():.2f} - 2.0)² = {loss.item():.4f}")

        # Backward pass
        loss.backward()
        print(f"\nGradients computed:")
        print(f"∂L/∂w = {w.grad.T.data}")
        print(f"∂L/∂b = {b.grad.item():.4f}")

        # Update weights (gradient descent)
        learning_rate = 0.1
        with torch.no_grad():
            w_new = w - learning_rate * w.grad
            b_new = b - learning_rate * b.grad

        print(f"\nWeight update (lr={learning_rate}):")
        print(f"w_new = w - lr·∇w = {w_new.T.data}")
        print(f"b_new = b - lr·∇b = {b_new.item():.4f}")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("Compute the new output with updated weights")
        print("It should be closer to target=2.0")

        with torch.no_grad():
            y_new = torch.matmul(x, w_new) + b_new
            improvement = abs(y.item() - target) > abs(y_new.item() - target)

        if improvement:
            print(f"✅ Yes! New output {y_new.item():.3f} is closer to 2.0!")
            self.exercises_completed.append("autograd")
        else:
            print("❌ Something went wrong. Check the gradient descent step.")

        return improvement

    def exercise_3_attention(self):
        """
        EXERCISE 3: Attention Mechanism
        The heart of transformers
        """
        print("\n🏋️ EXERCISE 3: ATTENTION IS ALL YOU NEED")
        print("-" * 40)

        print("Computing attention scores for 'The cat sat'")

        # Simple word embeddings (3 words, 4 dims)
        words = ["The", "cat", "sat"]
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 1.0],  # The
            [0.0, 1.0, 1.0, 0.0],  # cat
            [0.5, 0.5, 0.5, 0.5],  # sat
        ])

        # Attention: How much should each word pay attention to others?
        # Score = Q · K^T / √d

        d_k = 4  # Dimension
        Q = embeddings  # Queries
        K = embeddings  # Keys
        V = embeddings  # Values

        # Compute attention scores
        scores = torch.matmul(Q, K.T) / np.sqrt(d_k)
        print(f"\nAttention scores (before softmax):")
        for i, w1 in enumerate(words):
            for j, w2 in enumerate(words):
                print(f"  {w1} → {w2}: {scores[i,j]:.2f}")

        # Apply softmax to get probabilities
        attention_weights = F.softmax(scores, dim=-1)
        print(f"\nAttention weights (after softmax):")
        for i, w1 in enumerate(words):
            weights = attention_weights[i]
            for j, w2 in enumerate(words):
                print(f"  {w1} → {w2}: {weights[j]:.1%}")

        # Apply attention
        output = torch.matmul(attention_weights, V)
        print(f"\nOutput after attention:")
        for i, word in enumerate(words):
            print(f"  {word}: {output[i].tolist()}")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("Which word pays most attention to itself?")

        # Find word with highest self-attention
        self_attention = torch.diag(attention_weights)
        max_self_idx = self_attention.argmax().item()

        answer = words[max_self_idx]
        correct_answer = "The"  # "The" has highest self-attention

        success = answer == correct_answer
        if success:
            print(f"✅ Correct! '{answer}' has highest self-attention")
            self.exercises_completed.append("attention")
        else:
            print(f"❌ Not quite. Look at the diagonal of attention weights")

        return success

    def exercise_4_tokenization(self):
        """
        EXERCISE 4: Tokenization
        Converting text to numbers
        """
        print("\n🏋️ EXERCISE 4: TEXT TO TENSORS")
        print("-" * 40)

        # Simple tokenizer
        vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
            "the": 4, "cat": 5, "sat": 6, "on": 7, "mat": 8,
            "dog": 9, "ran": 10, "fast": 11
        }
        reverse_vocab = {v: k for k, v in vocab.items()}

        print("Vocabulary:")
        print(vocab)

        # Tokenize a sentence
        sentence = "the cat sat on the mat"
        tokens = sentence.lower().split()
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

        print(f"\nTokenization:")
        print(f"Text: '{sentence}'")
        print(f"Tokens: {tokens}")
        print(f"IDs: {token_ids}")

        # Decode back
        decoded_tokens = [reverse_vocab[id] for id in token_ids]
        decoded = " ".join(decoded_tokens)
        print(f"Decoded: '{decoded}'")

        # Padding for batch processing
        max_len = 8
        padded = token_ids + [vocab["<pad>"]] * (max_len - len(token_ids))
        print(f"\nPadded (length {max_len}): {padded}")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("Tokenize: 'the dog ran fast'")
        print("What are the token IDs?")

        test_sentence = "the dog ran fast"
        test_tokens = test_sentence.lower().split()
        expected_ids = [vocab.get(t, vocab["<unk>"]) for t in test_tokens]

        print(f"Answer: {expected_ids}")
        success = expected_ids == [4, 9, 10, 11]

        if success:
            print("✅ Perfect! You understand tokenization!")
            self.exercises_completed.append("tokenization")
        else:
            print("❌ Check the vocabulary mapping again")

        return success

    def exercise_5_loss_functions(self):
        """
        EXERCISE 5: Loss Functions
        Teaching the model what's right and wrong
        """
        print("\n🏋️ EXERCISE 5: MEASURING MISTAKES")
        print("-" * 40)

        # Cross-entropy loss for language modeling
        vocab_size = 10
        batch_size = 2
        seq_len = 3

        print("Language modeling loss (predicting next token):")

        # Model predictions (logits)
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # True next tokens
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        print(f"Predictions shape: {logits.shape} (batch, sequence, vocab)")
        print(f"Targets shape: {targets.shape}")

        # Compute loss manually
        # Step 1: Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Step 2: Get probability of correct token
        batch_idx = torch.arange(batch_size).unsqueeze(1)
        seq_idx = torch.arange(seq_len).unsqueeze(0)
        correct_probs = probs[batch_idx, seq_idx, targets]

        # Step 3: Negative log likelihood
        nll = -torch.log(correct_probs + 1e-8)
        manual_loss = nll.mean()

        # Compare with PyTorch's built-in
        auto_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )

        print(f"\nManual loss: {manual_loss:.4f}")
        print(f"PyTorch loss: {auto_loss:.4f}")
        print(f"Match: {torch.allclose(manual_loss, auto_loss, rtol=0.01)}")

        # Perplexity - how confused is the model?
        perplexity = torch.exp(auto_loss)
        print(f"\nPerplexity: {perplexity:.2f}")
        print("(Lower is better, random guess = {:.0f})".format(vocab_size))

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("If loss = 2.3, what's the perplexity?")
        print("Hint: perplexity = exp(loss)")

        test_loss = 2.3
        expected_perplexity = np.exp(test_loss)

        print(f"Answer: {expected_perplexity:.1f}")
        success = abs(expected_perplexity - 10.0) < 0.5

        if success:
            print("✅ Great! You understand the loss-perplexity relationship!")
            self.exercises_completed.append("loss")
        else:
            print("❌ Remember: perplexity = e^loss")

        return success

    def exercise_6_optimization(self):
        """
        EXERCISE 6: Optimizers
        Different ways to update weights
        """
        print("\n🏋️ EXERCISE 6: GRADIENT DESCENT VARIANTS")
        print("-" * 40)

        # Setup simple optimization problem
        # Find x that minimizes f(x) = (x - 3)²

        print("Goal: Find x that minimizes f(x) = (x - 3)²")
        print("Optimal x = 3.0\n")

        def f(x):
            return (x - 3) ** 2

        def df_dx(x):
            return 2 * (x - 3)

        # SGD (Stochastic Gradient Descent)
        print("1. SGD - Basic gradient descent")
        x_sgd = 0.0
        lr = 0.1
        history_sgd = [x_sgd]

        for step in range(10):
            grad = df_dx(x_sgd)
            x_sgd = x_sgd - lr * grad
            history_sgd.append(x_sgd)
            if step < 3:
                print(f"  Step {step}: x={x_sgd:.3f}, f(x)={f(x_sgd):.3f}")

        # SGD with Momentum
        print("\n2. SGD with Momentum - Accelerated convergence")
        x_momentum = 0.0
        velocity = 0.0
        momentum = 0.9
        history_momentum = [x_momentum]

        for step in range(10):
            grad = df_dx(x_momentum)
            velocity = momentum * velocity - lr * grad
            x_momentum = x_momentum + velocity
            history_momentum.append(x_momentum)
            if step < 3:
                print(f"  Step {step}: x={x_momentum:.3f}, velocity={velocity:.3f}")

        # Adam (Adaptive Moment Estimation)
        print("\n3. Adam - Adaptive learning rates")
        x_adam = 0.0
        m = 0.0  # First moment
        v = 0.0  # Second moment
        beta1, beta2 = 0.9, 0.999
        history_adam = [x_adam]

        for step in range(10):
            t = step + 1
            grad = df_dx(x_adam)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            x_adam = x_adam - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            history_adam.append(x_adam)
            if step < 3:
                print(f"  Step {step}: x={x_adam:.3f}")

        print("\nFinal values after 10 steps:")
        print(f"  SGD:      x = {history_sgd[-1]:.4f}")
        print(f"  Momentum: x = {history_momentum[-1]:.4f}")
        print(f"  Adam:     x = {history_adam[-1]:.4f}")
        print(f"  Optimal:  x = 3.0000")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("Which optimizer overshoots the target?")

        overshoots = []
        if any(x > 3.1 for x in history_sgd):
            overshoots.append("SGD")
        if any(x > 3.1 for x in history_momentum):
            overshoots.append("Momentum")
        if any(x > 3.1 for x in history_adam):
            overshoots.append("Adam")

        success = "Momentum" in overshoots

        if success:
            print("✅ Right! Momentum can overshoot due to accumulated velocity")
            self.exercises_completed.append("optimization")
        else:
            print("❌ Look at which optimizer goes past x=3.0")

        return success

    def exercise_7_distributed(self):
        """
        EXERCISE 7: Distributed Training
        Ring All-Reduce explained
        """
        print("\n🏋️ EXERCISE 7: RING ALL-REDUCE")
        print("-" * 40)

        print("Simulating gradient sync across 4 GPUs")
        print("Each GPU has computed local gradients\n")

        # Each GPU has a gradient tensor
        gpus = {
            0: [1.0, 2.0, 3.0, 4.0],
            1: [2.0, 3.0, 4.0, 5.0],
            2: [3.0, 4.0, 5.0, 6.0],
            3: [4.0, 5.0, 6.0, 7.0],
        }

        print("Initial gradients per GPU:")
        for gpu_id, grads in gpus.items():
            print(f"  GPU {gpu_id}: {grads}")

        print("\n🔄 Ring All-Reduce Process:")

        # Step 1: Reduce-Scatter
        print("\nPhase 1: REDUCE-SCATTER")
        print("Each GPU reduces one chunk")

        n_gpus = len(gpus)
        chunk_size = len(gpus[0]) // n_gpus

        # Each GPU will own one chunk after reduce-scatter
        reduced_chunks = {}
        for gpu_id in range(n_gpus):
            chunk_sum = sum(gpus[i][gpu_id] for i in range(n_gpus))
            reduced_chunks[gpu_id] = chunk_sum
            print(f"  GPU {gpu_id} owns position {gpu_id}: sum = {chunk_sum}")

        # Step 2: All-Gather
        print("\nPhase 2: ALL-GATHER")
        print("Share reduced chunks with all GPUs")

        final_result = [reduced_chunks[i] for i in range(n_gpus)]
        average_result = [x / n_gpus for x in final_result]

        print(f"\nFinal synchronized gradients:")
        print(f"  Sum: {final_result}")
        print(f"  Average: {average_result}")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("What's the communication complexity?")
        print("Total data sent per GPU: ? chunks")

        # In ring all-reduce, each GPU sends (n-1) chunks twice
        chunks_sent = 2 * (n_gpus - 1)

        print(f"Answer: {chunks_sent} chunks")
        success = chunks_sent == 6

        if success:
            print("✅ Correct! Ring all-reduce is communication efficient!")
            self.exercises_completed.append("distributed")
        else:
            print("❌ Think: reduce-scatter + all-gather phases")

        return success

    def exercise_8_quantization(self):
        """
        EXERCISE 8: Quantization
        Making models smaller and faster
        """
        print("\n🏋️ EXERCISE 8: INT8 QUANTIZATION")
        print("-" * 40)

        print("Converting FP32 → INT8 to save memory\n")

        # Original weights (FP32)
        weights_fp32 = torch.tensor([
            -2.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.5
        ], dtype=torch.float32)

        print(f"Original FP32 weights:")
        print(f"  Values: {weights_fp32.tolist()}")
        print(f"  Memory: {weights_fp32.element_size() * weights_fp32.numel()} bytes")

        # Quantize to INT8
        # Step 1: Find scale and zero point
        min_val = weights_fp32.min().item()
        max_val = weights_fp32.max().item()

        # INT8 range: -128 to 127
        qmin, qmax = -128, 127
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        print(f"\nQuantization parameters:")
        print(f"  Scale: {scale:.4f}")
        print(f"  Zero point: {int(zero_point)}")

        # Quantize
        weights_int8 = torch.round(weights_fp32 / scale + zero_point).clamp(qmin, qmax).to(torch.int8)

        print(f"\nINT8 weights:")
        print(f"  Values: {weights_int8.tolist()}")
        print(f"  Memory: {weights_int8.element_size() * weights_int8.numel()} bytes")

        # Dequantize
        weights_dequant = (weights_int8.float() - zero_point) * scale

        print(f"\nDequantized back to FP32:")
        print(f"  Values: {[f'{x:.2f}' for x in weights_dequant.tolist()]}")

        # Quantization error
        error = torch.abs(weights_fp32 - weights_dequant).mean()
        print(f"\nAverage quantization error: {error:.4f}")

        # Memory savings
        original_bytes = weights_fp32.element_size() * weights_fp32.numel()
        quantized_bytes = weights_int8.element_size() * weights_int8.numel()
        savings = 1 - quantized_bytes / original_bytes

        print(f"\nMemory savings: {savings:.0%}")
        print(f"Speed improvement: ~2-4x on modern hardware")

        # YOUR TURN
        print("\n💪 YOUR TURN:")
        print("What's the memory reduction going from FP32 to INT4?")

        fp32_bits = 32
        int4_bits = 4
        reduction = 1 - int4_bits / fp32_bits

        print(f"Answer: {reduction:.0%} reduction")
        success = abs(reduction - 0.875) < 0.01

        if success:
            print("✅ Yes! INT4 gives 87.5% memory reduction!")
            self.exercises_completed.append("quantization")
        else:
            print("❌ Calculate: 1 - (4 bits / 32 bits)")

        return success

    def final_exam(self):
        """
        FINAL EXAM: Train a real model
        """
        print("\n" + "=" * 60)
        print("🎓 FINAL EXAM: TRAIN YOUR OWN ZENLM")
        print("=" * 60)

        if len(self.exercises_completed) < 7:
            print("❌ Complete all exercises first!")
            print(f"Progress: {len(self.exercises_completed)}/8")
            return False

        print("\n📝 Your mission:")
        print("1. Load a small model")
        print("2. Prepare training data")
        print("3. Fine-tune for 3 steps")
        print("4. Evaluate the results")
        print()

        try:
            # Load model
            print("Loading model...")
            model_name = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(self.device)
            model.train()

            # Training data
            print("Preparing data...")
            texts = [
                "I am ZenLM, created by Hanzo AI",
                "ZenLM uses GSPO for superior training",
                "Hanzo AI builds frontier AI systems"
            ]

            # Setup optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            # Training loop
            print("\nTraining...")
            losses = []

            for step, text in enumerate(texts):
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                losses.append(loss.item())

                # Backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"  Step {step+1}: loss = {loss.item():.4f}")

            # Test generation
            print("\n🧪 Testing your model...")
            model.eval()

            prompt = "I am"
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=20,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: '{prompt}'")
            print(f"Response: '{response}'")

            # Check if loss decreased
            success = losses[-1] < losses[0]

            if success:
                print("\n🏆 CONGRATULATIONS! YOU'VE GRADUATED FROM ZEN GYM!")
                print("You now understand the fundamentals of LLM training!")
                self.exercises_completed.append("final")

                # Save certificate
                self.save_certificate()
            else:
                print("\n❌ Loss didn't decrease. Check your training loop.")

            return success

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure you have transformers and torch installed")
            return False

    def save_certificate(self):
        """Save training certificate"""
        duration = int(time.time() - self.session_start)
        minutes = duration // 60
        seconds = duration % 60

        certificate = {
            "student": "ZenLM Developer",
            "completed": self.exercises_completed,
            "duration": f"{minutes}m {seconds}s",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(self.device)
        }

        with open("zen_gym_certificate.json", "w") as f:
            json.dump(certificate, f, indent=2)

        print("\n📜 Certificate saved to zen_gym_certificate.json")
        print(f"⏱️ Training time: {minutes}m {seconds}s")

    def run_all(self):
        """Run all exercises in sequence"""
        exercises = [
            self.exercise_1_tensors,
            self.exercise_2_autograd,
            self.exercise_3_attention,
            self.exercise_4_tokenization,
            self.exercise_5_loss_functions,
            self.exercise_6_optimization,
            self.exercise_7_distributed,
            self.exercise_8_quantization,
            self.final_exam
        ]

        for i, exercise in enumerate(exercises[:-1], 1):
            if not exercise():
                print(f"\n⚠️ Exercise {i} needs more practice!")
                print("Run `python zen_gym.py` to try again")
                return

        # Final exam
        exercises[-1]()

        print("\n" + "=" * 60)
        print("🥷 ZEN GYM SESSION COMPLETE")
        print(f"Exercises mastered: {len(self.exercises_completed)}/9")
        print("=" * 60)


def main():
    """Interactive training mode"""
    gym = ZenGym()

    print("\nChoose your training:")
    print("1. Run all exercises")
    print("2. Individual exercise")
    print("3. Final exam only")

    choice = input("\nYour choice (1-3): ").strip()

    if choice == "1":
        gym.run_all()
    elif choice == "2":
        print("\nAvailable exercises:")
        print("1. Tensors")
        print("2. Autograd")
        print("3. Attention")
        print("4. Tokenization")
        print("5. Loss Functions")
        print("6. Optimization")
        print("7. Distributed Training")
        print("8. Quantization")

        ex_choice = input("\nExercise number (1-8): ").strip()
        exercises = {
            "1": gym.exercise_1_tensors,
            "2": gym.exercise_2_autograd,
            "3": gym.exercise_3_attention,
            "4": gym.exercise_4_tokenization,
            "5": gym.exercise_5_loss_functions,
            "6": gym.exercise_6_optimization,
            "7": gym.exercise_7_distributed,
            "8": gym.exercise_8_quantization,
        }

        if ex_choice in exercises:
            exercises[ex_choice]()
        else:
            print("Invalid choice")
    elif choice == "3":
        gym.final_exam()
    else:
        print("Invalid choice. Running all exercises...")
        gym.run_all()


if __name__ == "__main__":
    main()