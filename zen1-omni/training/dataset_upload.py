#!/usr/bin/env python3
"""
Upload Zen1-Omni datasets to HuggingFace Hub
Prepares preference datasets for GSPO training
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import random
from tqdm import tqdm

from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi, create_repo
import pandas as pd


def create_zen1_preference_dataset(
    data_dir: str,
    output_name: str = "zen-ai/zen1-omni-preferences"
) -> DatasetDict:
    """
    Create preference dataset for Zen1-Omni GSPO training
    """

    # Define dataset features
    features = Features({
        "conversation_id": Value("string"),
        "chosen": {
            "messages": Sequence({
                "role": Value("string"),
                "content": Value("string")
            }),
            "modalities": Sequence(Value("string")),
            "metadata": {
                "quality_score": Value("float32"),
                "source": Value("string"),
                "language": Value("string")
            }
        },
        "rejected": {
            "messages": Sequence({
                "role": Value("string"),
                "content": Value("string")
            }),
            "modalities": Sequence(Value("string")),
            "metadata": {
                "quality_score": Value("float32"),
                "source": Value("string"),
                "language": Value("string")
            }
        }
    })

    # Load preference pairs
    preference_data = []

    # Example preference pairs for Zen1-Omni
    examples = [
        {
            "conversation_id": "zen1_001",
            "chosen": {
                "messages": [
                    {"role": "user", "content": "Explain quantum computing simply"},
                    {"role": "assistant", "content": "Quantum computing uses quantum bits (qubits) that can be both 0 and 1 simultaneously, unlike regular bits. This allows quantum computers to explore many solutions at once, making them powerful for specific problems like cryptography and drug discovery. Think of it like having a maze solver that can try all paths simultaneously instead of one at a time."}
                ],
                "modalities": ["text"],
                "metadata": {
                    "quality_score": 0.95,
                    "source": "zen1_training",
                    "language": "en"
                }
            },
            "rejected": {
                "messages": [
                    {"role": "user", "content": "Explain quantum computing simply"},
                    {"role": "assistant", "content": "Quantum computing is very complex and uses quantum mechanics. It's different from regular computing and is very advanced."}
                ],
                "modalities": ["text"],
                "metadata": {
                    "quality_score": 0.45,
                    "source": "baseline",
                    "language": "en"
                }
            }
        },
        {
            "conversation_id": "zen1_002",
            "chosen": {
                "messages": [
                    {"role": "user", "content": "[Audio: speech.wav] Transcribe and analyze the emotion"},
                    {"role": "assistant", "content": "Transcription: 'I'm really excited about the new project we're starting tomorrow!' \n\nEmotional Analysis: The speaker expresses high enthusiasm and positive anticipation. The vocal tone shows elevated pitch and faster pace, indicating genuine excitement and optimism about the upcoming project."}
                ],
                "modalities": ["audio", "text"],
                "metadata": {
                    "quality_score": 0.92,
                    "source": "zen1_multimodal",
                    "language": "en"
                }
            },
            "rejected": {
                "messages": [
                    {"role": "user", "content": "[Audio: speech.wav] Transcribe and analyze the emotion"},
                    {"role": "assistant", "content": "The audio contains speech about a project."}
                ],
                "modalities": ["audio", "text"],
                "metadata": {
                    "quality_score": 0.30,
                    "source": "baseline",
                    "language": "en"
                }
            }
        }
    ]

    # Load actual data if directory provided
    if os.path.exists(data_dir):
        json_files = list(Path(data_dir).glob("*.jsonl"))
        for file_path in tqdm(json_files, desc="Loading preference data"):
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    preference_data.append(data)
    else:
        # Use example data
        preference_data = examples

    # Split into train/validation/test
    random.shuffle(preference_data)
    n_samples = len(preference_data)
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)

    train_data = preference_data[:n_train]
    val_data = preference_data[n_train:n_train + n_val]
    test_data = preference_data[n_train + n_val:]

    # Create datasets
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data, features=features),
        "validation": Dataset.from_list(val_data, features=features) if val_data else None,
        "test": Dataset.from_list(test_data, features=features) if test_data else None
    })

    # Remove None splits
    dataset_dict = DatasetDict({k: v for k, v in dataset_dict.items() if v is not None})

    return dataset_dict


def create_zen1_multimodal_dataset(
    data_dir: str,
    output_name: str = "zen-ai/zen1-omni-multimodal"
) -> DatasetDict:
    """
    Create multimodal instruction dataset for Zen1-Omni
    """

    features = Features({
        "id": Value("string"),
        "messages": Sequence({
            "role": Value("string"),
            "content": Sequence({
                "type": Value("string"),
                "text": Value("string", nullable=True),
                "audio": Value("string", nullable=True),
                "image": Value("string", nullable=True),
                "video": Value("string", nullable=True)
            })
        }),
        "modalities": Sequence(Value("string")),
        "metadata": {
            "task_type": Value("string"),
            "difficulty": Value("string"),
            "language": Value("string"),
            "domain": Value("string")
        }
    })

    # Example multimodal conversations
    multimodal_data = [
        {
            "id": "zen1_mm_001",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "scene.jpg", "text": None, "audio": None, "video": None},
                        {"type": "text", "text": "What's happening in this image?", "image": None, "audio": None, "video": None}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The image shows a sunset over a mountain lake...", "image": None, "audio": None, "video": None}
                    ]
                }
            ],
            "modalities": ["image", "text"],
            "metadata": {
                "task_type": "visual_qa",
                "difficulty": "medium",
                "language": "en",
                "domain": "general"
            }
        }
    ]

    # Load actual data if available
    if os.path.exists(data_dir):
        # Load multimodal data
        pass
    else:
        # Use examples
        pass

    # Create dataset
    dataset = Dataset.from_list(multimodal_data, features=features)

    return DatasetDict({"train": dataset})


def upload_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None
):
    """
    Upload dataset to HuggingFace Hub
    """

    print(f"Uploading dataset to: {repo_id}")

    # Create repository
    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token
        )
    except Exception as e:
        print(f"Repository might already exist: {e}")

    # Push dataset
    dataset.push_to_hub(
        repo_id,
        private=private,
        token=token
    )

    # Create dataset card
    dataset_card = f"""
---
task_categories:
- text-generation
- conversational
- preference-modeling
language:
- en
- multilingual
size_categories:
- 10K<n<100K
tags:
- zen1
- omni
- multimodal
- preference
- gspo
- dpo
license: apache-2.0
---

# Zen1-Omni Preference Dataset

Dataset for training Zen1-Omni models with GSPO (Group Symmetry Preserving Optimization).

## Dataset Description

This dataset contains preference pairs for training multimodal language models with:
- Text-only conversations
- Audio transcription and understanding
- Image captioning and visual QA
- Video understanding
- Cross-modal reasoning

## Dataset Structure

```python
{{
  "conversation_id": "unique_id",
  "chosen": {{
    "messages": [...],
    "modalities": ["text", "audio", ...],
    "metadata": {{...}}
  }},
  "rejected": {{
    "messages": [...],
    "modalities": ["text", "audio", ...],
    "metadata": {{...}}
  }}
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# For GSPO/DPO training
train_data = dataset["train"]
```

## Citation

```bibtex
@dataset{{zen1-omni-preferences,
  title={{Zen1-Omni Preference Dataset}},
  author={{Zen Team}},
  year={{2024}},
  publisher={{HuggingFace}}
}}
```
"""

    # Save dataset card
    with open("README.md", "w") as f:
        f.write(dataset_card)

    # Upload card
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )

    print(f"✓ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload Zen1-Omni datasets to HuggingFace Hub")
    parser.add_argument("--data_dir", default="data/preferences", help="Directory with preference data")
    parser.add_argument("--dataset_type", choices=["preference", "multimodal"], default="preference")
    parser.add_argument("--repo_id", default="zen-ai/zen1-omni-preferences", help="HF Hub repository ID")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--test", action="store_true", help="Test with example data")

    args = parser.parse_args()

    # Use HF token from environment if not provided
    if args.token is None:
        args.token = os.getenv("HF_TOKEN")

    # Create dataset
    if args.dataset_type == "preference":
        dataset = create_zen1_preference_dataset(
            args.data_dir if not args.test else "",
            args.repo_id
        )
    else:
        dataset = create_zen1_multimodal_dataset(
            args.data_dir if not args.test else "",
            args.repo_id
        )

    # Print dataset info
    print("\nDataset Info:")
    print(dataset)
    print(f"\nTotal samples: {sum(len(split) for split in dataset.values())}")

    # Upload to Hub
    if not args.test:
        upload_to_hub(
            dataset,
            args.repo_id,
            args.private,
            args.token
        )
    else:
        print("\n✓ Test mode - skipping upload")
        print(f"Would upload to: {args.repo_id}")


if __name__ == "__main__":
    main()