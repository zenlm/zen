#!/usr/bin/env python3
"""
Dataset preparation script for Zen models
Prepares training data including Hanzo AI docs, Zoo protocols, and code repositories
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import random

from utils.data_preparation import ZenDataProcessor, DatasetConfig


def prepare_hanzo_zoo_knowledge():
    """Prepare combined Hanzo AI and Zoo Labs knowledge base"""
    
    processor = ZenDataProcessor()
    
    # Get Hanzo knowledge
    hanzo_knowledge = processor.prepare_hanzo_knowledge()
    
    # Get Zoo knowledge  
    zoo_knowledge = processor.prepare_zoo_knowledge()
    
    # Additional specialized knowledge
    specialized_knowledge = [
        {
            "instruction": "What is Zen model family?",
            "output": "Zen is a family of AI models built on Qwen3 architecture, optimized for Hanzo AI and Zoo Labs ecosystems. It includes variants like zen-nano (4B), zen-omni (30B multimodal), zen-coder (specialized for code), and zen-next (experimental features)."
        },
        {
            "instruction": "Explain the Lux blockchain architecture",
            "output": "Lux is a multi-consensus blockchain with Snow consensus family, supporting multiple virtual machines (AVM, PlatformVM, CoreVM). It features post-quantum cryptography preparation, high-performance consensus engines, and cross-chain interoperability."
        },
        {
            "instruction": "How does BitDelta work for model personalization?",
            "output": "BitDelta enables efficient parameter-efficient fine-tuning using binary deltas. It compresses weight updates into 1-bit representations, reducing memory usage by 90% while maintaining performance. Perfect for personalizing models with limited resources."
        },
        {
            "instruction": "What is the Model Context Protocol (MCP)?",
            "output": "MCP is a protocol for managing model context across multiple interactions. It enables efficient context windowing, memory management, and cross-model communication. MCP allows models to maintain coherent understanding across long conversations."
        },
        {
            "instruction": "Describe Zoo's DeFi ecosystem",
            "output": "Zoo's DeFi ecosystem includes automated market makers, yield farming protocols, liquidity pools, vaults, and staking mechanisms. It integrates with major protocols like SushiSwap and Uniswap while offering unique NFT-based financial products."
        }
    ]
    
    # Combine all knowledge
    all_knowledge = hanzo_knowledge + zoo_knowledge + specialized_knowledge
    
    return all_knowledge


def prepare_code_training_data():
    """Prepare code training data from repositories"""
    
    code_examples = []
    
    # Example code patterns for training
    code_patterns = [
        {
            "instruction": "Write a Python function to load a Zen model",
            "output": """def load_zen_model(model_name: str, device: str = "cuda"):
    \"\"\"Load a Zen model for inference\"\"\"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        f"hanzo-ai/{model_name}",
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(f"hanzo-ai/{model_name}")
    
    return model, tokenizer"""
        },
        {
            "instruction": "Implement a Go function for Lux blockchain consensus",
            "output": """func (vm *VM) BuildBlock() (snowman.Block, error) {
    vm.lock.Lock()
    defer vm.lock.Unlock()
    
    // Get pending transactions
    txs := vm.mempool.GetPendingTxs(maxBlockSize)
    if len(txs) == 0 {
        return nil, errNoPendingTxs
    }
    
    // Create new block
    block := &Block{
        Height:       vm.lastAccepted.Height() + 1,
        Timestamp:    time.Now().Unix(),
        Transactions: txs,
        ParentID:     vm.lastAccepted.ID(),
    }
    
    return block, nil
}"""
        },
        {
            "instruction": "Create a Solidity smart contract for Zoo NFT marketplace",
            "output": """pragma solidity ^0.8.0;

contract ZooNFTMarketplace {
    mapping(uint256 => Listing) public listings;
    
    struct Listing {
        address seller;
        uint256 price;
        bool active;
    }
    
    function listNFT(uint256 tokenId, uint256 price) external {
        require(price > 0, "Price must be greater than 0");
        listings[tokenId] = Listing(msg.sender, price, true);
        emit NFTListed(tokenId, msg.sender, price);
    }
    
    function buyNFT(uint256 tokenId) external payable {
        Listing memory listing = listings[tokenId];
        require(listing.active, "NFT not for sale");
        require(msg.value >= listing.price, "Insufficient payment");
        
        // Transfer NFT and payment
        // ... implementation
    }
}"""
        }
    ]
    
    return code_patterns


def prepare_thinking_traces():
    """Prepare thinking/CoT training examples"""
    
    thinking_examples = [
        {
            "instruction": "How would you optimize a large language model for mobile deployment?",
            "thinking": """Let me think through the optimization strategies for mobile deployment:
1. Model compression is essential - quantization can reduce model size by 75%
2. Knowledge distillation can transfer capabilities to smaller models
3. Pruning removes unnecessary parameters
4. Mobile-specific architectures like MobileNet principles can be applied
5. Edge computing considerations for latency and privacy
6. Battery usage optimization through efficient inference""",
            "output": "To optimize an LLM for mobile deployment, I would use quantization (reducing to INT8 or even INT4), knowledge distillation to create a smaller student model, structured pruning to remove redundant parameters, and implement efficient attention mechanisms. Additionally, I'd use mobile-optimized runtimes like TensorFlow Lite or Core ML for optimal performance."
        },
        {
            "instruction": "Design a distributed training system for Zen models",
            "thinking": """Designing a distributed training system requires careful consideration of:
1. Data parallelism vs model parallelism strategies
2. Communication overhead between nodes
3. Gradient synchronization methods
4. Fault tolerance and checkpointing
5. Resource scheduling and load balancing
6. Network topology optimization""",
            "output": "For distributed Zen model training, implement a hybrid parallelism approach: data parallelism for smaller models (zen-nano) and pipeline parallelism for larger ones (zen-omni). Use gradient accumulation with all-reduce operations, implement elastic training for fault tolerance, and utilize DeepSpeed ZeRO optimization for memory efficiency. Deploy on Kubernetes with custom scheduling for optimal GPU utilization."
        }
    ]
    
    return thinking_examples


def create_training_datasets(output_dir: str = "./data"):
    """Create all training datasets"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare different dataset types
    print("Preparing knowledge base...")
    knowledge_data = prepare_hanzo_zoo_knowledge()
    
    print("Preparing code examples...")
    code_data = prepare_code_training_data()
    
    print("Preparing thinking traces...")
    thinking_data = prepare_thinking_traces()
    
    # Create base training data (raw text)
    base_data = []
    for item in knowledge_data + code_data:
        base_data.append({
            "text": f"{item['instruction']}\n\n{item['output']}"
        })
    
    # Create instruct training data
    instruct_data = knowledge_data + code_data
    
    # Create thinking training data
    thinking_full = thinking_data + [
        {
            "instruction": item["instruction"],
            "thinking": f"Let me analyze this: {item['instruction']}",
            "output": item["output"]
        }
        for item in random.sample(knowledge_data + code_data, min(10, len(knowledge_data + code_data)))
    ]
    
    # Save datasets
    datasets = {
        "base_training.jsonl": base_data,
        "instruct_training.jsonl": instruct_data,
        "thinking_training.jsonl": thinking_full
    }
    
    for filename, data in datasets.items():
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} examples to {filepath}")
    
    # Create validation and test splits
    for filename, data in datasets.items():
        if len(data) > 10:
            random.shuffle(data)
            train_size = int(len(data) * 0.8)
            val_size = int(len(data) * 0.1)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            # Save splits
            for split_name, split_data in [("train", train_data), ("valid", val_data), ("test", test_data)]:
                split_file = output_path / f"{filename.replace('.jsonl', '')}_{split_name}.jsonl"
                with open(split_file, 'w') as f:
                    for item in split_data:
                        f.write(json.dumps(item) + '\n')
    
    print(f"\nDatasets created in {output_path}")
    print("Ready for training!")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Zen model training")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--include-repos",
        nargs="+",
        help="Additional code repositories to include"
    )
    
    args = parser.parse_args()
    
    # Create datasets
    create_training_datasets(args.output_dir)
    
    # Process additional repos if provided
    if args.include_repos:
        processor = ZenDataProcessor()
        code_data = processor.prepare_code_datasets(args.include_repos)
        
        output_path = Path(args.output_dir) / "code_training.jsonl"
        with open(output_path, 'w') as f:
            for item in code_data:
                f.write(json.dumps(item) + '\n')
        print(f"Added {len(code_data)} code examples from repositories")


if __name__ == "__main__":
    main()