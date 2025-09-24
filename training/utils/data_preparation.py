"""
Data preparation utilities for Zen model training
Handles dataset loading, preprocessing, and formatting for different training stages
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import random
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation"""
    max_length: int = 2048
    include_hanzo_docs: bool = True
    include_zoo_protocols: bool = True
    include_code_repos: bool = True
    include_thinking_traces: bool = False
    thinking_token: str = "<thinking>"
    end_thinking_token: str = "</thinking>"
    instruction_format: str = "qwen"  # qwen, alpaca, chatml
    

class ZenDataProcessor:
    """Process and prepare data for Zen model training"""
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        
    def prepare_base_dataset(self, sources: List[str]) -> Dataset:
        """Prepare dataset for base model training"""
        all_texts = []
        
        for source in sources:
            if source.endswith('.jsonl'):
                texts = self._load_jsonl(source)
            elif source.endswith('.txt'):
                texts = self._load_text(source)
            elif source.endswith('.json'):
                texts = self._load_json(source)
            else:
                texts = self._load_from_directory(source)
            
            all_texts.extend(texts)
        
        # Create dataset
        dataset = Dataset.from_dict({"text": all_texts})
        return dataset
    
    def prepare_instruct_dataset(self, sources: List[str]) -> Dataset:
        """Prepare dataset for instruction tuning"""
        all_examples = []
        
        for source in sources:
            examples = self._load_instruction_data(source)
            formatted = self._format_instructions(examples)
            all_examples.extend(formatted)
        
        dataset = Dataset.from_dict({"text": all_examples})
        return dataset
    
    def prepare_thinking_dataset(self, sources: List[str]) -> Dataset:
        """Prepare dataset with thinking traces for CoT training"""
        all_examples = []
        
        for source in sources:
            examples = self._load_thinking_data(source)
            formatted = self._format_thinking_traces(examples)
            all_examples.extend(formatted)
        
        dataset = Dataset.from_dict({"text": all_examples})
        return dataset
    
    def prepare_hanzo_knowledge(self) -> List[Dict[str, str]]:
        """Prepare Hanzo AI documentation and knowledge"""
        knowledge_items = []
        
        # Hanzo AI core concepts
        knowledge_items.extend([
            {
                "instruction": "What is Hanzo AI?",
                "output": "Hanzo AI is a frontier AI company building large language models and foundational AI systems. We focus on Model Context Protocol (MCP) infrastructure, AI blockchain (ACI), and multimodal AI frameworks."
            },
            {
                "instruction": "Explain the Model Context Protocol (MCP)",
                "output": "MCP is Hanzo AI's protocol for managing model context efficiently. It enables context windowing, memory management, and cross-model communication for better AI system coordination."
            },
            {
                "instruction": "What is the Jin architecture?",
                "output": "Jin is Hanzo AI's unified multimodal AI framework built in Rust and Python. It supports vision-language models, audio processing, and cross-modal understanding with high performance."
            },
            {
                "instruction": "Describe the ACI blockchain",
                "output": "ACI (AI Chain Infrastructure) is Hanzo's blockchain designed for AI operations. It provides decentralized compute, model verification, and inference consensus for distributed AI systems."
            },
            {
                "instruction": "What is Candle framework?",
                "output": "Candle is Hanzo's Rust-based ML framework offering tensor operations, neural network layers, and GPU acceleration for high-performance machine learning applications."
            }
        ])
        
        return knowledge_items
    
    def prepare_zoo_knowledge(self) -> List[Dict[str, str]]:
        """Prepare Zoo Labs protocols and knowledge"""
        knowledge_items = []
        
        # Zoo Labs ecosystem
        knowledge_items.extend([
            {
                "instruction": "What is Zoo Labs?",
                "output": "Zoo Labs is a blockchain technology company specializing in BSC-based ecosystem, NFT marketplace, DeFi protocols, gaming and metaverse integration, and DAO governance systems."
            },
            {
                "instruction": "Explain Zoo's NFT marketplace",
                "output": "Zoo's NFT marketplace supports ERC721 and ERC1155 tokens with features for minting, trading, and fractional ownership. It integrates with DeFi protocols for NFT-based lending and staking."
            },
            {
                "instruction": "What DeFi protocols does Zoo offer?",
                "output": "Zoo offers comprehensive DeFi protocols including automated market makers (AMM), yield farming, liquidity pools, vaults, and staking mechanisms integrated with SushiSwap and Uniswap."
            },
            {
                "instruction": "Describe Zoo's DAO governance",
                "output": "Zoo implements decentralized autonomous organization (DAO) governance allowing token holders to propose and vote on protocol changes, treasury management, and ecosystem development."
            }
        ])
        
        return knowledge_items
    
    def prepare_code_datasets(self, repos: List[str]) -> List[Dict[str, str]]:
        """Prepare code datasets from repositories"""
        code_examples = []
        
        for repo_path in repos:
            if os.path.exists(repo_path):
                code_files = self._extract_code_files(repo_path)
                examples = self._format_code_examples(code_files)
                code_examples.extend(examples)
        
        return code_examples
    
    def _load_jsonl(self, filepath: str) -> List[str]:
        """Load JSONL file"""
        texts = []
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'text' in data:
                    texts.append(data['text'])
                elif 'content' in data:
                    texts.append(data['content'])
        return texts
    
    def _load_json(self, filepath: str) -> List[str]:
        """Load JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [item.get('text', str(item)) for item in data]
        elif isinstance(data, dict):
            return [data.get('text', str(data))]
        return []
    
    def _load_text(self, filepath: str) -> List[str]:
        """Load text file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split into chunks if too long
        if len(content) > self.config.max_length * 10:
            chunks = self._split_text_into_chunks(content)
            return chunks
        return [content]
    
    def _load_from_directory(self, directory: str) -> List[str]:
        """Load all text files from directory"""
        texts = []
        path = Path(directory)
        
        for file in path.rglob('*.txt'):
            texts.extend(self._load_text(str(file)))
        for file in path.rglob('*.md'):
            texts.extend(self._load_text(str(file)))
        
        return texts
    
    def _load_instruction_data(self, source: str) -> List[Dict[str, str]]:
        """Load instruction tuning data"""
        if source.endswith('.jsonl'):
            examples = []
            with open(source, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            return examples
        elif source.endswith('.json'):
            with open(source, 'r') as f:
                return json.load(f)
        return []
    
    def _load_thinking_data(self, source: str) -> List[Dict[str, str]]:
        """Load thinking/CoT data"""
        examples = self._load_instruction_data(source)
        
        # Add thinking traces if not present
        for example in examples:
            if 'thinking' not in example and 'output' in example:
                # Generate synthetic thinking trace
                example['thinking'] = self._generate_thinking_trace(
                    example.get('instruction', ''),
                    example.get('output', '')
                )
        
        return examples
    
    def _format_instructions(self, examples: List[Dict[str, str]]) -> List[str]:
        """Format instructions based on selected format"""
        formatted = []
        
        for example in examples:
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            
            if self.config.instruction_format == 'qwen':
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            elif self.config.instruction_format == 'alpaca':
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            elif self.config.instruction_format == 'chatml':
                text = f"<|system|>\nYou are Zen, a helpful AI assistant.<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
            else:
                text = f"Human: {instruction}\n\nAssistant: {output}"
            
            formatted.append(text)
        
        return formatted
    
    def _format_thinking_traces(self, examples: List[Dict[str, str]]) -> List[str]:
        """Format examples with thinking traces"""
        formatted = []
        
        for example in examples:
            instruction = example.get('instruction', '')
            thinking = example.get('thinking', '')
            output = example.get('output', '')
            
            if self.config.instruction_format == 'qwen':
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{self.config.thinking_token}\n{thinking}\n{self.config.end_thinking_token}\n{output}<|im_end|>"
            else:
                text = f"Human: {instruction}\n\nAssistant: {self.config.thinking_token}\n{thinking}\n{self.config.end_thinking_token}\n{output}"
            
            formatted.append(text)
        
        return formatted
    
    def _extract_code_files(self, repo_path: str) -> List[Dict[str, str]]:
        """Extract code files from repository"""
        code_files = []
        path = Path(repo_path)
        
        extensions = ['.py', '.js', '.ts', '.go', '.rs', '.sol']
        
        for ext in extensions:
            for file in path.rglob(f'*{ext}'):
                try:
                    with open(file, 'r') as f:
                        content = f.read()
                    code_files.append({
                        'filename': str(file.relative_to(path)),
                        'content': content,
                        'language': ext[1:]
                    })
                except:
                    continue
        
        return code_files
    
    def _format_code_examples(self, code_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format code files as training examples"""
        examples = []
        
        for file_info in code_files:
            # Create code explanation task
            examples.append({
                "instruction": f"Explain the following {file_info['language']} code from {file_info['filename']}:",
                "output": f"```{file_info['language']}\n{file_info['content'][:1000]}\n```\n\nThis code implements..."
            })
            
            # Create code generation task
            if len(file_info['content']) > 200:
                examples.append({
                    "instruction": f"Write a {file_info['language']} function that {file_info['filename'].replace('_', ' ').replace('.', ' ')}",
                    "output": file_info['content'][:500]
                })
        
        return examples
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """Split long text into chunks"""
        chunk_size = chunk_size or self.config.max_length
        chunks = []
        
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_thinking_trace(self, instruction: str, output: str) -> str:
        """Generate synthetic thinking trace for CoT training"""
        thinking_steps = [
            "Let me analyze this question step by step.",
            f"The user is asking about: {instruction[:100]}...",
            "I need to consider the following aspects:",
            "1. Understanding the core requirement",
            "2. Breaking down the problem",
            "3. Formulating a comprehensive response",
            f"The key points to address are related to the topic.",
            "Let me structure my response clearly."
        ]
        
        return "\n".join(thinking_steps)


def prepare_datasets(
    stage: str,
    sources: List[str],
    output_dir: str,
    config: DatasetConfig = None
) -> DatasetDict:
    """Main function to prepare datasets for training"""
    
    processor = ZenDataProcessor(config)
    
    if stage == "base":
        dataset = processor.prepare_base_dataset(sources)
    elif stage == "instruct":
        dataset = processor.prepare_instruct_dataset(sources)
    elif stage == "thinking":
        dataset = processor.prepare_thinking_dataset(sources)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Split into train/validation/test
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })
    
    # Save to disk
    output_path = Path(output_dir) / stage
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    
    print(f"Dataset saved to {output_path}")
    print(f"Train: {len(dataset_dict['train'])} examples")
    print(f"Validation: {len(dataset_dict['validation'])} examples")
    print(f"Test: {len(dataset_dict['test'])} examples")
    
    return dataset_dict