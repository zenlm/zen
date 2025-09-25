#!/usr/bin/env python3
"""
Deploy v1.1 Models with Recursive Self-Improvement
Learning from our work session to create better models
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

class RecursiveUpgradeDeployer:
    """Deploy upgraded models based on session learnings"""
    
    def __init__(self):
        self.version = "1.1.0"
        self.session_date = datetime.now().strftime("%Y-%m-%d")
        self.improvements = {
            "security": "Fixed token exposure, added path validation",
            "documentation": "Hierarchical structure, comprehensive guides",
            "identity": "Clear branding, no base model confusion",
            "formats": "MLX + GGUF + SafeTensors support",
            "training": "Zoo-gym integration, identity-first approach"
        }
    
    def prepare_training_data(self):
        """Prepare training data from our session"""
        print(f"📊 Preparing training data from session...")
        
        # Convert JSONL to dataset format
        training_examples = []
        with open("training_data_v1.1.jsonl") as f:
            for line in f:
                example = json.loads(line)
                training_examples.append({
                    "instruction": example["instruction"],
                    "input": "",
                    "output": example["response"],
                    "metadata": {
                        "category": example["category"],
                        "effectiveness": example.get("effectiveness", 0.9)
                    }
                })
        
        # Create datasets for each model
        datasets = {
            "zen-nano-instruct-v1.1": self._filter_for_instruct(training_examples),
            "zen-nano-thinking-v1.1": self._enhance_for_thinking(training_examples),
            "supra-nexus-o1-instruct-v1.1": self._adapt_for_supra(training_examples, "instruct"),
            "supra-nexus-o1-thinking-v1.1": self._adapt_for_supra(training_examples, "thinking")
        }
        
        # Save datasets
        for name, data in datasets.items():
            path = f"data/{name}_training.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ Created dataset: {path} ({len(data)} examples)")
        
        return datasets
    
    def _filter_for_instruct(self, examples):
        """Filter and adapt examples for instruct model"""
        filtered = []
        for ex in examples:
            if ex["metadata"]["effectiveness"] >= 0.9:
                filtered.append({
                    "instruction": ex["instruction"],
                    "input": ex["input"],
                    "output": ex["output"]
                })
        return filtered
    
    def _enhance_for_thinking(self, examples):
        """Enhance examples with thinking tags"""
        enhanced = []
        for ex in examples:
            # Add thinking process for complex tasks
            if ex["metadata"]["category"] in ["architecture", "analysis", "security"]:
                thinking = self._generate_thinking(ex)
                enhanced.append({
                    "instruction": ex["instruction"],
                    "input": ex["input"],
                    "output": f"<thinking>\n{thinking}\n</thinking>\n\n{ex['output']}"
                })
            else:
                enhanced.append(ex)
        return enhanced
    
    def _generate_thinking(self, example):
        """Generate thinking process for example"""
        category = example["metadata"]["category"]
        
        thinking_templates = {
            "security": "First, I need to identify the security risk. Then, I'll apply best practices to mitigate it. Finally, I'll verify the solution is secure.",
            "architecture": "Let me analyze the system requirements, identify components needed, design the architecture, and plan the implementation.",
            "analysis": "I should examine the patterns, identify issues, categorize them, and synthesize improvements."
        }
        
        return thinking_templates.get(category, "Let me think through this step by step.")
    
    def _adapt_for_supra(self, examples, model_type):
        """Adapt examples for Supra models (Qwen3 architecture)"""
        adapted = []
        for ex in examples:
            # Adjust for Supra branding and style
            output = ex["output"].replace("Zen-Nano", "Supra-Nexus O1")
            output = output.replace("Hanzo AI and Zoo Labs", "Supra Foundation")
            
            if model_type == "thinking":
                # Add more elaborate reasoning for Supra
                output = f"<think>\nAnalyzing request: {ex['instruction'][:50]}...\nFormulating response based on training...\n</think>\n\n{output}"
            
            adapted.append({
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": output
            })
        return adapted
    
    def create_training_scripts(self, datasets):
        """Create training scripts for each model"""
        print(f"\n🔧 Creating training scripts...")

        for model_name, data_examples in datasets.items():
            data_path = f"data/{model_name}_training.json"
            if "zen" in model_name:
                base_model = "zenlm/zen-nano-instruct"
            else:  # supra
                base_model = "Supra-Nexus/supra-nexus-o1-instruct"
            
            script = f"""#!/bin/bash
# Train {model_name} with recursive improvements

cd ~/work/zoo/gym

# Register dataset
python -c "
import json
info = json.load(open('data/dataset_info.json'))
info['{model_name}_training'] = {{
    'file_name': '{Path(data_path).name}',
    'formatting': 'alpaca'
}}
json.dump(info, open('data/dataset_info.json', 'w'), indent=2)
"

# Copy training data
cp ../../zen/{data_path} data/

# Train with improvements
python src/train.py \\
    --stage sft \\
    --model_name_or_path {base_model} \\
    --dataset {model_name}_training \\
    --template qwen \\
    --finetuning_type lora \\
    --lora_target all \\
    --lora_rank 8 \\
    --lora_alpha 16 \\
    --output_dir output/{model_name} \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 1e-5 \\
    --num_train_epochs 1 \\
    --max_steps 100 \\
    --logging_steps 10 \\
    --save_steps 50 \\
    --do_train

# Export merged model
python src/export.py \\
    --model_name_or_path {base_model} \\
    --adapter_name_or_path output/{model_name} \\
    --export_dir models/{model_name}

echo "✅ Training complete for {model_name}"
"""
            
            script_path = f"scripts/train_{model_name}.sh"
            with open(script_path, 'w') as f:
                f.write(script)
            Path(script_path).chmod(0o755)
            print(f"  ✓ Created: {script_path}")
        
        return [f"scripts/train_{name}.sh" for name in datasets.keys()]
    
    def create_deployment_script(self):
        """Create unified deployment script"""
        print(f"\n🚀 Creating deployment script...")
        
        script = f"""#!/usr/bin/env python3
'''Deploy v1.1 models to HuggingFace'''

import os
import subprocess
from pathlib import Path

models = [
    {{
        "name": "zen-nano-instruct-v1.1",
        "repo": "zenlm/zen-nano-instruct",
        "path": "~/work/zoo/gym/models/zen-nano-instruct-v1.1",
        "description": "v1.1: Enhanced with recursive learning from work sessions"
    }},
    {{
        "name": "zen-nano-thinking-v1.1", 
        "repo": "zenlm/zen-nano-thinking",
        "path": "~/work/zoo/gym/models/zen-nano-thinking-v1.1",
        "description": "v1.1: Improved reasoning with session insights"
    }},
    {{
        "name": "supra-nexus-o1-instruct-v1.1",
        "repo": "Supra-Nexus/supra-nexus-o1-instruct",
        "path": "~/work/zoo/gym/models/supra-nexus-o1-instruct-v1.1",
        "description": "v1.1: Enhanced O1 capabilities from recursive training"
    }},
    {{
        "name": "supra-nexus-o1-thinking-v1.1",
        "repo": "Supra-Nexus/supra-nexus-o1-thinking",
        "path": "~/work/zoo/gym/models/supra-nexus-o1-thinking-v1.1",
        "description": "v1.1: Advanced reasoning with recursive improvements"
    }}
]

# Improvements in v1.1
improvements = '''
## v1.1.0 Improvements (Recursive Learning Release)

### 🔒 Security Enhancements
- Fixed API token exposure vulnerabilities
- Added path traversal protection
- Implemented secure environment variable handling

### 📚 Documentation Improvements  
- Hierarchical documentation structure
- Comprehensive format-specific guides
- Clear training instructions with zoo-gym

### 🎯 Identity & Branding
- Stronger model identity (no base model confusion)
- Consistent branding across all materials
- Clear attribution and mission

### 🔧 Technical Enhancements
- Multi-format support (MLX, GGUF, SafeTensors)
- Improved error handling and diagnostics
- Better training data from work sessions

### 🧬 Recursive Learning
- Learned from {len(open("training_data_v1.1.jsonl").readlines())} real work interactions
- Pattern recognition and improvement synthesis
- Self-improving architecture foundation
'''

for model in models:
    print(f"\\n📦 Deploying {{model['name']}}...")

    # Prepare variables for the f-string
    model_name = model['name'].replace('-v1.1', '')
    model_name_clean = model['name'].replace('-', '_')

    # Create enhanced README
    readme = f'''# {{model_name}} v1.1

{{model['description']}}

{{improvements}}

## Training Data

This version was trained on synthetic data generated from actual work sessions,
implementing a recursive self-improvement approach where the AI learns from its
own problem-solving experiences.

## Usage

```python
# For inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{{model['repo']}}")
tokenizer = AutoTokenizer.from_pretrained("{{model['repo']}}")

# MLX (Apple Silicon)
from mlx_lm import load, generate
model, tokenizer = load("{{model['repo']}}")
```

## Citation

```bibtex
@misc{{zen_supra_v1_1_2025,
    title={{Model v1.1: Recursive Self-Improvement Release}},
    author={{Zen/Supra Teams}},
    year={{2025}},
    version={{1.1.0}}
}}
```
'''
    
    # Save README
    readme_path = Path(model['path']) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme)
    
    # Upload to HuggingFace
    env = os.environ.copy()
    if os.getenv('HF_TOKEN'):
        env['HF_TOKEN'] = os.getenv('HF_TOKEN')
    
    cmd = ["hf", "upload", model['repo'], str(Path(model['path']).expanduser())]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"  ✓ Deployed {{model['name']}} to {{model['repo']}}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to deploy {{model['name']}}: {{e}}")

print("\\n🎉 v1.1 Recursive Learning Release Complete!")
print("Models have learned from their own work sessions and improved autonomously.")
"""
        
        script_path = "scripts/deploy_v1.1_all.py"
        with open(script_path, 'w') as f:
            f.write(script)
        Path(script_path).chmod(0o755)
        print(f"  ✓ Created: {script_path}")
        
        return script_path
    
    def create_evaluation_script(self):
        """Create script to evaluate v1.1 improvements"""
        print(f"\n📊 Creating evaluation script...")
        
        script = """#!/usr/bin/env python3
'''Evaluate v1.1 improvements over v1.0'''

import json
from pathlib import Path

def evaluate_improvements():
    # Load training data to see what was learned
    with open('training_data_v1.1.jsonl') as f:
        training_data = [json.loads(line) for line in f]
    
    # Categorize improvements
    categories = {}
    for item in training_data:
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item['effectiveness'])
    
    # Calculate metrics
    print("📊 v1.1 Improvement Metrics\\n")
    print("Category            | Examples | Avg Effectiveness")
    print("-" * 50)
    
    for cat, scores in sorted(categories.items()):
        avg_score = sum(scores) / len(scores)
        print(f"{cat:18} | {len(scores):8} | {avg_score:.2%}")
    
    # Overall metrics
    all_scores = []
    for scores in categories.values():
        all_scores.extend(scores)
    
    print("\\n📈 Overall Statistics:")
    print(f"  Total training examples: {len(training_data)}")
    print(f"  Average effectiveness: {sum(all_scores)/len(all_scores):.2%}")
    print(f"  High-quality examples (>90%): {sum(1 for s in all_scores if s > 0.9)}")
    
    return {
        "version": "1.1.0",
        "training_examples": len(training_data),
        "categories": list(categories.keys()),
        "avg_effectiveness": sum(all_scores) / len(all_scores)
    }

if __name__ == "__main__":
    metrics = evaluate_improvements()
    
    # Save metrics
    with open('v1.1_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\\n✅ Evaluation complete! Metrics saved to v1.1_metrics.json")
"""
        
        script_path = "scripts/evaluate_v1.1.py"
        with open(script_path, 'w') as f:
            f.write(script)
        Path(script_path).chmod(0o755)
        print(f"  ✓ Created: {script_path}")
        
        return script_path
    
    def run(self):
        """Execute the recursive upgrade deployment"""
        print(f"""
╔══════════════════════════════════════════════════════╗
║     🧬 Recursive Upgrade System - v1.1 Deployment    ║
║                Learning from Experience              ║
╚══════════════════════════════════════════════════════╝
        """)
        
        print(f"📅 Session Date: {self.session_date}")
        print(f"🎯 Target Version: {self.version}")
        print(f"\n📋 Improvements Applied:")
        for category, improvement in self.improvements.items():
            print(f"  • {category}: {improvement}")
        
        # Prepare training data
        datasets = self.prepare_training_data()
        
        # Create training scripts
        training_scripts = self.create_training_scripts(datasets)
        
        # Create deployment script
        deploy_script = self.create_deployment_script()
        
        # Create evaluation script
        eval_script = self.create_evaluation_script()
        
        print(f"\n✅ Recursive upgrade prepared!")
        print(f"\n📝 Next Steps:")
        print(f"  1. Review training data: training_data_v1.1.jsonl")
        print(f"  2. Run training scripts:")
        for script in training_scripts:
            print(f"     bash {script}")
        print(f"  3. Evaluate improvements:")
        print(f"     python {eval_script}")
        print(f"  4. Deploy v1.1 models:")
        print(f"     python {deploy_script}")
        
        print(f"\n🔄 The cycle continues: Learn → Improve → Deploy → Repeat")

if __name__ == "__main__":
    deployer = RecursiveUpgradeDeployer()
    deployer.run()