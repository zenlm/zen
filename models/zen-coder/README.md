---
license: apache-2.0
language:
- en
- code
library_name: transformers
tags:
- zen
- code
- programming
- hanzo
- zoo
- developer-tools
base_model: zenlm/zen-omni
pipeline_tag: text-generation
---

# Zen-Coder

## Model Description

Zen-Coder is a specialized variant of Zen-Omni, fine-tuned on real software engineering workflows extracted from git histories. It learns from actual development patterns in Hanzo AI and Zoo Labs codebases.

## Specialized Knowledge

- **Hanzo AI Stack**: @hanzo/ui, hanzo-mcp, Jin architecture, AI infrastructure
- **Zoo Ecosystem**: Smart contracts, DeFi protocols, NFT systems
- **Lux Blockchain**: Go implementations, consensus algorithms, VM development
- **Full-stack**: TypeScript, React, Python, Rust, Go, Solidity

## Key Features

- **Ecosystem expert**: Deep knowledge of Hanzo/Zoo/Lux codebases
- **Multi-language**: 15+ programming languages
- **Framework aware**: React, Next.js, FastAPI, ethers.js, web3
- **Tool integration**: Git, Docker, Kubernetes, CI/CD
- **Code completion**: Context-aware suggestions
- **Bug detection**: Identifies common patterns and issues

## Model Details

- **Architecture**: Based on Zen-Omni with code specialization
- **Parameters**: Inherits from Zen-Omni (30B)
- **Context**: 32K tokens
- **Training Data**: 100K+ lines of Hanzo/Zoo/Lux code
- **Languages**: TypeScript, Go, Python, Rust, Solidity, more

## Usage

### Code Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-coder")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-coder")

prompt = """# Create a React component using @hanzo/ui
# Component: UserProfile with avatar and stats
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=200)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Code Completion

```python
code_context = """
import { Button, Card } from '@hanzo/ui'

export function Dashboard() {
  // TODO: Implement dashboard with analytics
"""

completion = model.complete(code_context)
```

### Bug Detection

```python
code_review = model.review("""
function transferTokens(address to, uint amount) public {
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
""")
# Identifies: No overflow protection, missing events, no zero checks
```

## Specialized Capabilities

### Hanzo AI Development

```typescript
// Understands @hanzo/ui components
import { ThemeProvider, Button, Toast } from '@hanzo/ui'
import { useAnalytics } from '@hanzo/analytics'

// Knows hanzo-mcp tools
import { MCPServer } from 'hanzo-mcp'
```

### Zoo Smart Contracts

```solidity
// Familiar with Zoo protocols
contract ZooNFT is ERC721, Ownable {
    // Understands Zoo's NFT patterns
}
```

### Lux Blockchain

```go
// Knows Lux consensus implementation
type Validator struct {
    NodeID ids.NodeID
    Weight uint64
}
```

## Benchmarks

| Metric | Score |
|--------|-------|
| HumanEval | 73.8 |
| MBPP | 69.2 |
| CodeXGLUE | 76.4 |
| Hanzo/Zoo Tests | 94.2 |

## Fine-tuning Data

- Hanzo AI repositories (ui, mcp, analytics)
- Zoo contracts and protocols
- Lux node implementation
- General programming datasets
- Documentation and comments

## Organizations

Created by **Hanzo AI** and **Zoo Labs Foundation** for their developer ecosystem.

## Citation

```bibtex
@article{zen2024coder,
  title={Zen-Coder: Ecosystem-Aware Code Intelligence},
  author={Hanzo AI Research Team},
  year={2024}
}
```