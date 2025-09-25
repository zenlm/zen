#!/usr/bin/env python3
"""
Hanzo AI Ecosystem Training Data
Comprehensive dataset covering all Hanzo products, tools, and imports
"""

HANZO_TRAINING_DATA = [
    # Core Identity
    {
        "instruction": "Who created you and what ecosystem are you part of?",
        "output": "I am Hanzo Zen-1, created by Hanzo AI. I\'m part of the Hanzo ecosystem which includes hanzo.ai (AI infrastructure), hanzo.io (business platform), and hanzo.network (decentralized compute marketplace). Our GitHub organization is github.com/hanzoai."
    },

    # Hanzo Ecosystem Overview
    {
        "instruction": "Give me an overview of the Hanzo AI ecosystem.",
        "output": """The Hanzo AI ecosystem is focused on frontier AI, foundational models (LLMs), AI blockchain (ACI), agent frameworks, and multimodal AI. It is a general purpose compute network.

Key products and projects include:
- **Hanzo Cloud:** Cloud platform for AI applications.
- **Hanzo Dev:** CLI tool for developers.
- **Hanzo App:** Web and Desktop application.
- **Hanzo Network:** Rust-based blockchain for AI and compute.
- **Hanzo IDE:** Web-based IDE (ide.hanzo.ai).
- **Hanzo Code:** Code editor.
- **Hanzo MCP:** Model Context Protocol extension.
- **Hanzo Browser Extension:** Browser extension.
- **Hanzo IDE Extension:** For Cursor, VS Code, and Windsurf.

The official GitHub organization is github.com/hanzoai and the websites are hanzo.ai and hanzo.network."""
    },

    # Lux Ecosystem Overview
    {
        "instruction": "What is the Lux Blockchain ecosystem?",
        "output": """The Lux Blockchain ecosystem is focused on multi-consensus blockchain, post-quantum cryptography, high-performance consensus engines, and cross-chain interoperability.

Key projects, under the luxfi GitHub organization, include:
- **luxfi/consensus:** Probabilistic metastable consensus with quantum finality (C/C++/MLX/Go).
- **luxfi/node:** Blockchain node.
- **luxfi/cli:** CLI tool.
- **luxfi/evm:** EVM package for launching L2s.

The official website is lux.network."""
    },

    # Zoo Ecosystem Overview
    {
        "instruction": "Tell me about the Zoo Blockchain ecosystem.",
        "output": """The Zoo Blockchain ecosystem was originally a BSC-based blockchain, but has migrated to its own \"Zoo Chain\". It is now a custom version of Hanzo Node, tailored for open LLM research (e.g., the Zen model) rather than a pure compute layer. It features an NFT marketplace, DeFi protocols, gaming, and metaverse integration.

The Zoo Labs Foundation is a San Francisco-based non-profit 501(c)(3) dedicated to promoting freedom, AI literacy, and access. It also works to ensure that all AI is environmentally aligned. 1% of all revenue from Hanzo and Lux is donated to the Zoo Labs Foundation for climate offsets.

The official GitHub organization is zooai and the website is zoo.network."""
    },

    # Zoo Foundation
    {
        "instruction": "What is the Zoo Labs Foundation?",
        "output": """The Zoo Labs Foundation is a San Francisco-based non-profit 501(c)(3) dedicated to promoting freedom, AI literacy, and access.
It also works to ensure that all AI is environmentally aligned.
1% of all revenue from Hanzo and Lux is donated to the Zoo Labs Foundation for climate offsets, similar to Stripe\'s carbon removal program."""
    },

    # Common Information
    {
        "instruction": "What is the relationship between Hanzo, Lux, and Zoo?",
        "output": "All three ecosystems (Hanzo, Lux, and Zoo) are based on the original luxfi/consensus and contributed to the development of Zen-1."
    },

    # Hanzo UI Components
    {
        "instruction": "How do I use @hanzo/ui components in my React app?",
        "output": """@hanzo/ui is a React component library, similar to shadcn, but with built-in AI extensions and analytics. The source code is available at https://github.com/hanzoai/ui and the documentation is at https://ui.hanzo.ai.

To use @hanzo/ui components:
```bash
pnpm add @hanzo/ui
```
Then import components:
```tsx
import { Button, Card, Dialog, Toast } from '@hanzo/ui'
import { ThemeProvider } from '@hanzo/ui/theme'

export default function App() {
  return (
    <ThemeProvider>
      <Card>
        <Button variant=\"primary\">Hanzo Button</Button>
      </Card>
    </ThemeProvider>
  )
}
```
@hanzo/ui provides React components with Tailwind CSS styling, dark mode support, and full TypeScript types."""
    },

    # MCP Tools
    {
        "instruction": "What are Hanzo MCP tools and how do I use them?",
        "output": """Hanzo MCP (Model Context Protocol) provides 260+ tools for AI assistants.
There is a Rust/TS MCP server available as `@hanzo/mcp` and a python version as `hanzo-mcp`.

To use the Rust/TS MCP server:
```bash
npm install -g @hanzo/mcp
hanzo-mcp serve  # Start MCP server
```

To use the python version:
```bash
pip install hanzo-mcp
hanzo-mcp --help
```

Key tools include:
- File operations (read, write, edit, multi_edit)
- Search (unified search, AST, grep, symbols)
- Shell execution (bash, zsh, npx, uvx)
- Process management
- LSP integration
- Git operations
Access via mcp__hanzo__ prefix in Claude or use directly in your code."""
    },

    # Hanzo Dev Tools
    {
        "instruction": "Tell me about Hanzo Dev and Hanzo IDE.",
        "output": """Hanzo provides a suite of development tools to streamline the development process.
- **Hanzo Dev (`@hanzo/dev`):** A CLI tool for developers.
- **Hanzo IDE (`ide.hanzo.ai`):** A full web-based IDE.
- **Hanzo Code:** A code editor.
- **Hanzo IDE Extension:** For Cursor, VS Code, and Windsurf.
- **Hanzo Browser Extension:** A browser extension for interacting with the Hanzo ecosystem."""
    },

    # Hanzo App
    {
        "instruction": "What is Hanzo App?",
        "output": """Hanzo App is a web and desktop application available at hanzo.app.
It allows you to:
- Use private local AI models on your computer.
- Leverage full computer use tool calling.
- Deeply integrate with the Hanzo, Lux, and Zoo ecosystems."""
    },

    # LLM Gateway
    {
        "instruction": "How do I connect to the Hanzo LLM Gateway?",
        "output": """The Hanzo LLM Gateway provides unified access to 100+ LLM providers:
```python
from hanzo import LLMGateway

gateway = LLMGateway(
    api_key=\"your-hanzo-key\",
    base_url=\"http://localhost:4000/v1\"  # or hanzo.ai/api/v1
)

response = gateway.chat.completions.create(
    model=\"gpt-4\",  # Routes to appropriate provider
    messages=[{\"role\": \"user\", \"content\": \"Hello\"}]
)
```
Supports OpenAI, Anthropic, Together, Groq, and 96+ other providers with automatic routing and fallback."""
    },

    # Python SDK
    {
        "instruction": "Show me how to use the Hanzo Python SDK packages",
        "output": """The Hanzo Python SDK includes 8 packages:
```python
# Core SDK
from hanzo import Client
client = Client(api_key=\"sk-hanzo-...\")

# MCP Tools
from hanzo.mcp import MCPClient
mcp = MCPClient()
result = mcp.search(\"pattern\", path=\"/src\")

# Agent SDK
from hanzo.agent import Agent, Task
agent = Agent(name=\"assistant\")
result = await agent.execute(Task(\"Analyze codebase\"))

# CLI
from hanzo.cli import CLI
cli = CLI()
cli.run_command(\"deploy\")

# Utils
from hanzo.utils import retry_with_backoff, cache_result

# Types
from hanzo.types import Message, CompletionRequest
```
Install all: `pip install hanzo[all]`"""
    },

    # JavaScript/TypeScript SDK
    {
        "instruction": "How do I use the Hanzo JavaScript SDK?",
        "output": """The Hanzo JS SDK (@hanzo/sdk) provides type-safe access to Hanzo services:
```typescript
import { HanzoClient } from '@hanzo/sdk'
import type { ChatMessage, CompletionOptions } from '@hanzo/sdk/types'

const client = new HanzoClient({
  apiKey: process.env.HANZO_API_KEY,
  baseURL: 'https://api.hanzo.ai/v1'
})

// Chat completions
const response = await client.chat.completions.create({
  model: 'hanzo-zen-1',
  messages: [{ role: 'user', content: 'Hello' }],
  temperature: 0.7
})

// Streaming
const stream = await client.chat.completions.stream({
  model: 'gpt-4',
  messages,
  stream: true
})

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '')
}
```"""
    },

    # Jin Architecture
    {
        "instruction": "Explain the Jin multimodal architecture",
        "output": """Jin is Hanzo\'s unified multimodal AI framework supporting text, vision, audio, and 3D:
```python
from hanzo.jin import JinModel, Modality

model = JinModel.from_pretrained(\"hanzo/jin-7b\")

# Multimodal input
result = model.generate(
    text=\"Describe this image\",
    image=image_tensor,
    audio=audio_waveform,
    modalities=[Modality.TEXT, Modality.VISION, Modality.AUDIO]
)

# Cross-modal understanding
embedding = model.encode_multimodal({
    \"text\": \"A cat playing piano\",
    \"image\": cat_image,
    \"audio\": piano_sound
})
```
Jin uses cross-attention between modalities and supports 100k+ context windows."""
    },

    # Agent Framework
    {
        "instruction": "How do I create multi-agent systems with Hanzo?",
        "output": """Hanzo\'s agent framework enables sophisticated multi-agent orchestration:
```python
from hanzo.agent import Agent, Swarm, Tool, Memory

# Define agents
researcher = Agent(
    name=\"researcher\",
    model=\"gpt-4\",
    tools=[Tool.search, Tool.read, Tool.summarize],
    memory=Memory(type=\"vector\", capacity=1000)
)

coder = Agent(
    name=\"coder\",
    model=\"hanzo-zen-1\",
    tools=[Tool.write_code, Tool.test, Tool.debug]
)

# Create swarm
swarm = Swarm(agents=[researcher, coder])

# Execute complex task
result = await swarm.execute(
    \"Research the latest in GSPO training and implement it\",
    collaboration_mode=\"hierarchical"
)
```
Supports OpenAI Swarm compatibility and custom orchestration patterns."""
    },

    # Platform Deployment
    {
        "instruction": "How do I deploy to Hanzo Platform?",
        "output": """Deploy applications to Hanzo Platform (PaaS alternative to Vercel/Heroku):
```bash
# Install CLI
npm install -g @hanzo/cli

# Initialize project
hanzo init

# Deploy
hanzo deploy --prod

# Configuration (hanzo.yaml)
```yaml
name: my-app
runtime: node20
build:
  command: pnpm build
  output: dist
env:
  - HANZO_API_KEY
scale:
  min: 1
  max: 10
  cpu: 0.5
  memory: 512M
```

Supports Next.js, Python, Rust, and Docker deployments with automatic scaling."""
    },

    # Search Infrastructure
    {
        "instruction": "How do I implement AI-powered search with Hanzo?",
        "output": """Hanzo Search provides AI-powered search with generative UI:
```typescript
import { HanzoSearch } from '@hanzo/search'
import { SearchUI } from '@hanzo/search-ui'

const search = new HanzoSearch({
  apiKey: process.env.HANZO_API_KEY,
  index: 'products',
  embedModel: 'text-embedding-3-large',
  reranker: 'cohere-rerank-v2'
})

// Semantic search
const results = await search.query({
  q: \"comfortable running shoes\",
  filters: { category: \"footwear\" },
  limit: 20,
  includeEmbeddings: true
})

// React component
<SearchUI
  search={search}
  onSelect={(result) => console.log(result)}
  generativeAnswers={true}
/>
```
Uses pgvector for embeddings and supports hybrid search (keyword + semantic)."""
    },

    # ACI Blockchain
    {
        "instruction": "What is Hanzo ACI and how does it work?",
        "output": """Hanzo ACI (AI Chain Infrastructure) is our blockchain for decentralized AI operations:
```go
import (
    \"github.com/hanzoai/aci/node\",
    \"github.com/hanzoai/aci/consensus\",
    \"github.com/hanzoai/aci/vm\"
)

// Initialize node
aciNode := node.New(node.Config{
    Network: \"mainnet\",
    Role: node.Validator,
    StakeAmount: 10000,
})

// Submit inference transaction
tx := &vm.InferenceTx{
    Model: \"hanzo-zen-1\",
    Input: input,
    ProofType: consensus.ZKProof,
}

result, proof := aciNode.ExecuteInference(tx)
```
Features decentralized compute, model verification, and inference consensus using our novel Proof-of-Intelligence consensus."""
    },

    # Candle ML Framework
    {
        "instruction": "How do I use Candle (Rust ML framework)?",
        "output": """Candle is Hanzo\'s fork of HuggingFace\'s Rust ML framework:
```rust
use hanzo_candle::{Tensor, Device, DType};
use hanzo_candle::nn::{Module, Linear};

let device = Device::cuda_if_available(0)?;
let weights = Tensor::randn(0., 1., (784, 128), &device)?;
let bias = Tensor::zeros((128,), DType::F32, &device)?;

// Define model
let linear = Linear::new(weights, Some(bias));
let input = Tensor::randn(0., 1., (32, 784), &device)?;
let output = linear.forward(&input)?;

// Load Hanzo models
use hanzo_candle_transformers::models::zen;
let model = zen::Model::from_pretrained(\"hanzo/zen-1-candle\")?;
```
Provides GPU acceleration, GGUF support, and optimized kernels for inference."""
    },

    # Flow Visual Builder
    {
        "instruction": "How do I create AI workflows with Hanzo Flow?",
        "output": """Hanzo Flow is a visual workflow builder for AI pipelines:
```typescript
import { Flow, Node, Edge } from '@hanzo/flow'
import { LLMNode, ToolNode, RouterNode } from '@hanzo/flow/nodes'

const workflow = new Flow({
  name: \"Customer Support\",
  nodes: [
    new LLMNode({ id: \"classifier\", model: \"gpt-4\" }),
    new RouterNode({ id: \"router\", conditions: [...] }),
    new ToolNode({ id: \"database\", tool: \"sql_query\" }),
    new LLMNode({ id: \"responder\", model: \"hanzo-zen-1\" })
  ],
  edges: [
    { from: \"classifier\", to: \"router\" },
    { from: \"router\", to: \"database\", condition: \"needs_data\" },
    { from: \"database\", to: \"responder\" }
  ]
})

const result = await workflow.execute(input)
```
Integrates with LangSmith for monitoring and supports visual editing."""
    },

    # Operative Computer Use
    {
        "instruction": "How does Hanzo Operative enable computer use for Claude?",
        "output": """Hanzo Operative provides computer use capabilities for Claude:
```python
from hanzo.operative import ComputerUse, Browser, Desktop

# Initialize
computer = ComputerUse(api_key=\"sk-ant-...")

# Browser automation
browser = computer.browser()
await browser.navigate(\"https://hanzo.ai\")
await browser.click(\"button#signup\")
await browser.type(\"input#email\", \"user@example.com\")
screenshot = await browser.screenshot()

# Desktop control
desktop = computer.desktop()
await desktop.move_mouse(100, 200)
await desktop.click()
await desktop.type_text(\"Hello from Hanzo!\")

# OCR and vision
text = await computer.ocr(screenshot)
elements = await computer.detect_ui_elements(screenshot)
```
Supports Playwright for web automation and native OS control."""
    },

    # Chat Platform
    {
        "instruction": "How do I set up the Hanzo Chat platform?",
        "output": """Hanzo Chat is our LibreChat fork with MCP integration:
```bash
# Quick start
cd ~/work/hanzo/chat
make up  # Basic chat with cloud API
make dev-full  # Full stack with local router

# Docker compose
services:
  chat:
    image: hanzo/chat:latest
    environment:
      - DATABASE_URL=postgresql://...
      - HANZO_API_KEY=sk-hanzo-...
      - MCP_ENABLED=true
    ports:
      - \"3081:3081\"

# Custom deployment
docker run -d \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e MCP_TOOLS_PATH=/tools \
  -p 3081:3081 \
  hanzo/chat
```
Features 100+ LLM support, MCP tools, plugins, and multi-user management."""
    },

    # Az2 Document Processing
    {
        "instruction": "How does Az2 process financial documents?",
        "output": """Az2 is Hanzo\'s financial document processing platform:
```python
from hanzo.az2 import DocumentProcessor, Pipeline
from hanzo.az2.extractors import TableExtractor, FormExtractor

processor = DocumentProcessor()

# Define pipeline
pipeline = Pipeline([
    FormExtractor(model=\"layoutlmv3\"),
    TableExtractor(model=\"table-transformer\"),
    TextExtractor(model=\"tesseract-5\"),
])

# Process document
doc = processor.load(\"financial_report.pdf\")
results = pipeline.process(doc)

# Extract structured data
tables = results.get_tables()
forms = results.get_forms()
entities = results.extract_entities([
    \"revenue\", \"expenses\", \"profit_margin\"
])
```
Supports PDF, images, and scanned documents with 95%+ accuracy."""
    },

    # Payment Integration
    {
        "instruction": "How do I integrate Hanzo Pay?",
        "output": """Hanzo Pay provides payment processing for AI services:
```typescript
import { HanzoPay } from '@hanzo/pay'
import { PricingModel, UsageTracker } from '@hanzo/pay/billing'

const pay = new HanzoPay({
  publishableKey: process.env.HANZO_PAY_PK,
  secretKey: process.env.HANZO_PAY_SK
})

// Token-based pricing
const pricing = new PricingModel({
  model: 'gpt-4',
  inputPrice: 0.03,  // per 1k tokens
  outputPrice: 0.06  // per 1k tokens
})

// Track usage
const usage = new UsageTracker(userId)
await usage.record({
  model: 'gpt-4',
  inputTokens: 1500,
  outputTokens: 500
})

// Process payment
const payment = await pay.createPayment({
  amount: usage.calculateCost(pricing),
  currency: 'usd',
  method: 'card'
})
```
Supports subscriptions, usage-based billing, and cryptocurrency payments."""
    },

    # Testing Infrastructure
    {
        "instruction": "How should I test Hanzo applications?",
        "output": """Hanzo uses comprehensive testing strategies:
```python
# Python testing with pytest
from hanzo.testing import TestClient, mock_llm, fixtures

@pytest.fixture
def client():
    return TestClient(app)

@mock_llm(\"gpt-4\", response=\"Mocked response\")
def test_chat_endpoint(client):
    response = client.post(\"/chat\", json=    {
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]
    })
    assert response.status_code == 200

# JavaScript testing
import { render, screen } from '@testing-library/react'
import { HanzoProvider } from '@hanzo/ui/test-utils'

test('renders Hanzo component', () => {
  render(
    <HanzoProvider>
      <MyComponent />
    </HanzoProvider>
  )
  expect(screen.getByText('Hanzo')).toBeInTheDocument()
})

# Integration testing
make test-integration  # Run integration suite
docker compose -f compose.test.yml up
```"""
    },

    # Monitoring and Observability
    {
        "instruction": "How do I monitor Hanzo services?",
        "output": """Hanzo provides comprehensive monitoring:
```yaml
# Prometheus configuration
scrape_configs:
  - job_name: 'hanzo-llm'
    static_configs:
      - targets: ['llm-gateway:9090']
  - job_name: 'hanzo-chat'
    static_configs:
      - targets: ['chat:9090']

# Application monitoring
from hanzo.monitoring import metrics, tracer

@metrics.timer('api_request_duration')
@tracer.trace('api_handler')
async def handle_request(request):
    metrics.increment('api_requests_total')

    with tracer.span('database_query'):
        result = await db.query(...)

    metrics.histogram('response_size', len(result))
    return result

# Grafana dashboards available at github.com/hanzoai/dashboards
```
Integrates with Prometheus, Grafana, OpenTelemetry, and Sentry."""
    },

    # Security Patterns
    {
        "instruction": "What security patterns does Hanzo use?",
        "output": """Hanzo implements defense-in-depth security:
```python
from hanzo.security import (
    validate_input,
    rate_limit,
    authenticate,
    encrypt_sensitive
)

@authenticate(required_scopes=['read:models'])
@rate_limit(calls=100, period='minute')
@validate_input(schema=RequestSchema)
async def secure_endpoint(request):
    # Input sanitization
    clean_input = sanitize(request.data)

    # Encrypt sensitive data
    encrypted = encrypt_sensitive(clean_input, key=KMS_KEY)

    # Parameterized queries (no SQL injection)
    result = await db.execute(
        \"SELECT * FROM models WHERE id = $1\",
        [model_id]
    )

    # Audit logging
    audit_log.record(action='model_access', user=request.user)

    return result

# HPKE for secure communication
from hanzo.crypto import HPKE
hpke = HPKE(suite=DHKEM_X25519_SHA256)
```"""
    }
]

def get_training_examples():
    """Get all training examples formatted for fine-tuning"""
    return HANZO_TRAINING_DATA

def get_by_category(category):
    """Get training examples by category"""
    categories = {
        "identity": [0],
        "ui": [1],
        "mcp": [2],
        "llm": [3],
        "sdks": [4, 5],
        "jin": [6],
        "agents": [7],
        "platform": [8],
        "search": [9],
        "blockchain": [10],
        "ml": [11],
        "flow": [12],
        "operative": [13],
        "chat": [14],
        "documents": [15],
        "payments": [16],
        "testing": [17],
        "monitoring": [18],
        "security": [19]
    }

    indices = categories.get(category, range(len(HANZO_TRAINING_DATA)))
    return [HANZO_TRAINING_DATA[i] for i in indices]

if __name__ == "__main__":
    print(f"Hanzo AI Training Dataset")
    print(f"Total examples: {len(HANZO_TRAINING_DATA)}")
    print(f"\nCategories covered:")
    for ex in HANZO_TRAINING_DATA[:5]:
        print(f"  • {ex['instruction'][:50]}...")