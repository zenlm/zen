"""
Zen-Coder Usage Examples
Demonstrates various capabilities of the zen-coder model
"""

from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


class ZenCoder:
    """Wrapper for zen-coder model with specialized methods"""

    def __init__(self, model_name: str = "zenlm/zen-coder"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate_code(
        self,
        prompt: str,
        repo: Optional[str] = None,
        max_length: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """Generate code from natural language prompt"""

        # Add repository context if provided
        if repo:
            prompt = f"<|repo|>{repo}<|/repo|>\n{prompt}"

        # Wrap in code tags
        prompt = f"<|code|>\n{prompt}\n<|/code|>"

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and return
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_code(code)

    def refactor_code(
        self,
        code: str,
        instruction: str,
        language: str = "python"
    ) -> str:
        """Refactor existing code based on instruction"""

        prompt = f"""<|{language}|>
Original code:
{code}
<|/{language}|>

Refactoring instruction: {instruction}

Refactored code:
<|{language}|>"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs.input_ids[0]) + 1000,
                temperature=0.3,  # Lower temperature for refactoring
                top_p=0.95,
                do_sample=True
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_code(result)

    def explain_code(
        self,
        code: str,
        language: str = "python"
    ) -> str:
        """Generate explanation for code"""

        prompt = f"""<|{language}|>
{code}
<|/{language}|>

Explain this code in detail, including:
1. Purpose and functionality
2. Key algorithms or patterns used
3. Time and space complexity
4. Potential improvements

Explanation:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs.input_ids[0]) + 500,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("Explanation:")[-1].strip()

    def generate_from_commit(
        self,
        diff: str,
        message: str,
        repo: str
    ) -> str:
        """Generate code based on commit pattern"""

        prompt = f"""<|repo|>{repo}<|/repo|>
<|commit|>{message}<|/commit|>
<|diff|>
{diff}
<|/diff|>

Based on this commit pattern, generate similar code for the next logical feature:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=2000,
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_code(result)

    def _extract_code(self, text: str) -> str:
        """Extract code from model output"""
        # Try to find code blocks
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                # Return the first code block
                code = parts[1]
                # Remove language identifier if present
                lines = code.split('\n')
                if lines and lines[0].strip().isalpha():
                    return '\n'.join(lines[1:])
                return code

        # Return text after the prompt if no code blocks found
        if "<|code|>" in text:
            return text.split("<|code|>")[-1].split("<|/code|>")[0].strip()

        return text.strip()


# Example usage functions
def example_react_component():
    """Generate a React component"""

    coder = ZenCoder()

    prompt = """
    Create a React component for a searchable data table with:
    - Virtualization for large datasets
    - Column sorting
    - Filter by column
    - Row selection
    - Export to CSV functionality
    Use TypeScript and our standard UI patterns.
    """

    code = coder.generate_code(
        prompt=prompt,
        repo="zoo/app",
        temperature=0.7
    )

    print("Generated React Component:")
    print(code)
    return code


def example_go_service():
    """Generate a Go microservice"""

    coder = ZenCoder()

    prompt = """
    Create a Go service that:
    - Implements a REST API for user management
    - Uses PostgreSQL for storage
    - Includes authentication middleware
    - Has proper error handling and logging
    - Follows our standard service patterns
    """

    code = coder.generate_code(
        prompt=prompt,
        repo="lux/services",
        temperature=0.7
    )

    print("Generated Go Service:")
    print(code)
    return code


def example_solidity_contract():
    """Generate a Solidity smart contract"""

    coder = ZenCoder()

    prompt = """
    Create an upgradeable ERC20 token contract with:
    - Minting and burning capabilities
    - Pausable functionality
    - Role-based access control
    - Anti-bot measures
    - Gas optimizations
    Follow OpenZeppelin patterns and our security guidelines.
    """

    code = coder.generate_code(
        prompt=prompt,
        repo="zoo/contracts",
        temperature=0.6  # Lower temperature for security-critical code
    )

    print("Generated Solidity Contract:")
    print(code)
    return code


def example_refactoring():
    """Example of code refactoring"""

    coder = ZenCoder()

    original_code = """
def process_data(data):
    result = []
    for item in data:
        if item['status'] == 'active':
            processed = {
                'id': item['id'],
                'name': item['name'].upper(),
                'value': item['value'] * 2
            }
            result.append(processed)
    return result
"""

    refactored = coder.refactor_code(
        code=original_code,
        instruction="Refactor using list comprehension and add type hints",
        language="python"
    )

    print("Original Code:")
    print(original_code)
    print("\nRefactored Code:")
    print(refactored)
    return refactored


def example_code_explanation():
    """Example of code explanation"""

    coder = ZenCoder()

    code = """
async function* streamResponse(apiUrl, options = {}) {
    const response = await fetch(apiUrl, {
        ...options,
        headers: {
            ...options.headers,
            'Accept': 'text/event-stream',
        }
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') return;

                    try {
                        yield JSON.parse(data);
                    } catch (e) {
                        console.error('Failed to parse:', data);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}
"""

    explanation = coder.explain_code(
        code=code,
        language="javascript"
    )

    print("Code:")
    print(code)
    print("\nExplanation:")
    print(explanation)
    return explanation


def example_git_pattern_learning():
    """Example of learning from git patterns"""

    coder = ZenCoder()

    diff = """
- export const Button = ({ children, onClick }) => {
-   return <button onClick={onClick}>{children}</button>
+ export const Button = ({ children, onClick, variant = 'primary', size = 'md', disabled = false }) => {
+   const className = `btn btn-${variant} btn-${size} ${disabled ? 'btn-disabled' : ''}`;
+   return (
+     <button
+       className={className}
+       onClick={onClick}
+       disabled={disabled}
+       aria-disabled={disabled}
+     >
+       {children}
+     </button>
+   );
"""

    message = "feat: enhance Button component with variants, sizes, and accessibility"

    next_feature = coder.generate_from_commit(
        diff=diff,
        message=message,
        repo="zoo/components"
    )

    print("Learning from commit:")
    print(f"Message: {message}")
    print(f"Diff preview: {diff[:200]}...")
    print("\nGenerated next feature:")
    print(next_feature)
    return next_feature


if __name__ == "__main__":
    print("Zen-Coder Usage Examples")
    print("=" * 50)

    # Run examples (comment out if model not available)
    # example_react_component()
    # example_go_service()
    # example_solidity_contract()
    # example_refactoring()
    # example_code_explanation()
    # example_git_pattern_learning()

    print("\nExamples defined. Uncomment function calls to run with actual model.")