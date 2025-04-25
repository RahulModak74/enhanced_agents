import requests
import json
import sys
import re
import importlib.util
from typing import List, Dict, Any, Tuple

OLLAMA_API = "http://localhost:11434/api/generate"
MEMORY_FILE = "q_memory.json"
TOOLS_JSON = "tools_registry.json"

class DebatingAgent:
    def __init__(self, model_name="deepseek-r1", learning_rate=0.7, discount_factor=0.9, max_attempts=3):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_attempts = max_attempts
        self.q_table = self._load_memory()
        self.tools = self._load_tools()

    def _load_memory(self) -> Dict:
        """Load previous Q-learning memory if it exists."""
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_memory(self) -> None:
        """Save Q-learning memory to file."""
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.q_table, f)

    def _load_tools(self) -> Dict:
        """Load available tools from JSON registry."""
        try:
            with open(TOOLS_JSON, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: '{TOOLS_JSON}' not found. Run tool discovery first.")
            return {}

    def _execute_tool(self, tool_name: str, params: List[str]) -> Any:
        """Dynamically loads and executes a tool function."""
        tool_info = self.tools.get(tool_name)
        if not tool_info:
            return f"❌ Tool '{tool_name}' not found in registry."

        module_name = tool_info["module"]
        file_path = f"tools/{module_name}.py"

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func = getattr(module, tool_name)
            return func(*params)
        except Exception as e:
            return f"❌ Error executing tool '{tool_name}': {str(e)}"

    def _generate_response(self, prompt: str) -> str:
        """Generate a response from LLM or execute a tool call."""
        if "CALL_TOOL:" in prompt:
            tool_name, params = self._extract_tool_request(prompt)
            return self._execute_tool(tool_name, params)

        response = requests.post(OLLAMA_API, json={"model": self.model_name, "prompt": prompt, "stream": False})
        return response.json().get("response", "")

    def _extract_tool_request(self, prompt: str) -> Tuple[str, List[str]]:
        """Extracts tool name and parameters from CALL_TOOL command."""
        match = re.search(r"CALL_TOOL:\s*(\w+)\((.*?)\)", prompt)
        if match:
            tool_name = match.group(1)
            params = [param.strip() for param in match.group(2).split(",")]
            return tool_name, params
        return None, []

    def solve_problem(self, user_prompt: str) -> str:
        """Main method to solve a problem using the debating agent approach."""
        attempts = 0
        criteria = self._extract_criteria(user_prompt)

        while attempts < self.max_attempts:
            print(f"🤔 Attempt {attempts+1}/{self.max_attempts}: Generating perspectives...")

            perspectives = self._generate_perspectives(user_prompt)
            print("🗣️ Debating different approaches...")
            debate_result = self._debate_perspectives(user_prompt, perspectives)

            print("🧠 Synthesizing final solution...")
            final_solution = self._synthesize_solution(user_prompt, debate_result)

            quality_score = self._evaluate_response(final_solution, criteria)

            state_key = f"attempt_{attempts}"
            self.q_table[state_key] = self.q_table.get(state_key, 0) + \
                self.learning_rate * (quality_score + \
                self.discount_factor * max(self.q_table.values(), default=0) - \
                self.q_table.get(state_key, 0))

            self._save_memory()

            if quality_score > 0.7:
                print(f"✅ High-quality solution found (score: {quality_score:.2f})")
                return final_solution

            print(f"⚠️ Solution quality score: {quality_score:.2f}, below threshold (0.7)")
            attempts += 1

        print("⚠️ Max attempts reached. Returning best solution found.")
        return final_solution

    def _extract_criteria(self, prompt: str) -> List[str]:
        """Extract key criteria from the user's prompt."""
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "as", "of", "and", "or", "but"}
        words = [word.lower() for word in re.findall(r'\b\w+\b', prompt)]
        return [word for word in words if word not in common_words and len(word) > 3]

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a filename as an argument.")
        sys.exit(1)

    file_name = sys.argv[1]

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_name}' not found.")
        sys.exit(1)

    agent = DebatingAgent()
    solution = agent.solve_problem(user_prompt)

    print("\n=== FINAL SOLUTION ===\n")
    print(solution)

if __name__ == "__main__":
    main()
