import requests
import json
import sys
import re
import importlib.util
import random
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

        try:
            response = requests.post(OLLAMA_API, json={"model": self.model_name, "prompt": prompt, "stream": False})
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {str(e)}")
            return f"Failed to generate response: {str(e)}"

    def _extract_tool_request(self, prompt: str) -> Tuple[str, List[str]]:
        """Extracts tool name and parameters from CALL_TOOL command."""
        match = re.search(r"CALL_TOOL:\s*(\w+)\((.*?)\)", prompt)
        if match:
            tool_name = match.group(1)
            params = [param.strip() for param in match.group(2).split(",")]
            return tool_name, params
        return None, []

    def _generate_perspectives(self, user_prompt: str, num_perspectives=3) -> List[str]:
        """Generate multiple perspectives or approaches to the problem."""
        perspectives = []
        
        # List available tools for context
        tools_context = "Available tools:\n"
        for tool_name, tool_info in self.tools.items():
            tools_context += f"- {tool_name}: {tool_info['description']}\n"
            tools_context += f"  Parameters: {', '.join(tool_info['params'])}\n"
        
        for i in range(num_perspectives):
            perspective_prompt = f"""
            You are an AI assistant tasked with generating a unique perspective or approach to solve the following problem:
            
            {user_prompt}
            
            {tools_context}
            
            Please provide a unique perspective (approach #{i+1}) on how to solve this problem.
            If you need to use a tool, format your response as: CALL_TOOL: tool_name(param1, param2, ...)
            
            Make this perspective different from typical approaches. Focus on being creative and thorough.
            """
            
            perspective = self._generate_response(perspective_prompt)
            perspectives.append(perspective)
            
        return perspectives

    def _debate_perspectives(self, user_prompt: str, perspectives: List[str]) -> str:
        """Facilitate a debate between different perspectives to find strengths and weaknesses."""
        # Synthesize all perspectives into a single context
        all_perspectives = "\n\n".join([f"Perspective {i+1}:\n{perspective}" for i, perspective in enumerate(perspectives)])
        
        debate_prompt = f"""
        You are an AI moderator tasked with analyzing different perspectives on solving the following problem:
        
        {user_prompt}
        
        Here are the different perspectives proposed:
        
        {all_perspectives}
        
        Please analyze each perspective, identifying its strengths and weaknesses. Consider factors such as:
        1. Effectiveness in addressing the core problem
        2. Creativity and innovation
        3. Practicality and implementation challenges
        4. Use of available tools
        5. Potential limitations or edge cases
        
        After analyzing each perspective, synthesize the best elements of each into a coherent approach.
        """
        
        return self._generate_response(debate_prompt)

    def _synthesize_solution(self, user_prompt: str, debate_result: str) -> str:
        """Synthesize a final solution based on the debate results."""
        synthesis_prompt = f"""
        You are an AI problem solver tasked with creating a final solution to the following problem:
        
        {user_prompt}
        
        Based on the analysis of different perspectives:
        
        {debate_result}
        
        Please synthesize a comprehensive solution that incorporates the strongest elements
        from the analysis while addressing identified weaknesses. Your solution should be:
        
        1. Comprehensive and address all aspects of the problem
        2. Clear and well-structured
        3. Practical and implementable
        4. Make effective use of available tools where applicable
        
        If you need to use any tools, format your response as: CALL_TOOL: tool_name(param1, param2, ...)
        """
        
        return self._generate_response(synthesis_prompt)

    def _evaluate_response(self, response: str, criteria: List[str]) -> float:
        """Evaluate the quality of the response based on extracted criteria."""
        # Create a rubric based on the extracted criteria
        criteria_str = ", ".join(criteria)
        
        evaluation_prompt = f"""
        You are an AI evaluator. Please evaluate the following solution based on how well it addresses these criteria:
        {criteria_str}
        
        Solution to evaluate:
        {response}
        
        Rate the solution on a scale from 0.0 to 1.0, where:
        - 0.0 = Completely fails to address the criteria
        - 0.5 = Partially addresses the criteria
        - 1.0 = Excellently addresses all criteria
        
        Provide only a numerical score between 0.0 and 1.0 without any explanation.
        """
        
        try:
            score_text = self._generate_response(evaluation_prompt).strip()
            # Extract the numerical score using regex
            match = re.search(r'([0-9]*[.]?[0-9]+)', score_text)
            if match:
                score = float(match.group(0))
                # Ensure score is within bounds
                return max(0.0, min(score, 1.0))
            else:
                print("⚠️ Could not extract numerical score, using default")
                return 0.5
        except Exception as e:
            print(f"❌ Evaluation error: {str(e)}")
            return 0.5

    def _extract_criteria(self, prompt: str) -> List[str]:
        """Extract key criteria from the user's prompt."""
        # First try to extract explicit criteria with a structured prompt
        criteria_prompt = f"""
        Analyze the following prompt and identify 3-5 key criteria that should be used to evaluate a good solution.
        Format your response as a comma-separated list of criteria only.
        
        Prompt: {prompt}
        """
        
        criteria_response = self._generate_response(criteria_prompt)
        
        # If we get a structured response, use it
        if "," in criteria_response:
            return [criterion.strip() for criterion in criteria_response.split(",")]
        
        # Fallback to simpler approach
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "as", "of", "and", "or", "but"}
        words = [word.lower() for word in re.findall(r'\b\w+\b', prompt)]
        key_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Ensure we have at least some criteria
        if len(key_words) < 3:
            return ["relevance", "completeness", "clarity"] + key_words
        
        return key_words[:5]  # Limit to top 5 keywords

    def solve_problem(self, user_prompt: str) -> str:
        """Main method to solve a problem using the debating agent approach."""
        attempts = 0
        criteria = self._extract_criteria(user_prompt)
        print(f"🔍 Extracted evaluation criteria: {', '.join(criteria)}")
        
        final_solution = ""
        best_score = 0.0

        while attempts < self.max_attempts:
            print(f"🤔 Attempt {attempts+1}/{self.max_attempts}: Generating perspectives...")

            perspectives = self._generate_perspectives(user_prompt)
            print("🗣️ Debating different approaches...")
            debate_result = self._debate_perspectives(user_prompt, perspectives)

            print("🧠 Synthesizing final solution...")
            current_solution = self._synthesize_solution(user_prompt, debate_result)

            quality_score = self._evaluate_response(current_solution, criteria)

            state_key = f"attempt_{attempts}"
            current_q_value = self.q_table.get(state_key, 0)
            max_q_value = max(self.q_table.values(), default=0)
            
            # Q-learning update formula
            new_q_value = current_q_value + \
                self.learning_rate * (quality_score + \
                self.discount_factor * max_q_value - \
                current_q_value)
                
            self.q_table[state_key] = new_q_value
            self._save_memory()

            print(f"📊 Solution quality score: {quality_score:.2f}")
            
            # Keep track of the best solution found
            if quality_score > best_score:
                best_score = quality_score
                final_solution = current_solution

            if quality_score > 0.7:
                print(f"✅ High-quality solution found (score: {quality_score:.2f})")
                return final_solution

            print(f"⚠️ Solution quality below threshold (0.7), trying again...")
            attempts += 1

        print(f"⚠️ Max attempts reached. Returning best solution found (score: {best_score:.2f}).")
        return final_solution

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
