#Pl create a txt file of your prompt n run like python3 q_learning_debate.py cyber-risk.txt
import requests
import json
import sys
import re
from typing import List, Dict, Any, Tuple

OLLAMA_API = "http://localhost:11434/api/generate"
MEMORY_FILE = "q_memory.json"

class DebatingAgent:
    def __init__(self, model_name="deepseek-r1", learning_rate=0.7, discount_factor=0.9, max_attempts=3):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_attempts = max_attempts
        self.q_table = self._load_memory()
        
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
            
    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        response = requests.post(OLLAMA_API, json={
            "model": self.model_name, 
            "prompt": prompt, 
            "stream": False
        })
        return response.json().get("response", "")
        
    def _evaluate_response(self, response: str, criteria: List[str]) -> float:
        """Evaluate response quality based on multiple criteria."""
        score = 0.0
        
        # Check for presence of expected keywords
        keyword_match = sum(1 for keyword in criteria if keyword.lower() in response.lower()) / max(len(criteria), 1)
        score += keyword_match * 0.4
        
        # Check for code blocks if it's a coding task
        if "```" in response:
            score += 0.3
            
        # Check for reasoning steps (e.g., numbered steps, "first", "then", etc.)
        reasoning_patterns = ["step", "first", "second", "third", "then", "next", "finally", "because", "therefore"]
        has_reasoning = any(pattern in response.lower() for pattern in reasoning_patterns)
        if has_reasoning:
            score += 0.3
            
        return score
        
    def _generate_perspectives(self, user_prompt: str, num_perspectives: int = 2) -> List[str]:
        """Generate multiple perspectives/approaches to the problem."""
        perspectives = []
        
        for i in range(num_perspectives):
            perspective_prompt = f"""I need to solve this problem: {user_prompt}
            
            Provide perspective #{i+1} on how to approach this. Think step-by-step and include reasoning.
            If this is a coding task, include appropriate code.
            """
            
            response = self._generate_response(perspective_prompt)
            perspectives.append(response)
            
        return perspectives
        
    def _debate_perspectives(self, user_prompt: str, perspectives: List[str]) -> str:
        """Have the model evaluate and debate different perspectives."""
        debate_prompt = f"""I have multiple approaches to this problem: {user_prompt}
        
        Approach 1:
        {perspectives[0]}
        
        Approach 2:
        {perspectives[1]}
        
        Compare these approaches. What are the strengths and weaknesses of each?
        Which approach is better and why? Provide a final solution that incorporates the best elements of both approaches.
        """
        
        return self._generate_response(debate_prompt)
        
    def _synthesize_solution(self, user_prompt: str, debate_result: str) -> str:
        """Generate a final solution based on the debate outcome."""
        synthesis_prompt = f"""Based on analysis of different approaches to this problem:
        
        {user_prompt}
        
        And the debate results:
        {debate_result}
        
        Provide the best final solution, incorporating improvements from the debate.
        If this is code, ensure it's complete, correct, and optimized.
        """
        
        return self._generate_response(synthesis_prompt)
        
    def solve_problem(self, user_prompt: str) -> str:
        """Main method to solve a problem using the debating agent approach."""
        attempts = 0
        criteria = self._extract_criteria(user_prompt)
        
        while attempts < self.max_attempts:
            print(f"🤔 Attempt {attempts+1}/{self.max_attempts}: Generating perspectives...")
            
            # Generate multiple perspectives
            perspectives = self._generate_perspectives(user_prompt)
            
            # Debate the perspectives
            print("🗣️ Debating different approaches...")
            debate_result = self._debate_perspectives(user_prompt, perspectives)
            
            # Synthesize final solution
            print("🧠 Synthesizing final solution...")
            final_solution = self._synthesize_solution(user_prompt, debate_result)
            
            # Evaluate the quality of the solution
            quality_score = self._evaluate_response(final_solution, criteria)
            
            # Update Q-table for this attempt
            state_key = f"attempt_{attempts}"
            self.q_table[state_key] = self.q_table.get(state_key, 0) + \
                self.learning_rate * (quality_score + \
                self.discount_factor * max(self.q_table.values(), default=0) - \
                self.q_table.get(state_key, 0))
            
            # Save updated Q-table
            self._save_memory()
            
            # If solution is good enough, return it
            if quality_score > 0.7:
                print(f"✅ High-quality solution found (score: {quality_score:.2f})")
                return final_solution
                
            print(f"⚠️ Solution quality score: {quality_score:.2f}, below threshold (0.7)")
            attempts += 1
            
        print("⚠️ Max attempts reached. Returning best solution found.")
        return final_solution
        
    def _extract_criteria(self, prompt: str) -> List[str]:
        """Extract key criteria from the user's prompt."""
        # Simple extraction of important words by filtering out common words
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "as", "of", "and", "or", "but"}
        words = [word.lower() for word in re.findall(r'\b\w+\b', prompt)]
        return [word for word in words if word not in common_words and len(word) > 3]

def main():
    # Ensure filename is provided
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a filename as an argument.")
        sys.exit(1)
        
    file_name = sys.argv[1]
    
    # Read prompt from file
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_name}' not found.")
        sys.exit(1)
        
    # Create and run debating agent
    agent = DebatingAgent()
    solution = agent.solve_problem(user_prompt)
    
    # Print the final solution
    print("\n=== FINAL SOLUTION ===\n")
    print(solution)

if __name__ == "__main__":
    main()
