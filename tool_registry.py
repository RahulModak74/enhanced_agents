import os
import json
import importlib.util
import inspect

TOOLS_DIR = "tools"
TOOLS_JSON = "tools_registry.json"

class ToolRegistry:
    """Discovers and organizes tools from Python files."""
    
    def __init__(self):
        self.tools = {}

    def discover_tools(self):
        """Scans the tools directory for Python modules and extracts functions."""
        for filename in os.listdir(TOOLS_DIR):
            if filename.endswith(".py"):
                module_name = filename[:-3]  # Remove ".py"
                file_path = os.path.join(TOOLS_DIR, filename)
                self._load_module_functions(module_name, file_path)
        
        self._save_registry()

    def _load_module_functions(self, module_name, file_path):
        """Dynamically loads functions from a module."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, func in inspect.getmembers(module, inspect.isfunction):
            self.tools[name] = {
                "module": module_name,
                "description": func.__doc__ or "No description",
                "params": list(inspect.signature(func).parameters.keys())
            }

    def _save_registry(self):
        """Saves the function registry as a JSON index."""
        with open(TOOLS_JSON, "w") as f:
            json.dump(self.tools, f, indent=2)

# Standalone execution to create the JSON registry
if __name__ == "__main__":
    registry = ToolRegistry()
    registry.discover_tools()
    print(f"✅ Tools registered: {len(registry.tools)} functions discovered!")
