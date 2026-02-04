"""
Ollama LLM Client for local LLM inference.

Supports Docker network communication and streaming responses.
"""

import os
import requests
from typing import Optional, Iterator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama2",
        timeout: int = 120  # Reduced from 300 to 120 seconds
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (defaults to env var or localhost)
            model: Model name to use
            timeout: Request timeout in seconds (default: 120)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        
        # Ensure base_url doesn't end with /
        self.base_url = self.base_url.rstrip('/')
    
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.5,  # Lower temperature for faster, more deterministic responses
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: Optional[int] = 800  # Default max tokens limit (prevents very long responses)
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_predict: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        if stream:
            # Collect streamed response
            full_response = ""
            for chunk in self.generate_stream(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_predict=num_predict
            ):
                full_response += chunk
            return full_response
        
        # Non-streaming request
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        }
        
        if num_predict:
            payload["options"]["num_predict"] = num_predict
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: Optional[int] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_predict: Maximum tokens to generate
        
        Yields:
            Text chunks as they are generated
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k
            }
        }
        
        if num_predict:
            payload["options"]["num_predict"] = num_predict
        
        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        import json
                        chunk_data = json.loads(line)
                        if "response" in chunk_data:
                            yield chunk_data["response"]
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {str(e)}")
    
    def check_connection(self) -> bool:
        """
        Check if Ollama is accessible.
        
        Returns:
            True if connection is successful
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """
        List available models.
        
        Returns:
            List of available model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error listing models: {str(e)}")
    
    def set_model(self, model: str):
        """
        Set the model to use.
        
        Args:
            model: Model name
        """
        self.model = model
