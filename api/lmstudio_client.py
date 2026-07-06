import os
from typing import Dict, List, Optional, Generator
import requests
from dotenv import load_dotenv

load_dotenv()


class LMStudioClient:
    """Client for interacting with LM Studio API."""
    
    def __init__(self):
        self.base_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234/api/v1")
        self.model = os.getenv("MODEL", "qwen2.5-7b-instruct")
        self.session = requests.Session()
    
    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict:
        """
        Create a chat completion using LM Studio API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Response dictionary with generated text
        """
        url = f"{self.base_url}/chat"
        
        # Extract system prompt and user input from messages
        system_prompt = ""
        user_input_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                system_prompt = content
            else:
                user_input_parts.append(f"{role}: {content}")
        
        user_input = "\n".join(user_input_parts) if user_input_parts else messages[-1].get('content', '') if messages else ''
        
        payload = {
            "model": self.model,
            "input": user_input,
            "temperature": temperature,
            "stream": stream
        }
        
        # Only add context_length if user explicitly provided a max_tokens value
        # LM Studio's context_length controls total context (input + output), not just max output
        if max_tokens is not None:
            payload["context_length"] = max_tokens
        
        # Add system_prompt separately if present
        if system_prompt:
            payload["system_prompt"] = system_prompt
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def chat_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Stream chat completion responses.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Generated text chunks
        """
        url = f"{self.base_url}/chat"
        
        # Extract system prompt and user input from messages
        system_prompt = ""
        user_input_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                system_prompt = content
            else:
                user_input_parts.append(f"{role}: {content}")
        
        user_input = "\n".join(user_input_parts) if user_input_parts else messages[-1].get('content', '') if messages else ''
        
        payload = {
            "model": self.model,
            "input": user_input,
            "temperature": temperature,
            "stream": True
        }
        
        # Only add context_length if user explicitly provided a max_tokens value
        if max_tokens is not None:
            payload["context_length"] = max_tokens
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
        
        response = self.session.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                # LM Studio native SSE format: event: message.delta / data: {...}
                if line.startswith('data: '):
                    data = line[6:]
                    try:
                        import json
                        chunk = json.loads(data)
                        # Look for content in message.delta events
                        if chunk.get('type') == 'message.delta':
                            content = chunk.get('content', '')
                            if content:
                                yield content
                        # Also handle OpenAI-compatible format as fallback
                        elif chunk.get('choices'):
                            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '') or chunk.get('choices', [{}])[0].get('text', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    def load_model(self, model: str) -> Dict:
        """
        Load a model in LM Studio.
        
        Args:
            model: Model identifier to load
            
        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/models/load"
        
        payload = {"model": model}
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def download_model(self, model: str) -> Dict:
        """
        Download a model in LM Studio.
        
        Args:
            model: Model identifier to download
            
        Returns:
            Response dictionary with job_id
        """
        url = f"{self.base_url}/models/download"
        
        payload = {"model": model}
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_download_status(self, job_id: str) -> Dict:
        """
        Get download status for a model download job.
        
        Args:
            job_id: Download job identifier
            
        Returns:
            Response dictionary with download status
        """
        url = f"{self.base_url}/models/download/status/{job_id}"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def get_models(self) -> List[str]:
        """
        Get available models from LM Studio.
        
        Returns:
            List of available model names
        """
        url = f"{self.base_url}/models"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        return [model['id'] for model in data.get('data', [])]
    
    def health_check(self) -> bool:
        """
        Check if LM Studio API is accessible.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/models"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


# Singleton instance
_lmstudio_client = None


def get_lmstudio_client() -> LMStudioClient:
    """Get or create the LM Studio client singleton."""
    global _lmstudio_client
    if _lmstudio_client is None:
        _lmstudio_client = LMStudioClient()
    return _lmstudio_client


def call_llm(messages: List[Dict], temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    """
    Convenience function to call the LLM with messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
    """
    client = get_lmstudio_client()
    response = client.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
    
    # Extract content from response based on LM Studio native API format
    # Response format: {"output": [{"type": "message", "content": "..."}, ...], ...}
    output = response.get('output', [])
    if output and isinstance(output, list):
        content_parts = []
        for item in output:
            if isinstance(item, dict) and item.get('content'):
                content_parts.append(item['content'])
        if content_parts:
            return '\n'.join(content_parts)
    # Fallback: top-level content field
    if 'content' in response:
        return response['content']
    # Fallback: OpenAI-compatible format
    if 'choices' in response and len(response['choices']) > 0:
        return response['choices'][0].get('message', {}).get('content', '') or response['choices'][0].get('text', '')
    # Last resort: return string representation
    return str(response)
