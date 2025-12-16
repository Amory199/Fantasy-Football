"""
LLM Layer for Graph-RAG: Generate answers using retrieved context
Compares 3 different HuggingFace models using Chat Completions API
"""

from huggingface_hub import InferenceClient
from typing import Dict, List, Optional
import time
import os

class LLMLayer:
    """
    Handles LLM inference using HuggingFace models
    Compares multiple models for answer quality
    """
    
    # Working models on HuggingFace Inference API (serverless) - Dec 2025
    # Using Chat Completions API which is more reliable than text_generation
    MODELS = {
        'llama': 'meta-llama/Llama-3.2-3B-Instruct',  # Meta's Llama 3.2
        'qwen': 'Qwen/Qwen2.5-72B-Instruct',  # Alibaba's Qwen 2.5
        'gemma': 'google/gemma-2-9b-it',  # Google's Gemma 2
    }
    
    # System prompt for FPL assistant
    SYSTEM_PROMPT = """You are a Fantasy Premier League expert assistant. Your role is to provide accurate, helpful answers about FPL player statistics and performance based on the knowledge graph data provided.

Instructions:
- Answer the question using only the information provided in the context
- Be specific and cite relevant statistics
- If the context doesn't contain enough information to fully answer the question, say so
- Keep your answer concise but informative
- Use natural, conversational language"""
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize with optional HuggingFace token
        Note: Token required for reliable access to models
        """
        self.client = InferenceClient(token=hf_token)
        self.has_token = bool(hf_token)
    
    def create_prompt(self, context: str, question: str) -> str:
        """
        Create structured prompt with persona, context, and task
        (Used for display/logging - actual API uses messages format)
        """
        prompt = f"""You are a Fantasy Premier League expert assistant. Your role is to provide accurate, helpful answers about FPL player statistics and performance based on the knowledge graph data provided.

Context from Knowledge Graph:
{context}

User Question: {question}

Instructions:
- Answer the question using only the information provided in the context
- Be specific and cite relevant statistics
- If the context doesn't contain enough information to fully answer the question, say so
- Keep your answer concise but informative
- Use natural, conversational language

Answer:"""
        return prompt
    
    def _create_messages(self, context: str, question: str) -> List[Dict[str, str]]:
        """
        Create messages array for Chat Completions API
        """
        user_message = f"""Context from Knowledge Graph:
{context}

User Question: {question}

Please answer the question based on the context provided."""
        
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    
    def query_model(self, model_key: str, prompt: str, max_tokens: int = 300) -> Dict:
        """
        Query a specific model using Chat Completions API and return response with metadata
        Note: prompt parameter kept for backward compatibility but we extract context/question
        """
        model_name = self.MODELS[model_key]
        start_time = time.time()
        
        try:
            # Extract context and question from the formatted prompt for chat API
            # Or use prompt directly if it's just a question
            if "Context from Knowledge Graph:" in prompt and "User Question:" in prompt:
                # Parse the structured prompt
                parts = prompt.split("User Question:")
                context_part = parts[0].split("Context from Knowledge Graph:")[-1].strip()
                question_part = parts[1].split("Instructions:")[0].strip()
                messages = self._create_messages(context_part, question_part)
            else:
                # Simple question
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            
            # Use Chat Completions API (more reliable than text_generation)
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
            )
            
            answer = response.choices[0].message.content
            elapsed_time = time.time() - start_time
            
            return {
                'model': model_key,
                'model_name': model_name,
                'answer': answer.strip() if answer else "",
                'time_seconds': round(elapsed_time, 2),
                'success': True,
                'error': None
            }
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            error = str(e).strip()
            if not error:
                error = f"{type(e).__name__}: {e!r}"
            return {
                'model': model_key,
                'model_name': model_name,
                'answer': None,
                'time_seconds': round(elapsed_time, 2),
                'success': False,
                'error': error
            }
    
    def compare_models(self, context: str, question: str) -> List[Dict]:
        """
        Get answers from all 3 models and compare
        """
        prompt = self.create_prompt(context, question)
        results = []
        
        for model_key in self.MODELS.keys():
            print(f"\nQuerying {model_key}...")
            result = self.query_model(model_key, prompt)
            results.append(result)
            
            if result['success']:
                print(f"[OK] {model_key} responded in {result['time_seconds']}s")
            else:
                print(f"[FAIL] {model_key} failed: {result['error']}")
        
        return results
    
    def format_comparison(self, results: List[Dict]) -> str:
        """
        Format model comparison results for display
        """
        output = []
        output.append("\n" + "="*70)
        output.append("MODEL COMPARISON")
        output.append("="*70)
        
        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. {result['model'].upper()} ({result['model_name']})")
            output.append(f"   Time: {result['time_seconds']}s")
            
            if result['success']:
                output.append(f"   Answer:")
                # indent the answer
                answer_lines = result['answer'].split('\n')
                for line in answer_lines:
                    output.append(f"   {line}")
            else:
                output.append(f"   ERROR: {result['error']}")
            
            output.append("")
        
        return "\n".join(output)


def main():
    """Test LLM layer with sample context"""
    print("="*60)
    print("TESTING LLM LAYER")
    print("="*60)
    
    # sample context and question
    context = """User Question: Who scored the most goals in 2022-23?

Query Intent: top_players

Structured Data from Knowledge Graph:
  1. player: Erling Haaland, total_goals: 36
  2. player: Harry Kane, total_goals: 30
  3. player: Ivan Toney, total_goals: 20
  4. player: Callum Wilson, total_goals: 18
  5. player: Mohamed Salah, total_goals: 19

Semantically Similar Players:
  1. Erling Haaland (similarity: 0.99)
  2. Harry Kane (similarity: 0.98)
  3. Ivan Toney (similarity: 0.97)
"""
    
    question = "Who scored the most goals in 2022-23?"
    
    # initialize LLM layer (reads token from env if available)
    hf_token = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    llm = LLMLayer(hf_token if hf_token else None)
    
    # show the prompt
    prompt = llm.create_prompt(context, question)
    print("\nGenerated Prompt:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    
    # compare models
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)
    
    results = llm.compare_models(context, question)
    
    # show comparison
    comparison = llm.format_comparison(results)
    print(comparison)
    
    print("\n" + "="*60)
    print("LLM TESTING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
