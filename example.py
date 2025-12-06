"""
Example usage of batch_finder package.
"""

import torch
import torch.nn as nn
from batch_finder import batch_finder, find_max_batch, find_max_docs, find_max_timesteps


# Example 1: Simple model
class SimpleModel(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        return self.activation(self.linear(x))


# Example 2: RAG Model
class RAGModel(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8),
            num_layers=6
        )
        self.config = type('Config', (), {'vocab_size': vocab_size, 'hidden_size': hidden_size})()
    
    def forward(self, input_ids, input_ids_encoder=None, **kwargs):
        # input_ids: (batch_size, seq_len)
        # input_ids_encoder: (n_docs, batch_size, seq_len)
        batch_emb = self.embedding(input_ids)
        
        if input_ids_encoder is not None:
            # Process encoder docs
            n_docs, batch_size, seq_len = input_ids_encoder.shape
            docs_emb = self.embedding(input_ids_encoder.view(-1, seq_len))
            docs_emb = docs_emb.view(n_docs, batch_size, seq_len, -1)
            # Simple aggregation: mean over docs
            docs_emb = docs_emb.mean(dim=0)
            batch_emb = batch_emb + docs_emb
        
        output = self.transformer(batch_emb.transpose(0, 1))
        return {'logits': output.transpose(0, 1)}


if __name__ == "__main__":
    print("="*60)
    print("Batch Finder Examples")
    print("="*60)
    
    # Example 1: Find max batch size
    print("\n1. Finding max batch size for SimpleModel...")
    model1 = SimpleModel()
    max_batch = find_max_batch(
        model=model1,
        input_shape=(128, 768),
        max_batch_size=32,
        inference_only=True
    )
    print(f"   Result: Max batch size = {max_batch}")
    
    # Example 2: Find max docs for RAG model
    print("\n2. Finding max docs for RAGModel...")
    model2 = RAGModel()
    max_docs = find_max_docs(
        model=model2,
        batch_size=4,
        seq_len=128,
        max_docs=50,
        inference_only=True
    )
    print(f"   Result: Max docs = {max_docs}")
    
    # Example 3: Find max timesteps
    print("\n3. Finding max timesteps for SimpleModel...")
    max_timesteps = find_max_timesteps(
        model=model1,
        batch_size=4,
        max_timesteps=512,
        inference_only=True
    )
    print(f"   Result: Max timesteps = {max_timesteps}")
    
    # Example 4: Find all at once
    print("\n4. Finding all maximums for SimpleModel...")
    results = batch_finder(
        model=model1,
        batch_size_range=(1, 32),
        docs_range=(1, 20),
        timesteps_range=(64, 256),
        inference_only=True
    )
    print(f"   Results: {results}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)

