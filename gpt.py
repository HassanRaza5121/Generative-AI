import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. Tokenization and Vocabulary
# -----------------------------

class SimpleTokenizer:
    def __init__(self, text, max_vocab_size=10000):
        self.tokens = self.tokenize(text)
        self.vocab = self.build_vocab(self.tokens, max_vocab_size)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def tokenize(self, text):
        """Simple whitespace tokenizer."""
        return text.lower().split()
    
    def build_vocab(self, tokens, max_vocab_size):
        """Build vocabulary of the most frequent tokens."""
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        vocab = [token for token, _ in sorted_tokens[:max_vocab_size]]
        vocab = ['<PAD>', '<UNK>'] + vocab  # Adding special tokens
        return vocab
    
    def encode(self, tokens):
        """Convert tokens to indices."""
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
    
    def decode(self, indices):
        """Convert indices back to tokens."""
        return [self.idx2word.get(idx, '<UNK>') for idx in indices]

# -----------------------------
# 2. Positional Encoding
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# -----------------------------
# 3. Multi-Head Self-Attention
# -----------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Define linear layers for queries, keys, and values
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output linear layer
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Tensor of shape (batch_size, 1, 1, seq_len) or similar
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.query(x)  # (batch_size, seq_len, d_model)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        context = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
        
        # Final linear layer
        out = self.out(context)  # (batch_size, seq_len, d_model)
        
        return out

# -----------------------------
# 4. Feed-Forward Neural Network
# -----------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# -----------------------------
# 5. Transformer Block
# -----------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-Head Self-Attention
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))  # Add & Norm
        
        # Feed-Forward Network
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))  # Add & Norm
        
        return x

# -----------------------------
# 6. GPT Transformer Model
# -----------------------------

class GPTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=512, dropout=0.1):
        super(GPTTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
            mask: Tensor of shape (batch_size, 1, 1, seq_len) or similar
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Token Embedding
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Positional Encoding
        x = self.position_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Transformer Blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final Layer Normalization
        x = self.layer_norm(x)  # (batch_size, seq_len, d_model)
        
        # Output Layer
        logits = self.output_linear(x)  # (batch_size, seq_len, vocab_size)
        
        return logits

# -----------------------------
# 7. Dataset and DataLoader
# -----------------------------

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        tokens = self.tokenizer.tokenize(text)
        self.token_ids = self.tokenizer.encode(tokens)
        self.num_samples = len(self.token_ids) - seq_length
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.token_ids[idx:idx+self.seq_length], dtype=torch.long),
            torch.tensor(self.token_ids[idx+1:idx+self.seq_length+1], dtype=torch.long)
        )

# -----------------------------
# 8. Training the GPT Transformer
# -----------------------------

def train_gpt():
    # Sample text data
    sample_text = """
    In the beginning God created the heaven and the earth.
    And the earth was without form, and void; and darkness was upon the face of the deep.
    And the Spirit of God moved upon the face of the waters.
    And God said, Let there be light: and there was light.
    And God saw the light, that it was good: and God divided the light from the darkness.
    And God called the light Day, and the darkness he called Night.
    And the evening and the morning were the first day.
    """
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(sample_text)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary Size: {vocab_size}")
    
    # Define dataset and dataloader
    seq_length = 10
    dataset = TextDataset(sample_text, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = GPTTransformer(vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=256, max_seq_len=50)
    model.train()
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)   # (batch_size, seq_len)
            batch_targets = batch_targets.to(device) # (batch_size, seq_len)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)  # (batch_size, seq_len, vocab_size)
            loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print("Training complete.")
    
    return model, tokenizer

# -----------------------------
# 9. Text Generation Function
# -----------------------------

def generate_text(model, tokenizer, prompt, max_length=20):
    model.eval()
    tokens = tokenizer.tokenize(prompt)
    token_ids = tokenizer.encode(tokens)
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
    
    generated = token_ids.copy()
    
    # Define sequence length used during training
    seq_length = input_ids.size(1)
    
    # Move input_ids to the same device as the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)  # (1, seq_len, vocab_size)
            last_token_logits = outputs[0, -1, :]  # (vocab_size)
            probabilities = F.softmax(last_token_logits, dim=0)
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            generated.append(next_token_id)
            # Update input_ids by appending the next token and removing the first token
            input_ids = torch.cat([input_ids[:, 1:], torch.tensor([[next_token_id]], device=device)], dim=1)
    
    generated_text = tokenizer.decode(generated)
    return ' '.join(generated_text)

# -----------------------------
# 10. Main Execution
# -----------------------------

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_gpt()
    
    # Generate text
    prompt = "And God saw"
    generated = generate_text(model, tokenizer, prompt, max_length=20)
    print("\nGenerated Text:")
    print(generated)
