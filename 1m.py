import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import requests
import multiprocessing
import os
import random
from collections import defaultdict
import pandas as pd


# Set the environment variable to disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable Metal Performance Shaders (MPS) backend
torch.backends.mps.enable_ddp = True

class GROKFAST(Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.9, lambda_factor=0.1):
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.lambda_factor = lambda_factor
        super(GROKFAST, self).__init__(params, {})

        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mu'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'base_optimizer': self.base_optimizer,
            'alpha': self.alpha,
            'lambda_factor': self.lambda_factor,
            'state': self.state,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Calculate EMA of gradients
                state['mu'].mul_(self.alpha).add_(p.grad, alpha=1 - self.alpha)

                # Filter gradients
                filtered_grad = p.grad + self.lambda_factor * state['mu']

                # Replace the original gradient with the filtered gradient
                p.grad.copy_(filtered_grad)

        # Call the base optimizer's step function
        self.base_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x)[0]

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        freqs = torch.einsum('i,j->ij', t.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RotarySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        positions = torch.arange(T, device=x.device)
        rot_emb = self.rotary_emb(positions)
        q = (q * rot_emb.cos()) + (rotate_half(q) * rot_emb.sin())
        k = (k * rot_emb.cos()) + (rotate_half(k) * rot_emb.sin())

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = RotarySelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.ln1(x)))
        x = x + self.dropout2(self.ff(self.ln2(x)))
        return x

# Update the SimpleLLM class to use the new TransformerBlock
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.fc(x)
        return logits

class NeurIPSLLMDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        full_text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        encoding = self.tokenizer(full_text,
                                  truncation=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_subset_dataset(file_path, subset_size):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get('data', [])

    # Ensure we don't try to take more data than available
    subset_size = min(subset_size, len(data))

    # Take a random subset of the data
    subset_data = random.sample(data, subset_size)

    return subset_data

class CachedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, cache_dir='./cache'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cache_file = os.path.join(self.cache_dir, f'item_{idx}.pt')
        if os.path.exists(cache_file):
            return torch.load(cache_file)

        item = self.data[idx]
        full_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        encoding = self.tokenizer(full_text,
                                  truncation=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        torch.save(result, cache_file)
        return result

"""def prepare_dataset(tokenizer, max_length, batch_size, file_path, subset_size=None):
    data = load_subset_dataset(file_path, subset_size) if subset_size else load_subset_dataset(file_path, len(data))
    dataset = CachedDataset(data, tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader"""

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def download_dataset(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def download_json(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    return json.loads(response.text)

def prepare_dataset(tokenizer, max_length, batch_size, split='train'):
    splits = {'train': 'train_dataset.json', 'test': 'eval_dataset.json'}
    df = pd.read_json(f"hf://datasets/upaya07/NeurIPS-LLM-data/{splits[split]}")

    # Create our custom dataset
    custom_dataset = NeurIPSLLMDataset(df, tokenizer, max_length)

    # Create the DataLoader
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=4, pin_memory=True)

    return data_loader

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs, gradient_accumulation_steps=4):
    scaler = GradScaler()
    model.train()
    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with autocast(enabled=device.type == 'cuda'):
                outputs = model(input_ids)
                loss = F.cross_entropy(outputs.view(-1, model.fc.out_features), labels.view(-1), ignore_index=-100)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Train Loss: {avg_loss:.4f}")

        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Average Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping!")
            break

        model.train()

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with autocast(enabled=device.type == 'cuda'):
                outputs = model(input_ids)
                loss = F.cross_entropy(outputs.view(-1, model.fc.out_features), labels.view(-1), ignore_index=-100)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def main():
    vocab_size = 50257
    embed_dim = 256
    num_heads = 4
    ff_dim = 512
    num_layers = 4
    max_seq_len = 512
    batch_size = 32
    learning_rate = 2e-4
    epochs = 20  # Increased number of epochs
    subset_size = None  # Increased subset size
    gradient_accumulation_steps = 4

    # GROKFAST hyperparameters
    grokfast_alpha = 0.9
    grokfast_lambda = 0.1

    # Optimal weight decay as per the paper
    weight_decay = 0.01  # This value showed ×3.72 faster generalization

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = SimpleLLM(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len)

    # Check if CUDA is available, otherwise use MPS or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using device: {device}")

    train_loader = prepare_dataset(tokenizer, max_seq_len, batch_size, split='train')
    val_loader = prepare_dataset(tokenizer, max_seq_len, batch_size, split='test')

    base_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    optimizer = GROKFAST(model.parameters(), base_optimizer, alpha=grokfast_alpha, lambda_factor=grokfast_lambda)

    # Update the scheduler to use the base_optimizer


    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = total_steps // 10  # 10% of total steps for warmup

    scheduler = get_linear_schedule_with_warmup(base_optimizer, warmup_steps, total_steps)

    train(model, train_loader, val_loader, optimizer, scheduler, device, epochs, gradient_accumulation_steps)

    prompt = "Question: What is the meaning of life?\nAnswer:"
    generated_text = generate(model, tokenizer, prompt, device)
    print(f"Generated text:\n{generated_text}")

    torch.save(model.state_dict(), "simple_llm_2m_neurips_grokfast.pth")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


# Update the generate function to use the specified device
@torch.inference_mode()
def generate(model, tokenizer, prompt, device, max_tokens=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    for _ in range(max_tokens):
        with autocast(enabled=device.type == 'cuda'):
            outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0])