#!/usr/bin/env python
# coding: utf-8

# In[ ]:


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers":12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
import torch
import torch.nn as nn


# In[2]:


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Deep learning is so much fun"
txt2 = "It was breakthrough in LLMs"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)


# In[3]:


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0)

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads

    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_length, context_length),
                   diagonal=1)
    )
  def forward(self, x):
    b, num_tokens, d_in = x.shape

    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    attn_scores = queries @ keys.transpose(2, 3)

    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

    attn_scores.masked_fill_(mask_bool, -torch.inf)

    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context_vec = (attn_weights @ values).transpose(1, 2)

    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)

    return context_vec


# In[4]:


import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    self.drop_emb = nn.Dropout(cfg["drop_rate"])

    self.trf_block = nn.Sequential(
        *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )
    self.final_norm = DummyLayerNorm(cfg["emb_dim"])
    self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
    )

  def forward(self, in_idx):
    batch_size, seq_length = in_idx.shape
    token_embeds = self.token_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_length, device=in_idx.device))
    x = token_embeds + pos_embeds
    x = self.drop_emb(x)
    x = self.trf_block(x)
    x = self.final_norm(x)
    logits = self.out_head(x)

    return logits

class DummyTransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = MultiHeadAttention(
        d_in = cfg["emb_dim"],
        d_out = cfg["emb_dim"],
        context_length = cfg["context_length"],
        num_heads = cfg["n_heads"],
        dropout = cfg["drop_rate"],
        qkv_bias = cfg["qkv_bias"]
    )
    self.ff = FeedForward(cfg)
    self.norm1 = DummyLayerNorm(cfg["emb_dim"])
    self.norm2 = DummyLayerNorm(cfg["emb_dim"])
    self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.dropout_shortcut(x)
    x += shortcut


    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.dropout_shortcut(x)
    x += shortcut
    return x

class DummyLayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale*norm_x + self.shift

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5* x* (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 /torch.pi))*(x+0.044715*pow(x, 3))
    ))

class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
        GELU(),
        nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
    )

  def forward(self, x):
    return self.layers(x)


# In[5]:


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)


# In[6]:


total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,}")


# In[7]:


print(model.token_emb.weight.shape)
print(model.out_head.weight.shape)


# In[8]:


actual_total_params_of_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"{actual_total_params_of_gpt2:,}")


# In[9]:


total_size = total_params * 4
total_size_of_gpt2 = actual_total_params_of_gpt2 * 4
total_size_of_gpt2_mb = total_size_of_gpt2 / (1024*1024)
total_size_mb = total_size / (1024 * 1024)
print(total_size_mb, "MB")
print(total_size_of_gpt2_mb, "MB")


# In[10]:


def generate_next_token(model, idx, max_new_tokens, context_size):

  for _ in range(max_new_tokens):
    idx_cont = idx[:, -context_size:]

    with torch.no_grad():
      logits = model(idx_cont)

    logits = logits[:, -1, :]

    probas = torch.softmax(logits, dim=-1)

    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
  return idx


# In[11]:


start_context = "Computer Virus Programming ."
tokenized_text = tokenizer.encode(start_context)
print("encoded text: ", tokenized_text)
encoded_tensor = torch.tensor(tokenized_text).unsqueeze(0)
print("Encoded Text: ", encoded_tensor)


# In[12]:


model.eval()
out = generate_next_token(
    model=model,
    idx = encoded_tensor,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print(out)
print(len(out[0]))


# In[13]:


decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)


# In[14]:


def generate_text_to_token(text, tokenizer):
  encoded = tokenizer.encode(text, allowed_special={'|endoftext|'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)
  return encoded_tensor

def generate_token_to_text(token_ids, tokenizer):
  return tokenizer.decode(token_ids.squeeze(0).tolist())

starting_text_1 = "Computer Programming is a fun course"
starting_text_2 = "I got an A grade in that"
tokenizer = tiktoken.get_encoding("gpt2")
print(generate_text_to_token(starting_text_1, tokenizer))
print(generate_text_to_token(starting_text_2, tokenizer))

token_ids = generate_next_token(
    model=model,
    idx=generate_text_to_token(starting_text_1, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print(token_ids)
decoded_text = generate_token_to_text(token_ids, tokenizer)
print(decoded_text)


# In[15]:


inputs = torch.tensor([[34556, 30297,   318,   257,  1257],
                       [40, 1392,  281,  317, 9559]])

target = torch.tensor([[30297,   318,   257,  1257,  1781],
                       [1392,  281,  317, 9559, 287]])


# In[16]:


with torch.no_grad():
  logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)


# In[17]:


token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(token_ids)


# In[18]:


print(generate_token_to_text(target[0], tokenizer))
print(generate_token_to_text(token_ids[0].flatten(), tokenizer))


# In[19]:


print(generate_token_to_text(target[1], tokenizer))
print(generate_token_to_text(token_ids[1].flatten(), tokenizer))


# In[20]:


nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())


# In[21]:


text_idx = 0
target_probs_1 = probas[text_idx, [0, 1, 2, 3, 4], target[text_idx]]
print(target_probs_1)

text_idx = 1
target_probs_2 = probas[text_idx, [0, 1, 2, 3, 4], target[text_idx]]
print(target_probs_2)


# In[22]:


log_of_probas = torch.log(torch.cat((target_probs_1, target_probs_2)))
print(log_of_probas)


# In[23]:


avg_log_probas = torch.mean(log_of_probas)
print(avg_log_probas)

neg_of_avg = avg_log_probas*-1
print(neg_of_avg)


# In[24]:


import os
import urllib.request

file_path = "book_training_data.txt"
with open(file_path, "r", encoding='utf-8') as f:
  text_data = f.read()


# 

# In[25]:


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print(total_characters)
print(total_tokens)


# In[26]:


from torch.utils.data import DataLoader, Dataset

class GPTdataSetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.inputs = []
    self.targets = []

    token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i+max_length]
      target_chunk = token_ids[i+1:i+1+max_length]

      self.inputs.append(torch.tensor(input_chunk))
      self.targets.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    return self.inputs[idx], self.targets[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTdataSetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# In[27]:


from re import split
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)
train_dataloader = create_dataloader_v1(
    train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0
)

val_dataloader = create_dataloader_v1(
    val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False, num_workers=0
)


# In[28]:


for x, y in train_dataloader:
  print(x.shape, y.shape)

print("Val data")
for x, y in val_dataloader:
  print(x.shape, y.shape)

print(len(train_dataloader))
print(len(val_dataloader))


# In[29]:


def calc_loss_bybatch(input_batch, target_batch, model ,device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_byloader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if len(dataloader) == 0:
         return float("nan")
    elif num_batches is None:
         num_batches = len(dataloader)
    else:
         num_batches = min(num_batches, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
         if i < num_batches:
            loss = calc_loss_bybatch(input_batch, target_batch, model, device)
            total_loss += loss.item()

         else:
            break
    return total_loss / num_batches


# In[30]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(123)

#with torch.no_grad():
   #train_loss = calc_loss_byloader(train_dataloader, model, device)
   #val_loss = calc_loss_byloader(val_dataloader, model, device)

#print(train_loss)
#print(val_loss)


# In[31]:


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_bybatch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# In[32]:


def evaluate_model(model, train_dataloader, val_dataloader, device, eval_iter):
  model.eval()
  with torch.no_grad():
     train_loss = calc_loss_byloader(train_dataloader, model, device, num_batches=eval_iter)
     val_loss = calc_loss_byloader(val_dataloader, model, device, num_batches=eval_iter)

  model.train()
  return train_loss, val_loss


# In[33]:


def generate_and_print_sample(model, tokenizer, device, start_context):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = generate_text_to_token(start_context, tokenizer).to(device)
  with torch.no_grad():
     token_ids = generate_next_token(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
     decoded_text = generate_token_to_text(token_ids, tokenizer)
     print(decoded_text.replace("\n", " "))
     model.train()


# In[ ]:


import time
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, token_seen = train_model_simple(model, train_dataloader, val_dataloader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="Computer Programming is a fun course", tokenizer=tokenizer)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in: {execution_time_minutes:.2f} minutes")


# In[ ]:




