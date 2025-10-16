import json
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.training_data import TrainingData
from src.llm import LLMSettings, LLM
from src.tokenizer import *
from pathlib import Path


class FeedForward(nn.Module):

    def __init__(self, n_dropout: float, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), ## 4* from the paper. just because
            nn.ReLU(),
            # Intro of skips
            nn.Linear(4 * n_embed, n_embed),
            # Finale
            nn.Dropout(n_dropout)
        )

    def forward(self, x) -> torch.tensor:
        return self.net(x)
    


class Block(nn.Module):
    """ Transformer block: comms followed by compute """

    def __init__(self, settings: LLMSettings, n_head: int):
        super().__init__()

        head_size = settings.n_embed // n_head
        self.sa = MultiHeadAttention(settings, n_head, head_size)
        self.ffwd = FeedForward(settings.n_dropout, settings.n_embed)

        self.ln1 = nn.LayerNorm(settings.n_embed)
        self.ln2 = nn.LayerNorm(settings.n_embed)

    def forward(self, x) -> torch.tensor:
        # x+ means we're include both the calc path, and the skip path
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Head(nn.Module):
    """ one head of self-attention : 1'01 """

    def __init__(self, settings: LLMSettings, head_size: int):
        super().__init__()

        self.key = nn.Linear(settings.n_embed, head_size, bias=False)
        self.query = nn.Linear(settings.n_embed, head_size, bias=False)
        self.value = nn.Linear(settings.n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(settings.block_size, settings.block_size)))

        # FInale
        self.dropout = nn.Dropout(settings.n_dropout)


    def forward(self, x) -> torch.tensor:
        B,T,C = x.shape

        k = self.key(x)       # (B,T,C)
        q = self.query(x)     # (B,T,C)

        # compute attention scores ("affinities")
        #wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T) # with scale
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T) # don't comm with past
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)

        # perform weight aggregator of values
        v = self.value(x)   # (B,T,C)
        out = wei @ v       # (B,T,T) @ (B,T,C) -> (B,T,C)

        return out



class MultiHeadAttention(nn.Module):

    def __init__(self, settings: LLMSettings, num_heads: int, head_size: int):
        super().__init__()

        self.heads = nn.ModuleList([Head(settings, head_size) for _ in range(num_heads)])
        # Intro of skips
        # using head_size * num_heads instead of n_embed, as the use of varying scaling
        # parameters means it's no longer square
        self.proj = nn.Linear(head_size * num_heads, settings.n_embed)
        # FInale
        self.dropout = nn.Dropout(settings.n_dropout)

    
    def forward(self, x) -> torch.tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class BigramLM(nn.Module):

  def __init__(self, device: str, settings: LLMSettings, data: Tokenizer | TrainingData):
    super().__init__()

    self.settings = settings    # for access in member fn
    self.device = device
    self.data = data if isinstance(data, TrainingData) else None
    self.tokenizer_inst = data.tokenizer_inst if isinstance(data, TrainingData) else data

    # Temp locals for easy reading
    vocab_size = self.tokenizer_inst.vocab_size
    n_embed = settings.n_embed
    block_size = settings.block_size

    # Setup new stuff
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)

    self.sa_heads = MultiHeadAttention(settings, 4, n_embed//4) ## 4 heads of 8-dimensional self-attention
    self.lm_head = nn.Linear(n_embed, vocab_size)

    # a
    # self.ffwd = FeedForward(n_embed)

    # b
    # self.blocks = nn.Sequential(
    #   Block(n_embed, n_head=4),
    #   Block(n_embed, n_head=4),
    #   Block(n_embed, n_head=4),
    #   nn.LayerNorm(n_embed),
    # )

    # c
    self.blocks = nn.Sequential(*[Block(settings, n_head=settings.n_head) for _ in range(settings.n_layer)])
    self.ln_f =  nn.LayerNorm(n_embed)

    # What do we need to save?
    self.export_attr = ['token_embedding_table', 'position_embedding_table', 'lm_head', 'blocks', 'ln_f']

    # better init, not covered in the original GPT video, but important, will cover in followup video
    self.apply(self._init_weights)

    # torch.manual_seed(1337)


  def __str__(self):
    return f"Bigram model, {self.model_parameters()} parameters, n_embed={self.settings.n_embed} block_size={self.settings.block_size}"


  def model_parameters(self):
    return sum(p.numel() for p in self.parameters())


  def _init_weights(self, module) -> None:
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  def exportModel(self, filepath: str) -> None:
    # These are binary formats
    Path(filepath).mkdir(parents=True, exist_ok=True)
    
    for k in self.export_attr:
      path = "{0}/{1}.pt".format(filepath, k)
      torch.save(getattr(self, k), path)

    with open(f"{filepath}/stats.json", "w") as f:
      f.write(json.dumps(self.settings, default=vars))


  def importModel(self, filepath: str) -> None:
    for k in self.export_attr:
      path = "{0}/{1}.pt".format(filepath, k)
      setattr(self, k, torch.load(path, weights_only=False))


  def forward(self, idx: tuple, targets=None) -> tuple[tuple, any]:
    B, T = idx.shape

    # We change our embeddings to hold both the token and the token's position
    tok_embeddings = self.token_embedding_table(idx) # (B,T,C)
    pos_embeddings = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
    x = tok_embeddings + pos_embeddings # (B,T,C)
    # a
    #x = self.sa_heads(x)
    #x = self.ffwd(x)
    # b
    #x = self.blocks(x) # (B,T,C)
    # logits = self.lm_head(x) # (B,T,vocab_size)
    
    # c
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    
    # Reshape
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits,loss
  

  def generate(self, idx: tuple, max_new_tokens: int) -> tuple:
    # idx is (B,T)
    for _ in range(max_new_tokens):
      # crop to block size
      idx_cond = idx[:, -self.settings.block_size:]

      #get predictions
      logits,loss = self(idx_cond)

      # focus on last time step
      logits = logits[:, -1, :] # becomes (B,C)

      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B,C)

      #sample from distr
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1) because 1=num_samples

      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)

    return idx


  # no back propagation will occur on the data in this fn (so no exta data is stored)
  @torch.no_grad()
  def estimate_loss(self, data: Tokenizer, settings: LLMSettings, eval_iters: int) -> dict:
    out = {}
    self.eval() # change to eval phase

    for split in LLM.SPLITS:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
        X,Y = data.get_batch(split, settings.batch_size, settings.block_size)
        logits, loss = self(X,Y)
        losses[k] = loss.item()
        out[split] = losses.mean()
  
    self.train()

    return out
            

  def trainStep(self, optimizer) -> float:
    # sample a batch of data
    xb,yb = self.data.get_batch(LLM.TRAINING, self.settings.batch_size, self.settings.block_size)

    # eval the loss
    logits,loss = self(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss


  def generateOutputFromTensor(self, initial_tensor: torch.tensor, max_tokens: int) -> str:
     token_tensors = self.generate(initial_tensor, max_new_tokens=max_tokens)
     token_list = token_tensors[0].tolist()
     output = self.tokenizer_inst.decode(token_list)

     return output


  def generateOutput(self, max_tokens: int) -> str:
     # Initial fill with zeros, of type:
     # (1,1) = (B,T)
     return self.generateOutputFromTensor(torch.zeros((1,1), dtype=torch.long, device=self.device), max_tokens)


  def generateOutputFrom(self, initial: any, max_tokens: int) -> str:
    try:
      encoded = self.tokenizer_inst.encode(initial)
      initial_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)

      return self.generateOutputFromTensor(initial_tensor, max_tokens)
    except:
      return "Unable to generate output. Possibly because input can not be tokenised"

