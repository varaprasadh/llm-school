import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import ModuleTree from "../components/viz/building-the-model/ModuleTree";

export default function Chapter() {
  return (
    <>
      <p>
        This is the payoff. For the last several chapters we dissected the transformer one idea at a
        time — <a href="/chapter/self-attention" className="prose-link">self-attention</a>,{" "}
        <a href="/chapter/multi-head-attention" className="prose-link">multiple heads</a>,{" "}
        <a href="/chapter/positional-encoding" className="prose-link">positional encodings</a>,{" "}
        <a href="/chapter/transformer-block" className="prose-link">the feed-forward block, residuals
        and normalization</a>. Each piece made sense on its own. Now we assemble them into{" "}
        <strong>one complete, runnable file</strong>: a GPT in roughly 120 lines of PyTorch, faithful
        to Andrej Karpathy&rsquo;s <code>nanoGPT</code>. By the end you&rsquo;ll read every line and
        think &ldquo;of course.&rdquo;
      </p>
      <p>
        We build it bottom-up, module by module, and each <code>CodeBlock</code> below is a slice of
        the same file, <code>model.py</code>. Paste them together top to bottom and you have a working
        implementation you can train in the <a href="/chapter/training-loop" className="prose-link">next
        chapter</a>.
      </p>

      <Callout type="key" title="What we're building">
        <p>
          A decoder-only transformer language model. It maps a batch of token IDs of shape{" "}
          <M>{"(B, T)"}</M> to logits of shape <M>{"(B, T, V)"}</M> — a score for every vocabulary
          token at every position — and, given targets, returns a single scalar loss. That object,
          trained on enough text, <em>is</em> a base LLM.
        </p>
      </Callout>

      <h2>The plan &amp; the config</h2>
      <p>
        Before any layers, we name the model&rsquo;s shape. A small <code>@dataclass</code> holds every
        architectural hyperparameter in one place, so the modules below can take a single{" "}
        <code>config</code> object instead of a dozen loose arguments. This is exactly how nanoGPT,
        GPT-2, and most research code organize things.
      </p>
      <ul>
        <li>
          <code>vocab_size</code> — number of distinct tokens (the embedding table and output head
          are both this wide).
        </li>
        <li>
          <code>block_size</code> — the maximum context length <M>{"T"}</M> the model can ever see.
          It fixes the size of the positional table and the causal mask.
        </li>
        <li>
          <code>n_layer</code>, <code>n_head</code>, <code>n_embd</code> — depth (number of blocks),
          number of attention heads, and the residual-stream width <M>{"d"}</M>. We require{" "}
          <M>{"d"}</M> to be divisible by <code>n_head</code>.
        </li>
        <li>
          <code>dropout</code> — regularization probability, and <code>bias</code> — whether{" "}
          <code>Linear</code>/<code>LayerNorm</code> layers carry bias terms.
        </li>
      </ul>

      <CodeBlock language="python" filename="model.py" highlight={[14, 15, 16]}>
{`import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024    # max context length (T)
    vocab_size: int = 50304   # size of the token vocabulary (V)
    n_layer: int = 12         # number of transformer blocks
    n_head: int = 12          # number of attention heads per block
    n_embd: int = 768         # residual-stream / embedding width (d)
    dropout: float = 0.0      # dropout probability
    bias: bool = True         # use bias in Linear and LayerNorm layers?`}
      </CodeBlock>

      <Callout type="note" title="Why these defaults">
        <p>
          They describe GPT-2 (124M): 12 layers, 12 heads, width 768, context 1024. The vocabulary is
          padded from GPT-2&rsquo;s natural 50,257 up to 50,304 — the nearest multiple of 64 — purely
          so matrix shapes are GPU-friendly. We&rsquo;ll instantiate something <em>much</em> smaller for
          the sanity check at the end.
        </p>
      </Callout>

      <h2>Causal self-attention</h2>
      <p>
        First the engine. In{" "}
        <a href="/chapter/self-attention" className="prose-link">Chapter 8</a> we wrote attention for a
        single head over a single sequence. Here we do it for real:{" "}
        <strong>batched, multi-head, and causal</strong>, all in one module. Three implementation
        choices make this efficient:
      </p>
      <ol>
        <li>
          <strong>One fused QKV projection.</strong> Instead of three separate <code>Linear</code>
          layers we use a single <code>c_attn</code> of shape <M>{"d \\to 3d"}</M> and split its output
          into <M>{"Q, K, V"}</M>. One big matmul beats three small ones.
        </li>
        <li>
          <strong>Reshape into heads.</strong> We split the last dimension into{" "}
          <code>n_head</code> chunks of size <M>{"d/\\text{n\\_head}"}</M> and move the head axis next
          to the batch axis, giving tensors of shape <M>{"(B, n_h, T, d/n_h)"}</M>. Attention then runs
          on all heads in parallel.
        </li>
        <li>
          <strong>A registered causal mask buffer.</strong> A lower-triangular matrix of ones, stored
          with <code>register_buffer</code> so it moves to the GPU with the model but is{" "}
          <em>not</em> a learnable parameter. Where it is zero, we set scores to{" "}
          <M>{"-\\infty"}</M> so the softmax assigns them zero weight — the model cannot peek at the
          future.
        </li>
      </ol>

      <CodeBlock language="python" filename="model.py" highlight={[10, 25, 27, 28]}>
{`class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # fused projection for query, key, value (d -> 3d)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection (d -> d)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask, a (1, 1, T, T) lower-triangular buffer (not a parameter)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()                       # batch, time, channels (= n_embd)

        # project to q, k, v and split the last dim into three
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape each into (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # scaled dot-product attention with a causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                              # (B, nh, T, head_size)

        # re-assemble all head outputs side by side, then project back to d
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y`}
      </CodeBlock>

      <p>
        Trace the shapes once and it locks in. The query/key product{" "}
        <M>{"q\\,k^{\\top}"}</M> is <M>{"(B, n_h, T, T)"}</M> — one attention matrix per head per
        batch element. We scale by <M>{"1/\\sqrt{d/n_h}"}</M> (the per-head dimension), mask, softmax
        over the last axis, then weight the values. The final{" "}
        <code>transpose(1, 2).contiguous().view(B, T, C)</code> is the &ldquo;concatenate the
        heads&rdquo; step: it glues every head&rsquo;s output back into a single width-<M>{"d"}</M>
        vector per token.
      </p>

      <Callout type="pitfall" title="Mask with −∞, not 0">
        <p>
          A natural-looking bug is to zero out the masked <em>scores</em>. That&rsquo;s wrong:{" "}
          <M>{"e^{0} = 1"}</M>, so a zeroed score still receives weight after the softmax. You must
          fill masked positions with <code>float('-inf')</code> <em>before</em> the softmax, because{" "}
          <M>{"e^{-\\infty} = 0"}</M>. That is what makes the future genuinely invisible.
        </p>
      </Callout>

      <Callout type="industry" title="Flash Attention in one line">
        <p>
          Modern PyTorch ships a fused, memory-efficient kernel. The four lines from{" "}
          <code>att = q @ k...</code> down to <code>y = att @ v</code> can be replaced by a single
          call:
        </p>
        <CodeBlock language="python" filename="model.py (alternative)">
{`y = F.scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0,
    is_causal=True,   # builds the causal mask internally — no buffer needed
)`}
        </CodeBlock>
        <p>
          This is <a href="/chapter/inference-optimization" className="prose-link">FlashAttention</a>:
          mathematically identical, but it never materializes the full{" "}
          <M>{"T \\times T"}</M> matrix in memory. We keep the explicit version above because{" "}
          <em>seeing</em> every step is the point of this chapter.
        </p>
      </Callout>

      <h2>The MLP (feed-forward network)</h2>
      <p>
        After attention mixes information <em>across</em> tokens, the MLP processes each token{" "}
        <em>independently</em>, giving the model room to transform what it gathered. The recipe is
        fixed across virtually every transformer: project up to <M>{"4d"}</M>, apply a nonlinearity,
        project back down to <M>{"d"}</M>. The <M>{"4\\times"}</M> expansion is the standard GPT
        ratio. We use <strong>GELU</strong>, the smooth activation GPT-2 used.
      </p>

      <CodeBlock language="python" filename="model.py" highlight={[5, 6, 7]}>
{`class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)      # (B, T, d) -> (B, T, 4d)
        x = self.gelu(x)
        x = self.c_proj(x)    # (B, T, 4d) -> (B, T, d)
        x = self.dropout(x)
        return x`}
      </CodeBlock>

      <p>
        Notice the shape leaves and returns at <M>{"(B, T, d)"}</M> — the MLP, like attention, is a
        function the residual stream can add into without changing dimensions. Also notice the
        arithmetic: with two <M>{"d \\times 4d"}</M> matrices, this single module holds roughly{" "}
        <M>{"8d^2"}</M> parameters, about two-thirds of every block. The MLP, not attention, is where
        most of a transformer&rsquo;s weights live.
      </p>

      <h2>The Block</h2>
      <p>
        A <strong>Block</strong> wires attention and the MLP together with the two structural ideas
        from <a href="/chapter/transformer-block" className="prose-link">Chapter 11</a>:{" "}
        <strong>residual connections</strong> and <strong>pre-normalization</strong>. The whole block
        is two lines:
      </p>
      <MB>{String.raw`x \;\leftarrow\; x + \text{attn}\big(\text{LN}_1(x)\big), \qquad x \;\leftarrow\; x + \text{mlp}\big(\text{LN}_2(x)\big)`}</MB>
      <p>
        Each sub-layer reads a <em>normalized</em> copy of the stream and adds its result back. The
        un-normalized stream <M>{"x"}</M> flows straight through the <M>{"+"}</M>, forming an
        uninterrupted gradient highway from the loss all the way down to the embeddings — which is
        what lets us stack dozens of these without the signal vanishing.
      </p>

      <CodeBlock language="python" filename="model.py" highlight={[10, 11]}>
{`class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's own forbids bias=False)."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # pre-norm attention sub-layer
        x = x + self.mlp(self.ln_2(x))    # pre-norm MLP sub-layer
        return x`}
      </CodeBlock>

      <Callout type="note" title="Pre-norm vs. post-norm">
        <p>
          The original 2017 Transformer put LayerNorm <em>after</em> each sub-layer (
          <code>x = LN(x + attn(x))</code>). Every modern LLM instead normalizes <em>before</em> the
          sub-layer, as above. Pre-norm keeps the residual path clean and additive, which makes deep
          stacks dramatically more stable to train. The one-line difference matters enormously.
        </p>
      </Callout>

      <h2>The full GPT module</h2>
      <p>
        Now the container. The constructor assembles everything: a token embedding table{" "}
        <code>wte</code>, a positional embedding table <code>wpe</code>, dropout, the{" "}
        <code>ModuleList</code> of blocks, a final <code>LayerNorm</code>, and the linear{" "}
        <code>lm_head</code> that turns each token&rsquo;s vector into vocabulary logits. Three details
        deserve attention:
      </p>
      <ul>
        <li>
          <strong>Weight tying.</strong> The line{" "}
          <code>self.transformer.wte.weight = self.lm_head.weight</code> makes the input embedding and
          the output projection the <em>same</em> matrix. This is a classic trick: it saves{" "}
          <M>{"V \\times d"}</M> parameters (tens of millions at GPT-2 scale) and improves quality,
          since &ldquo;the vector for token <M>{"i"}</M>&rdquo; and &ldquo;how strongly to predict
          token <M>{"i"}</M>&rdquo; are naturally related.
        </li>
        <li>
          <strong>Custom initialization.</strong> We init all weights to a small Gaussian (
          <M>{"\\mathcal{N}(0,\\,0.02)"}</M>), and scale the residual projections by an extra{" "}
          <M>{"1/\\sqrt{2 n_\\text{layer}}"}</M> — the GPT-2 trick that keeps the variance of the
          residual stream from growing as we stack more layers.
        </li>
        <li>
          <strong><code>get_num_params</code>.</strong> A small helper to report the parameter count
          (subtracting the position embeddings, by nanoGPT convention).
        </li>
      </ul>

      <CodeBlock language="python" filename="model.py" highlight={[18, 19, 24]}>
{`class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),   # token   embeddings
            wpe  = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: the embedding table IS the output projection
        self.transformer.wte.weight = self.lm_head.weight

        # initialize all weights
        self.apply(self._init_weights)
        # special scaled init for the residual projections (GPT-2 paper, §2.3)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n`}
      </CodeBlock>

      <Callout type="pitfall" title="Tie the weights, not the values">
        <p>
          Weight tying must <em>share the same tensor object</em>, not copy numbers once. The
          assignment <code>wte.weight = lm_head.weight</code> makes both attributes point at one{" "}
          <code>nn.Parameter</code>, so a single gradient updates both roles every step. If you
          instead did <code>wte.weight.data = lm_head.weight.data.clone()</code> they would drift apart
          immediately and you&rsquo;d silently double the embedding parameters.
        </p>
      </Callout>

      <h2>The forward pass</h2>
      <p>
        Here is where the data actually flows, and it&rsquo;s short. Embed the tokens, add positional
        information, run the stack, normalize, and project to logits. When training{" "}
        <code>targets</code> are supplied, we also compute the loss with{" "}
        <code>F.cross_entropy</code>; at inference time we skip that work and only score the last
        position.
      </p>
      <p>
        The one subtlety is <code>cross_entropy</code>&rsquo;s expected shapes. It wants logits of
        shape <M>{"(N, V)"}</M> and integer targets of shape <M>{"(N,)"}</M>, so we flatten the batch
        and time axes together: <M>{"(B, T, V) \\to (B\\cdot T, V)"}</M>. Each of the{" "}
        <M>{"B\\cdot T"}</M> positions is one independent next-token classification problem.
      </p>

      <CodeBlock language="python" filename="model.py" highlight={[8, 9, 10, 11, 21]}>
{`    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \\
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)   # (T,)

        # token + positional embeddings, then dropout
        tok_emb = self.transformer.wte(idx)   # (B, T, d)
        pos_emb = self.transformer.wpe(pos)   # (T, d) -> broadcasts over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # the transformer stack
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)          # final LayerNorm, still (B, T, d)

        if targets is not None:
            # training: logits for every position, then cross-entropy loss
            logits = self.lm_head(x)          # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, V)
                targets.view(-1),                    # (B*T,)
                ignore_index=-1,
            )
        else:
            # inference: only the last position is needed to predict next token
            logits = self.lm_head(x[:, [-1], :])     # (B, 1, V)
            loss = None

        return logits, loss`}
      </CodeBlock>

      <Callout type="key" title="Cross-entropy is the whole training signal">
        <p>
          Everything the model ever learns flows from that one <code>F.cross_entropy</code> call. It
          compares the predicted distribution at each position to the token that actually came next,
          and produces the scalar we differentiate. We derive it in full in{" "}
          <a href="/chapter/training-objective" className="prose-link">Chapter 12</a>; here it is just
          two lines, because PyTorch fuses the softmax and the negative-log-likelihood for numerical
          stability.
        </p>
      </Callout>

      <h2>Generation</h2>
      <p>
        Training learns the distribution; <strong>generation</strong> samples from it. The loop is
        the autoregressive idea made literal: predict the next token, append it, feed the longer
        sequence back in, repeat. Four knobs shape the output:
      </p>
      <ul>
        <li>
          <strong>Cropping.</strong> If the running sequence grows past <code>block_size</code>, we
          keep only the most recent <code>block_size</code> tokens — the positional table and mask
          aren&rsquo;t defined beyond that.
        </li>
        <li>
          <strong>Temperature.</strong> Dividing logits by a temperature <M>{"\\tau"}</M> before the
          softmax sharpens (<M>{"\\tau < 1"}</M>) or flattens (<M>{"\\tau > 1"}</M>) the distribution
          — less or more random.
        </li>
        <li>
          <strong>Top-<M>{"k"}</M>.</strong> Optionally keep only the <M>{"k"}</M> most likely tokens
          and renormalize, so the tail of unlikely junk can never be sampled.
        </li>
        <li>
          <strong>Sampling.</strong> <code>torch.multinomial</code> draws one token from the final
          probabilities. (Greedy decoding is just the <M>{"\\tau \\to 0"}</M> limit.)
        </li>
      </ul>
      <p>
        The whole loop runs under <code>@torch.no_grad()</code> — we&rsquo;re not learning, so there&rsquo;s
        no reason to build a gradient graph.
      </p>

      <CodeBlock language="python" filename="model.py" highlight={[7, 8, 12, 21]}>
{`    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """idx is a (B, T) tensor of context tokens. Extends it by
        max_new_tokens, sampling one token at a time."""
        for _ in range(max_new_tokens):
            # crop the context to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size \\
                       else idx[:, -self.config.block_size:]
            # forward the model; logits is (B, 1, V) at inference time
            logits, _ = self(idx_cond)
            # take the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature        # (B, V)
            # optionally crop to the top-k most likely tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # softmax -> probabilities, then sample one token
            probs = F.softmax(logits, dim=-1)              # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            # append the sampled token and continue
            idx = torch.cat((idx, idx_next), dim=1)        # (B, T+1)
        return idx`}
      </CodeBlock>

      <Callout type="pitfall" title="Forgetting to crop the context">
        <p>
          The single easiest way to break generation: drop the crop on line 7&ndash;8 and feed the
          full, ever-growing sequence in. The moment it exceeds <code>block_size</code>, the position
          embedding <code>wpe</code> is indexed out of range and you get a runtime crash (or, with a
          learned mask, silent garbage). Always slice to the last <code>block_size</code> tokens — the
          model has no positions defined past it.
        </p>
      </Callout>

      <h2>Parameter count &amp; sanity check</h2>
      <p>
        That&rsquo;s the entire model. Before trusting it, do what you should do with <em>any</em> network
        you write: instantiate a tiny version, count its parameters, push random input through it, and
        assert the output shape is what you expect. If this runs without error, the wiring is correct.
      </p>

      <CodeBlock language="python" filename="model.py" highlight={[6, 12, 16, 17]}>
{`if __name__ == "__main__":
    torch.manual_seed(1337)

    # a tiny config — small enough to run on a laptop CPU in a blink
    config = GPTConfig(
        vocab_size=65, block_size=64,
        n_layer=4, n_head=4, n_embd=128, dropout=0.0,
    )
    model = GPT(config)
    print(f"parameters: {model.get_num_params()/1e6:.2f}M")   # -> 0.80M

    # a fake batch of token IDs: B=2 sequences of length T=16
    B, T = 2, 16
    idx     = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets)          # training-mode forward
    assert logits.shape == (B, T, config.vocab_size)
    print("logits:", tuple(logits.shape), "loss:", round(loss.item(), 3))

    # untrained model -> loss should be near ln(vocab_size)
    print("ln(V) =", round(math.log(config.vocab_size), 3))   # ~4.17

    # generate 20 tokens from a single start token
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20, temperature=0.8, top_k=10)
    print("generated ids:", out[0].tolist())`}
      </CodeBlock>

      <p>
        Two things are worth checking beyond &ldquo;it didn&rsquo;t crash.&rdquo; First, the parameter count:
        this config reports <strong>0.80M</strong> non-embedding parameters (0.81M including the
        position table). Second, the loss. An <em>untrained</em> model has no idea what comes next, so
        it spreads probability roughly uniformly over all <M>{"V"}</M> tokens. The cross-entropy of a
        uniform distribution over <M>{"V"}</M> outcomes is exactly <M>{"\\ln V"}</M>:
      </p>
      <MB>{String.raw`\text{loss}_{\text{init}} \;\approx\; -\ln\!\frac{1}{V} \;=\; \ln V \;=\; \ln 65 \;\approx\; 4.17`}</MB>
      <p>
        Seeing your fresh model report a loss near <M>{"\\ln V"}</M> is the classic first-step
        smoke-test — it confirms the output head and loss are wired up sanely. If you saw{" "}
        <M>{"0.01"}</M> or <M>{"40"}</M>, something is wrong before you&rsquo;ve spent a single GPU-hour.
      </p>

      <p>
        The figure below is that same model laid out as an object tree — exactly what{" "}
        <code>print(model)</code> would show you, but live. Drag <code>n_layer</code> and{" "}
        <code>n_embd</code> and watch every module&rsquo;s parameter count, and the running total, update.
        Hover any node to see its type, output shape, and share of the parameters. Notice three things:
        the blocks dominate the count; inside each block the MLP outweighs attention; and{" "}
        <code>lm_head</code> reports <em>zero</em> parameters because it is tied to <code>wte</code>.
      </p>

      <Figure
        n="14.1"
        title="The GPT module tree, with live parameter counts"
        caption="Every box is a real module from model.py. The total scales roughly with n_layer × n_embd² — depth is linear, width is quadratic. That d² is why widening a model is so much more expensive than deepening it."
      >
        <ModuleTree />
      </Figure>

      <Callout type="industry" title="Where the parameters actually go">
        <p>
          For a transformer of width <M>{"d"}</M> and depth <M>{"L"}</M>, each block holds about{" "}
          <M>{"12 d^2"}</M> parameters (<M>{"4d^2"}</M> in attention&rsquo;s four projections,{" "}
          <M>{"8d^2"}</M> in the MLP&rsquo;s two), so the trunk is roughly <M>{"12 L d^2"}</M>. The
          embeddings add <M>{"V d"}</M>. At GPT-2 scale (<M>{"L=12, d=768"}</M>) that&rsquo;s{" "}
          <M>{"\\approx 85\\text{M}"}</M> in the blocks plus <M>{"\\approx 39\\text{M}"}</M> in the
          (tied) embeddings — about 124M total. This back-of-envelope formula{" "}
          <M>{"N \\approx 12 L d^2"}</M> is worth memorizing; we use it directly in{" "}
          <a href="/chapter/scaling-laws" className="prose-link">Chapter 17</a> on scaling laws.
        </p>
      </Callout>

      <h2>The complete file, in one breath</h2>
      <p>
        Step back and look at what you just read. The nine code blocks above are one file. Read it as
        a story:
      </p>
      <ol>
        <li>a <code>config</code> names the shape;</li>
        <li><code>CausalSelfAttention</code> lets every token gather from its past;</li>
        <li><code>MLP</code> thinks about what each token gathered;</li>
        <li>a <code>Block</code> wraps both in pre-norm residual connections;</li>
        <li><code>GPT</code> stacks the blocks between embeddings and an output head;</li>
        <li><code>forward</code> flows data through and computes the loss;</li>
        <li><code>generate</code> samples new text autoregressively;</li>
        <li>a sanity check proves the whole thing runs.</li>
      </ol>
      <p>
        There is no hidden magic, no special sauce we skipped. The model that powers a frontier LLM is{" "}
        <em>this exact file</em> — scaled to thousands of times more layers-times-width, trained on
        trillions of tokens, with the efficiency tricks of{" "}
        <a href="/chapter/distributed-training" className="prose-link">Part IV</a>. But the
        architecture you can now write from memory.
      </p>

      <Figure
        n="14.2"
        title="model.py at a glance"
        caption="The dependency graph of the file. Arrows are 'contains / calls'. Read bottom-up to build it; read top-down to run it."
      >
        <ModuleMap />
      </Figure>

      <h2>Bridge to Chapter 15: training the thing</h2>
      <p>
        We have a model and a loss, but right now the weights are random Gaussians — it generates
        gibberish. The next chapter brings it to life. We&rsquo;ll feed it real token batches, call{" "}
        <code>loss.backward()</code> to get gradients, and step an <strong>AdamW</strong> optimizer to
        nudge those weights downhill. Wrapped in a loop with gradient accumulation, mixed precision,
        learning-rate warmup and checkpointing, this same <code>GPT</code> class will watch its loss
        fall from <M>{"\\ln V"}</M> toward something fluent — and start producing text that looks like
        the data it learned from.
      </p>
      <p>
        Keep <code>model.py</code> open. In{" "}
        <a href="/chapter/training-loop" className="prose-link">Chapter 15</a> we import this exact{" "}
        <code>GPT</code> and put it to work.
      </p>
    </>
  );
}

/**
 * A small static SVG dependency map of model.py — the "at a glance" figure.
 * Boxes are modules; arrows are "contains / calls". Deterministic, no state.
 */
function ModuleMap() {
  const boxes = [
    { id: "GPT", x: 250, y: 16, w: 160, h: 38, c: "#5b7dff", label: "GPT", sub: "(B,T) → (B,T,V)" },
    { id: "fwd", x: 40, y: 84, w: 150, h: 34, c: "#5b7dff", label: "forward()", sub: "+ loss" },
    { id: "gen", x: 470, y: 84, w: 150, h: 34, c: "#5b7dff", label: "generate()", sub: "sampling loop" },
    { id: "emb", x: 250, y: 84, w: 160, h: 34, c: "#34d399", label: "wte + wpe", sub: "embeddings" },
    { id: "block", x: 250, y: 150, w: 160, h: 38, c: "#a855f7", label: "Block × n_layer", sub: "pre-norm residual" },
    { id: "attn", x: 120, y: 222, w: 160, h: 38, c: "#f59e0b", label: "CausalSelfAttention", sub: "c_attn · c_proj" },
    { id: "mlp", x: 380, y: 222, w: 160, h: 38, c: "#fb7185", label: "MLP", sub: "c_fc · GELU · c_proj" },
    { id: "ln", x: 250, y: 294, w: 160, h: 34, c: "#22d3ee", label: "LayerNorm ×3", sub: "ln_1 · ln_2 · ln_f" },
  ];
  const byId = Object.fromEntries(boxes.map((b) => [b.id, b]));
  const edges = [
    ["GPT", "fwd"],
    ["GPT", "gen"],
    ["GPT", "emb"],
    ["GPT", "block"],
    ["block", "attn"],
    ["block", "mlp"],
    ["block", "ln"],
  ];
  const center = (b) => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });

  return (
    <div className="overflow-x-auto scrollbar-thin">
      <svg width={660} height={344} className="mx-auto block" style={{ maxWidth: "100%" }}>
        {edges.map(([a, b], i) => {
          const pa = center(byId[a]);
          const pb = center(byId[b]);
          return (
            <line
              key={i}
              x1={pa.x}
              y1={byId[a].y + byId[a].h}
              x2={pb.x}
              y2={byId[b].y}
              stroke="#27314a"
              strokeWidth={1.5}
            />
          );
        })}
        {boxes.map((b) => (
          <g key={b.id}>
            <rect
              x={b.x}
              y={b.y}
              width={b.w}
              height={b.h}
              rx={8}
              fill="#141a2e"
              stroke={b.c}
              strokeWidth={1.5}
            />
            <rect x={b.x} y={b.y} width={5} height={b.h} rx={2} fill={b.c} />
            <text
              x={b.x + b.w / 2}
              y={b.sub ? b.y + 17 : b.y + b.h / 2 + 4}
              textAnchor="middle"
              fontSize="12.5"
              fontWeight="600"
              fontFamily="JetBrains Mono, monospace"
              fill="#e2e8f0"
            >
              {b.label}
            </text>
            {b.sub && (
              <text
                x={b.x + b.w / 2}
                y={b.y + 30}
                textAnchor="middle"
                fontSize="9.5"
                fontFamily="JetBrains Mono, monospace"
                fill="#64748b"
              >
                {b.sub}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}
