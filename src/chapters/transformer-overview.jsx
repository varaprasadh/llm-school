import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import TransformerStack from "../components/viz/transformer-overview/TransformerStack";

export default function Chapter() {
  return (
    <>
      <p>
        You have met the pieces individually — tokens become integer IDs, IDs become{" "}
        <a href="/chapter/embeddings" className="prose-link">
          vectors
        </a>
        , and you have seen a tantalizing fourteen-line forward pass. This chapter is the{" "}
        <strong>map</strong>. We are going to take a single sequence of tokens and follow it all the
        way through a modern <strong>decoder-only transformer</strong>, naming every component it
        passes through and tracking the exact shape of the tensor at each step. By the end you will
        know <em>what</em> every box does and <em>where</em> it sits. The deep dives — how attention
        actually works, why we split it into heads, how position is encoded, what the feed-forward
        layer is for — each get their own chapter right after this one. Think of this as the wiring
        diagram you will keep coming back to.
      </p>
      <p>
        To keep the map readable we name each block but defer its internals to a dedicated chapter:{" "}
        <a href="/chapter/self-attention" className="prose-link">
          self-attention (Ch. 8)
        </a>
        ,{" "}
        <a href="/chapter/multi-head-attention" className="prose-link">
          multi-head attention (Ch. 9)
        </a>
        ,{" "}
        <a href="/chapter/positional-encoding" className="prose-link">
          positional encoding (Ch. 10)
        </a>
        , and{" "}
        <a href="/chapter/transformer-block" className="prose-link">
          feed-forward, norm &amp; residuals (Ch. 11)
        </a>
        . Read this chapter for the shape of the whole thing; read those for the machinery inside
        each box.
      </p>

      <h2>The big picture</h2>
      <p>
        Strip away the jargon and a decoder-only transformer is a tall stack of <em>identical</em>{" "}
        layers sandwiched between an input stage and an output stage. Text comes in at the bottom as
        token IDs; logits — scores for the next token — come out at the top. In between, the same
        kind of block is applied over and over, each one nudging the representation a little closer
        to "what comes next here."
      </p>
      <p>The whole machine, bottom to top, is just six stages:</p>
      <ol>
        <li>
          <strong>Embed.</strong> Look up each token ID in a table to get a vector, and add a
          positional vector so order is preserved.
        </li>
        <li>
          <strong>Stack of blocks.</strong> Apply <M>{"N"}</M> identical transformer blocks. Each
          block mixes information across tokens (attention) and transforms each token on its own
          (a small MLP).
        </li>
        <li>
          <strong>Final norm.</strong> One last LayerNorm to clean up the representation.
        </li>
        <li>
          <strong>Unembed.</strong> A linear layer projects each token's vector to a score for every
          word in the vocabulary.
        </li>
        <li>
          <strong>Softmax.</strong> Turn those scores into a probability distribution over the next
          token.
        </li>
      </ol>
      <p>
        Play with the diagram below. Drag the slider to add or remove layers and watch the stack
        grow. Click any component to see what it does and — crucially — the shape of the tensor
        flowing through it. Then step the cyan <span className="text-accent-cyan">signal</span> from
        the token chips at the bottom all the way up to the logits, to feel how information travels
        in one direction: forward, and up.
      </p>

      <Figure
        n="7.1"
        title="The decoder-only transformer, end to end"
        caption="One vertical pass: token IDs enter at the bottom, flow up through N identical blocks, and exit at the top as logits. Notice the dashed residual stream running straight through the block stack — every block reads from it and writes back into it. Add layers and the stack literally grows taller."
      >
        <TransformerStack />
      </Figure>

      <p>
        That is the entire architecture. Everything else in Part III is a magnifying glass held over
        one of those boxes. Let us walk the path the way a tensor does — from the bottom up.
      </p>

      <h2>The input: token and positional embeddings</h2>
      <p>
        Our input is a batch of tokenized sequences: a grid of integers of shape{" "}
        <M>{"(B, T)"}</M>, where <M>{"B"}</M> is the batch size (how many sequences we process at
        once) and <M>{"T"}</M> is the sequence length in tokens. Each integer is just an index into
        the vocabulary — it carries identity, not meaning. The first job is to give every token a
        vector the network can compute with.
      </p>
      <p>
        Two embeddings do this, and they are simply <em>added</em> together:
      </p>
      <ul>
        <li>
          <strong>Token embedding.</strong> A lookup table of shape <M>{"(V, d)"}</M> — one row per
          vocabulary entry, each row a learned <M>{"d"}</M>-dimensional vector. Indexing it with our{" "}
          <M>{"(B, T)"}</M> IDs produces a tensor of shape <M>{"(B, T, d)"}</M>. This is where the
          model's knowledge of <em>what each token means</em> lives, and we built it in{" "}
          <a href="/chapter/embeddings" className="prose-link">
            Chapter 6
          </a>
          .
        </li>
        <li>
          <strong>Positional embedding.</strong> Attention, as we will see, has no built-in sense of
          order — to it, a sentence is a <em>bag</em> of tokens. So we add a second vector that
          depends only on the <em>position</em> <M>{"0, 1, 2, \\dots, T-1"}</M>, telling each token
          where it sits. The mechanics (learned tables, sinusoids, or rotary embeddings) are the
          subject of{" "}
          <a href="/chapter/positional-encoding" className="prose-link">
            Chapter 10
          </a>
          .
        </li>
      </ul>
      <p>The combined input to the first block is therefore</p>
      <MB>{String.raw`x_0 = \underbrace{\text{TokEmb}(\text{ids})}_{(B,\,T,\,d)} \;+\; \underbrace{\text{PosEmb}(0{:}T)}_{(T,\,d)\ \to\ \text{broadcast}} \quad\in\quad \mathbb{R}^{B \times T \times d}.`}</MB>
      <p>
        From this point until the very end, the tensor keeps the shape <M>{"(B, T, d)"}</M>. That
        invariance is not an accident — it is the secret that lets us stack the same block as many
        times as we like.
      </p>

      <Callout type="pitfall" title="Adding, not concatenating">
        <p>
          Position is <em>added</em> to the token vector, not glued on as extra dimensions. Both
          live in the same <M>{"d"}</M>-dimensional space, and the model learns to disentangle
          "which word" from "which slot" using different directions of that space. If you
          concatenated instead, every downstream weight matrix would have to grow — and you would
          have invented a strictly worse design.
        </p>
      </Callout>

      <h2>Inside one block</h2>
      <p>
        Here is the heart of the architecture, and the part worth memorizing. A transformer block
        takes <M>{"x"}</M> of shape <M>{"(B, T, d)"}</M> and returns a tensor of the{" "}
        <em>exact same shape</em>, having refined it. Modern transformers use the{" "}
        <strong>pre-norm</strong> arrangement, where normalization happens <em>before</em> each
        sub-layer rather than after. One block is precisely two steps:
      </p>
      <MB>{String.raw`\begin{aligned} x &\;\leftarrow\; x + \text{MultiHeadAttention}\big(\text{LayerNorm}(x)\big) \\[0.4em] x &\;\leftarrow\; x + \text{FeedForward}\big(\text{LayerNorm}(x)\big) \end{aligned}`}</MB>
      <p>Read in order, the data flows like this:</p>
      <ol>
        <li>
          <strong>LayerNorm.</strong> Normalize each token's vector to zero mean and unit variance
          (then rescale with learned parameters). This keeps the numbers in a sane range so the deep
          stack stays trainable. Shape unchanged: <M>{"(B, T, d)"}</M>.
        </li>
        <li>
          <strong>Multi-Head Attention.</strong> The communication step. Every token looks at the
          other tokens (only earlier ones — see masking below) and pulls in relevant information.
          This is the <em>only</em> place tokens talk to each other. Shape unchanged.
        </li>
        <li>
          <strong>Residual add (<M>{"\\oplus"}</M>).</strong> Add the attention output back onto the
          input that entered the block. Nothing is overwritten — the block <em>contributes</em> to
          the running representation.
        </li>
        <li>
          <strong>LayerNorm</strong> again, on the updated vector.
        </li>
        <li>
          <strong>Feed-Forward Network (FFN).</strong> A small two-layer MLP applied to each token{" "}
          <em>independently</em> — no mixing across positions. This is where most of the per-token
          "thinking" and stored knowledge lives. Shape unchanged.
        </li>
        <li>
          <strong>Residual add</strong> once more.
        </li>
      </ol>
      <p>
        Attention <em>moves information between tokens</em>; the FFN <em>processes each token on its
        own</em>. Alternating these two — mix, then think, mix, then think — is the entire recipe.
        The internals of attention are{" "}
        <a href="/chapter/self-attention" className="prose-link">
          Chapter 8
        </a>{" "}
        and{" "}
        <a href="/chapter/multi-head-attention" className="prose-link">
          Chapter 9
        </a>
        ; the FFN, LayerNorm, and why the residual matters are{" "}
        <a href="/chapter/transformer-block" className="prose-link">
          Chapter 11
        </a>
        .
      </p>

      <Callout type="history" title="Pre-norm vs. post-norm">
        <p>
          The original 2017 "Attention Is All You Need" transformer put LayerNorm <em>after</em>{" "}
          each sub-layer (<em>post-norm</em>). It worked, but deep post-norm stacks are finicky to
          train and need careful learning-rate warmup. Around 2019 practitioners moved the norm{" "}
          <em>inside</em> the residual branch (<em>pre-norm</em>), which keeps a clean,
          un-normalized path from input to output and makes very deep models train stably. Essentially
          every modern LLM — GPT-2 onward, LLaMA, and friends — is pre-norm. That is the version we
          teach and build.
        </p>
      </Callout>

      <p>
        One detail rides along inside the attention step: <strong>causal masking</strong>. Because
        we are training the model to predict the <em>next</em> token, position <M>{"t"}</M> is only
        ever allowed to attend to positions <M>{"\\le t"}</M>. Letting it peek at token{" "}
        <M>{"t+1"}</M> would leak the answer. This single constraint is what makes the architecture a{" "}
        <em>decoder</em>, and it is why we can train on every position of a sequence in parallel yet
        still generate left-to-right at inference. We will draw the triangular mask in{" "}
        <a href="/chapter/self-attention" className="prose-link">
          Chapter 8
        </a>
        .
      </p>

      <h2>The residual stream</h2>
      <p>
        Now the single most important idea for actually <em>understanding</em> transformers — more
        load-bearing than attention itself. Look again at the two update lines for a block. Each one
        has the form <M>{"x \\leftarrow x + (\\text{something})"}</M>. The block never replaces{" "}
        <M>{"x"}</M>; it only ever <em>adds</em> to it. That running vector — the thing every block
        reads from and writes back into — is called the <strong>residual stream</strong>.
      </p>
      <p>
        Picture a single highway of shape <M>{"(B, T, d)"}</M> running straight up the entire model,
        from the embeddings at the bottom to the final norm at the top (it is the dashed line in
        Figure 7.1). Each sub-layer is a little off-ramp: it <em>reads</em> the current state of the
        stream, computes an update, and <em>adds</em> that update back on. Information is never
        destroyed by an overwrite — it accumulates. Early layers might write "this token is a noun";
        a later layer reads that and adds "and it is the subject of the sentence."
      </p>
      <MB>{String.raw`x_{\ell} = x_{\ell-1} + \text{Attn}_\ell\big(\text{LN}(x_{\ell-1})\big) + \text{FFN}_\ell\big(\text{LN}(\cdot)\big), \qquad \text{so}\quad x_N = x_0 + \sum_{\ell=1}^{N} (\text{contributions of block } \ell).`}</MB>

      <Callout type="key" title="The residual stream is the backbone">
        <p>
          Every transformer block <strong>reads from and writes to a shared residual stream</strong>{" "}
          of shape <M>{"(B, T, d)"}</M>. Sub-layers never overwrite it — they only add their
          contribution. Information therefore <em>accumulates</em> as you go up: the final
          representation is the initial embedding plus the sum of every block's edits. This is also
          why gradients flow cleanly to the bottom (the <M>{"+x"}</M> gives every layer a direct
          path), which is what makes 100-layer models trainable at all.
        </p>
      </Callout>

      <p>
        Once you see the residual stream, a lot of the architecture stops looking like a pipeline of
        transformations and starts looking like what it is: many small modules collaborating by
        leaving messages on a shared scratchpad, each free to ignore the parts it does not care
        about.
      </p>

      <h2>Stacking N blocks</h2>
      <p>
        Because every block preserves the shape <M>{"(B, T, d)"}</M>, we can chain as many as we
        want — that is the slider in the figure above. A real model uses <M>{"N"}</M> ranging from a
        dozen blocks (GPT-2 small, <M>{"N = 12"}</M>) to roughly a hundred in the largest frontier
        models. Depth is one of the main dials of capability.
      </p>
      <p>
        What does depth <em>buy</em> you? Each block refines the representation a little. The famous
        example is the word <em>"bank"</em> in <em>"the river bank was muddy."</em> Right after the
        embedding, the vector for <em>"bank"</em> is the same in every sentence — it is a context-free
        average of "financial institution" and "edge of a river." But as the sequence climbs the
        stack, attention lets <em>"bank"</em> read the word <em>"river,"</em> and successive blocks
        sharpen its representation until, near the top, the vector unambiguously means{" "}
        <em>riverbank</em>. The token did not change; its <strong>contextual representation</strong>{" "}
        did, layer by layer.
      </p>

      <Callout type="industry" title="Width, depth, and heads come together">
        <p>
          Three numbers define the size of the stack: <M>{"d"}</M> (the model width / residual
          dimension), <M>{"N"}</M> (the number of blocks / depth), and <M>{"h"}</M> (the number of
          attention heads per block). Bigger models scale all three roughly together. GPT-2 small is{" "}
          <M>{"d=768,\\ N=12,\\ h=12"}</M>; GPT-3 175B is <M>{"d=12288,\\ N=96,\\ h=96"}</M>.
          Choosing these jointly to spend a compute budget well is exactly what{" "}
          <a href="/chapter/scaling-laws" className="prose-link">
            Chapter 17
          </a>{" "}
          is about.
        </p>
      </Callout>

      <h2>The output head: from vectors to logits</h2>
      <p>
        After the last block, the residual stream <M>{"x_N"}</M> is a context-rich vector for every
        position — still shape <M>{"(B, T, d)"}</M>. Three small steps turn it into next-token
        predictions:
      </p>
      <ol>
        <li>
          <strong>Final LayerNorm.</strong> One concluding normalization, so the output head sees a
          clean, unit-scale representation. Shape stays <M>{"(B, T, d)"}</M>.
        </li>
        <li>
          <strong>LM head (the "unembed").</strong> A single linear layer with weight matrix of
          shape <M>{"(d, V)"}</M> projects each <M>{"d"}</M>-vector to one score per vocabulary
          entry. The tensor becomes <M>{"(B, T, V)"}</M>. These raw scores are the{" "}
          <strong>logits</strong>.
        </li>
        <li>
          <strong>Softmax.</strong> Over the vocabulary dimension, softmax turns the logits into a
          probability distribution — for <em>every</em> position simultaneously.
        </li>
      </ol>
      <MB>{String.raw`P(\text{next token at position } t) = \text{softmax}\big(\,x_N\, W_{\text{unembed}}\,\big)_t \in \Delta^{V-1}, \qquad \text{logits shape } (B, T, V).`}</MB>
      <p>
        At <em>training</em> time we use all <M>{"T"}</M> positions at once: position <M>{"t"}</M>'s
        distribution is scored against the token that actually followed it, which is how a single
        sequence yields <M>{"T"}</M> prediction targets. At <em>generation</em> time we only care
        about the <em>last</em> position — we sample its distribution to get the next token, append
        it, and run the whole stack again. That sampling loop is{" "}
        <a href="/chapter/what-is-an-llm" className="prose-link">
          Chapter 1
        </a>
        ; the loss that turns logits into a learning signal is{" "}
        <a href="/chapter/training-objective" className="prose-link">
          Chapter 12
        </a>
        .
      </p>

      <Callout type="tip" title="Weight tying">
        <p>
          The unembed matrix <M>{"(d, V)"}</M> and the token-embedding matrix <M>{"(V, d)"}</M> are
          transposes of the same shape. Many models — including GPT-2 — <strong>tie</strong> them:
          one matrix is used to map IDs to vectors at the bottom and vectors to logits at the top.
          This saves a large chunk of parameters (<M>{"V d"}</M> of them) and tends to help, since
          "the direction that means token <M>{"k"}</M>" should be the same whether you are reading it
          in or scoring it out.
        </p>
      </Callout>

      <h2>Counting parameters</h2>
      <p>
        Where do a model's parameters actually live? Almost everywhere, in matrices whose size is
        governed by the width <M>{"d"}</M>. Ignoring biases and the tiny LayerNorm scales (each only{" "}
        <M>{"O(d)"}</M>), the budget breaks into two buckets.
      </p>
      <p>
        <strong>Embeddings.</strong> The token table is <M>{"V \\times d"}</M> parameters. The
        positional table (if learned) adds <M>{"T_{\\max} \\times d"}</M>, usually small next to{" "}
        <M>{"Vd"}</M>. With weight tying the output head reuses the token table for free.
      </p>
      <p>
        <strong>Per block.</strong> Inside one transformer block:
      </p>
      <ul>
        <li>
          <strong>Attention:</strong> four <M>{"d \\times d"}</M> matrices — the query, key, and
          value projections <M>{"W_Q, W_K, W_V"}</M> plus the output projection <M>{"W_O"}</M> —
          giving <M>{"4d^2"}</M>.
        </li>
        <li>
          <strong>Feed-forward:</strong> two matrices, <M>{"d \\to 4d"}</M> and <M>{"4d \\to d"}</M>{" "}
          (the standard <M>{"4\\times"}</M> expansion), giving <M>{"4d^2 + 4d^2 = 8d^2"}</M>.
        </li>
      </ul>
      <p>So each block costs about</p>
      <MB>{String.raw`\underbrace{4d^2}_{\text{attention}} + \underbrace{8d^2}_{\text{FFN}} = 12\,d^2 \quad\text{parameters},`}</MB>
      <p>
        and an <M>{"N"}</M>-block stack is therefore roughly <M>{"12\\,N\\,d^2"}</M> parameters,{" "}
        <em>independent of vocabulary and sequence length</em>. Add the embeddings and you have the
        whole model:
      </p>
      <MB>{String.raw`\#\text{params} \;\approx\; \underbrace{V d}_{\text{embeddings}} \;+\; \underbrace{12\,N\,d^2}_{\text{the blocks}}.`}</MB>

      <Callout type="math" title="Worked example: GPT-2 small (124M)">
        <p>
          Plug in the real GPT-2 small config — <M>{"d = 768"}</M>, <M>{"N = 12"}</M>,{" "}
          <M>{"V = 50257"}</M>, <M>{"T_{\\max} = 1024"}</M>:
        </p>
        <ul>
          <li>
            Blocks: <M>{"12 \\times 12 \\times 768^2 = 84{,}934{,}656 \\approx 84.9\\text{M}"}</M>.
          </li>
          <li>
            Token embedding: <M>{"50257 \\times 768 = 38{,}597{,}376 \\approx 38.6\\text{M}"}</M>{" "}
            (and the tied head reuses it for free).
          </li>
          <li>
            Positional embedding: <M>{"1024 \\times 768 = 786{,}432 \\approx 0.8\\text{M}"}</M>.
          </li>
        </ul>
        <p>
          Total: <M>{"\\approx 124.4"}</M> million parameters — exactly the well-known 124M figure
          for GPT-2 small. Notice the two buckets are the same order of magnitude here, but as{" "}
          <M>{"d"}</M> grows the <M>{"d^2"}</M> block term dominates: at GPT-3 scale the blocks are
          well over 99% of the parameters.
        </p>
      </Callout>

      <h2>Encoder-decoder vs. decoder-only</h2>
      <p>
        The 2017 transformer was an <strong>encoder-decoder</strong>: an encoder read the whole input
        (e.g., a French sentence) with <em>bidirectional</em> attention, and a separate decoder
        generated the output (the English translation) while cross-attending to the encoder's
        representation. That split is natural for translation. But for a general language model the
        "input" and "output" are the same stream of text, and we just want to keep predicting the
        next token — so modern LLMs drop the encoder entirely and keep a single{" "}
        <strong>decoder-only</strong> stack with causal masking. It is simpler, every position is a
        training target, the same weights serve prompt and continuation, and it scales beautifully.
        GPT, LLaMA, and essentially every chat model you have used are decoder-only — which is why it
        is the architecture this book builds.
      </p>

      <h2>The complete forward pass in code</h2>
      <p>
        Everything above, in runnable PyTorch. This is the full skeleton — the <code>Block</code>{" "}
        class with its two pre-norm sub-layers and residual adds, and the <code>GPT</code> module
        that embeds, stacks, norms, and unembeds. The attention and MLP <em>internals</em> are
        stubbed to their public shape here; we fill them in across the next four chapters and assemble
        the production version in{" "}
        <a href="/chapter/building-the-model" className="prose-link">
          Chapter 14
        </a>
        .
      </p>

      <CodeBlock language="python" filename="gpt.py" highlight={[27, 28, 29, 30, 43, 44, 45, 46]}>
{`import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One pre-norm transformer block. Shape in == shape out: (B, T, d)."""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)   # Ch. 8 & 9
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(                        # Ch. 11
            nn.Linear(d_model, 4 * d_model),              # expand  d -> 4d
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),              # project 4d -> d
        )

    def forward(self, x):
        # Pre-norm + residual: read the stream, add a contribution back.
        x = x + self.attn(self.ln1(x))   # communication: tokens mix
        x = x + self.ffn(self.ln2(x))    # computation:  per-token MLP
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, d_model, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # (V, d) token table
        self.pos_emb = nn.Embedding(block_size, d_model)  # (T, d) position table
        self.blocks  = nn.ModuleList(
            [Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)                 # final norm
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # unembed (d, V)
        self.head.weight = self.tok_emb.weight            # weight tying

    def forward(self, idx):                # idx: (B, T) integer token IDs
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)          # 0,1,...,T-1
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B, T, d)  the residual stream
        for block in self.blocks:                          # climb the stack
            x = block(x)                                   # (B, T, d) -> (B, T, d)
        x = self.ln_f(x)                                   # (B, T, d)
        logits = self.head(x)                              # (B, T, V)
        return logits                                      # softmax + loss handled by caller`}
      </CodeBlock>

      <p>
        Read the two highlighted regions side by side: the four <em>shape-preserving</em> lines of{" "}
        <code>GPT.forward</code> (embed, loop, norm, unembed) and the two <em>residual</em> lines of{" "}
        <code>Block.forward</code>. That is the transformer. Forty lines, most of them bookkeeping.
      </p>

      <h2>Counting parameters in code</h2>
      <p>
        And the arithmetic from the previous section as a one-liner you can drop into any model. With
        weight tying the head shares the embedding tensor, so <code>parameters()</code> already
        avoids double-counting it.
      </p>

      <CodeBlock language="python" filename="count.py">
{`def count_parameters(model, trainable_only=True):
    """Total number of parameters, in millions."""
    params = (p for p in model.parameters() if p.requires_grad or not trainable_only)
    total = sum(p.numel() for p in params)
    return total

model = GPT(vocab_size=50257, n_layer=12, n_head=12,
            d_model=768, block_size=1024)
n = count_parameters(model)
print(f"{n:,} parameters  ({n/1e6:.1f}M)")
# ->  124,439,808 parameters  (124.4M)   # GPT-2 small`}
      </CodeBlock>

      <Callout type="warning" title="Parameters are not the runtime cost">
        <p>
          Parameter count tells you how much the model <em>weighs</em> in memory, but not how
          expensive it is to <em>run</em>. The compute and memory of a forward pass also scale with
          the sequence length <M>{"T"}</M> — and attention in particular costs <M>{"O(T^2)"}</M>,
          because every token compares against every other. Two models with identical parameter
          counts can have wildly different inference costs depending on context length. We untangle
          this in{" "}
          <a href="/chapter/inference-optimization" className="prose-link">
            Chapter 22
          </a>
          .
        </p>
      </Callout>

      <h2>Where we go next</h2>
      <p>
        You now own the map: token IDs in at <M>{"(B, T)"}</M>, embeddings lift them to{" "}
        <M>{"(B, T, d)"}</M>, <M>{"N"}</M> pre-norm blocks read and write a shared residual stream
        while attention mixes tokens and the FFN thinks about each one, a final norm tidies up, and
        an unembedding projects to <M>{"(B, T, V)"}</M> logits that softmax turns into next-token
        probabilities. Every box has a name and a shape.
      </p>
      <p>
        Now we open the boxes. The very next chapter zooms all the way into the single component that
        makes the whole thing work — the one place tokens are allowed to talk to each other —{" "}
        <a href="/chapter/self-attention" className="prose-link">
          self-attention
        </a>
        . If you understand that one mechanism deeply, the rest of this architecture, as you have
        just seen, is bookkeeping.
      </p>
    </>
  );
}
