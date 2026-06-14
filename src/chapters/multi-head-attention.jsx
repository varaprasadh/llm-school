import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import AttentionExplorer from "../components/viz/AttentionExplorer";
import HeadSplit from "../components/viz/multi-head-attention/HeadSplit";

export default function Chapter() {
  return (
    <>
      <p>
        In the <a href="/chapter/self-attention" className="prose-link">last chapter</a> we built a
        single attention operation: every token forms a query, compares it against every key, and
        pulls a weighted blend of values. It’s powerful — but it has exactly <em>one</em> opinion per
        layer about what “relevant” means. Real language needs many opinions at once. The token{" "}
        <em>“it”</em> wants to track <em>which noun it refers to</em>, but the verb{" "}
        <em>“sat”</em> simultaneously wants to find <em>its subject</em>, and another part of the
        model just wants to know <em>what word came immediately before</em>. One attention pattern
        can’t do all three. <strong>Multi-head attention</strong> is the simple, beautiful fix: run
        several attention operations in parallel, each in its own slice of the representation, then
        glue the results together.
      </p>

      <h2>Why one head isn’t enough</h2>
      <p>
        A single attention layer produces, for each query token, <em>one</em> probability
        distribution over the other tokens — one row of the attention matrix. That distribution has
        to be a compromise between every relationship the model might care about. But the
        relationships are genuinely different and often conflicting:
      </p>
      <ul>
        <li>
          <strong>Coreference:</strong> <em>“it”</em> → <em>“cat”</em> (a long-range, content-based
          link).
        </li>
        <li>
          <strong>Syntax:</strong> a verb → its subject; an adjective → the noun it modifies.
        </li>
        <li>
          <strong>Position:</strong> “attend to the token immediately to my left,” regardless of
          content.
        </li>
      </ul>
      <p>
        You cannot encode all of these in a single softmax distribution without smearing them
        together into mush. The dot product <M>{"q \\cdot k"}</M> measures one notion of similarity
        in one geometry. To measure <em>several</em> notions, you need several geometries — several
        independent sets of <M>{"W_Q, W_K, W_V"}</M> projecting into <em>different</em> subspaces.
        That is exactly what a head is.
      </p>

      <Callout type="key" title="The whole idea">
        <p>
          A <strong>head</strong> is one complete attention computation with its own learned
          query/key/value projections. Multi-head attention runs <M>{"h"}</M> of them in parallel,
          each in a smaller subspace, so the model can attend to many different relationships
          simultaneously and then combine what they found.
        </p>
      </Callout>

      <p>
        The visualizer below is the same one from the last chapter — but now read each preset as a{" "}
        <em>different head</em>. “Previous token,” “First / BOS,” and “Coreference” are three
        distinct relationships a real layer learns in parallel. No single distribution could be all
        three at once; three heads can.
      </p>

      <Figure
        n="9.1"
        title="Each head = a different learned relationship"
        caption="Switch heads to see three genuinely different attention patterns over the same sentence. In a real layer these run simultaneously — a positional head, a coreference head, and more — and their outputs are concatenated."
      >
        <AttentionExplorer />
      </Figure>

      <h2>Splitting into heads</h2>
      <p>
        Here’s the elegant part: we don’t make the model <em>wider</em> to add heads. We take the
        existing <M>{"d_{\\text{model}}"}</M>-dimensional representation and <strong>partition</strong>{" "}
        it into <M>{"h"}</M> equal pieces. Each head operates on a subspace of size
      </p>
      <MB>{String.raw`d_k = \frac{d_{\text{model}}}{h}.`}</MB>
      <p>
        So with <M>{"d_{\\text{model}} = 512"}</M> and <M>{"h = 8"}</M> heads, each head works in a
        cozy 64-dimensional subspace. Concretely, we project the input with full-width matrices{" "}
        <M>{"W_Q, W_K, W_V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}"}</M>, then{" "}
        <em>reshape</em> the result so the last dimension is split into{" "}
        <M>{"(h, d_k)"}</M>. In tensor-shape terms, for a batch of <M>{"B"}</M> sequences of length{" "}
        <M>{"T"}</M>:
      </p>
      <MB>{String.raw`(B,\, T,\, d_{\text{model}}) \;\xrightarrow{\text{reshape}}\; (B,\, T,\, h,\, d_k) \;\xrightarrow{\text{transpose}}\; (B,\, h,\, T,\, d_k).`}</MB>
      <p>
        That final transpose moves the head axis next to the batch axis, so PyTorch treats{" "}
        <M>{"(B, h)"}</M> as one big batch and runs all heads at once. Drag the slider in the diagram
        below to watch a 12-dimensional token vector get sliced into <M>{"h"}</M> colored heads, each
        of size <M>{"d_k = 12/h"}</M>, routed to its own little attention box, and then reassembled.
      </p>

      <Figure
        n="9.2"
        title="The anatomy of multi-head attention"
        caption="A d_model = 12 token vector is split into h colored heads (d_k = 12/h each), each head runs its own attention in its own subspace, the outputs are concatenated, and W_O mixes them back together. Try h = 1 (plain attention) through h = 12 (one dimension per head)."
      >
        <HeadSplit />
      </Figure>

      <Callout type="pitfall" title="Heads partition the width — they don’t add to it">
        <p>
          A common misconception is that more heads make the model bigger. They don’t: <M>{"h"}</M>{" "}
          and <M>{"d_k"}</M> trade off so that <M>{"h \\cdot d_k = d_{\\text{model}}"}</M> stays
          fixed. Doubling the heads <em>halves</em> each head’s dimension. More heads means more
          diverse-but-smaller views; fewer heads means fewer-but-richer views. It’s a partition, not
          an extension.
        </p>
      </Callout>

      <h2>Attention per head, in parallel</h2>
      <p>
        Inside each head <M>{"i"}</M>, nothing new happens — it’s the exact scaled dot-product
        attention from the last chapter, just with the smaller dimension <M>{"d_k"}</M>:
      </p>
      <MB>{String.raw`\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right) V_i,`}</MB>
      <p>
        where <M>{"Q_i, K_i, V_i \\in \\mathbb{R}^{T \\times d_k}"}</M> are the slices of the
        projected query/key/value belonging to head <M>{"i"}</M>. Note the scaling factor is{" "}
        <M>{"\\sqrt{d_k}"}</M>, not <M>{"\\sqrt{d_{\\text{model}}}"}</M> — each head normalizes by{" "}
        <em>its own</em> dimension, which is exactly why the variance argument from the last chapter
        still holds per head. Because the heads share no parameters and touch disjoint slices of the
        data, they are embarrassingly parallel: one batched matrix multiply computes all{" "}
        <M>{"h"}</M> attention matrices at once on the GPU.
      </p>

      <Callout type="math" title="The same √dₖ trick, per head">
        <p>
          Each head’s scores are dot products of <M>{"d_k"}</M>-dimensional vectors, so their
          variance is <M>{"d_k"}</M> (not <M>{"d_{\\text{model}}"}</M>). Dividing by{" "}
          <M>{"\\sqrt{d_k}"}</M> renormalizes <em>each head’s</em> logits to unit variance, keeping
          every head’s softmax in its high-gradient regime no matter how you set <M>{"h"}</M>.
        </p>
      </Callout>

      <h2>Concatenate + output projection</h2>
      <p>
        After all <M>{"h"}</M> heads produce their <M>{"T \\times d_k"}</M> outputs, we{" "}
        <strong>concatenate</strong> them along the feature axis, recovering a{" "}
        <M>{"T \\times d_{\\text{model}}"}</M> matrix (since <M>{"h \\cdot d_k = d_{\\text{model}}"}</M>
        ). But a raw concatenation just stacks the heads side by side — head 3 still lives in columns
        96–127 and never talks to head 1. So we apply one final learned <strong>output projection</strong>{" "}
        <M>{"W_O \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}"}</M> that mixes the
        heads together into a single coherent representation:
      </p>
      <MB>{String.raw`\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O.`}</MB>
      <p>
        <M>{"W_O"}</M> is doing real work: it lets the model decide how to <em>combine</em> what the
        heads discovered — maybe weighting the coreference head heavily for one output dimension and
        the positional head for another. Without it, the heads would be permanently siloed. This is
        the last <span className="font-mono">W_O</span> box in the diagram above, taking the
        multicolored concatenated strip and blending it into a uniform output vector.
      </p>

      <Callout type="industry" title="Four matrices, often fused">
        <p>
          A multi-head attention block has four weight matrices: <M>{"W_Q, W_K, W_V, W_O"}</M>. In
          practice <M>{"W_Q, W_K, W_V"}</M> are frequently fused into a single{" "}
          <M>{"d_{\\text{model}} \\times 3d_{\\text{model}}"}</M> matrix for one big efficient GEMM,
          then split. Variants like <em>multi-query</em> and <em>grouped-query</em> attention (GQA)
          go further and <em>share</em> keys/values across heads to shrink the KV cache at inference
          time — a trick we’ll meet again when we optimize serving.
        </p>
      </Callout>

      <h2>What heads learn</h2>
      <p>
        Because each head is free to specialize, interpretability researchers have found that
        individual heads in trained transformers often take on recognizable, nameable jobs. A few
        well-documented examples — kept grounded, because heads are messy and rarely <em>purely</em>{" "}
        one thing:
      </p>
      <ul>
        <li>
          <strong>Positional / previous-token heads:</strong> some heads attend almost entirely to
          the token at a fixed relative offset (e.g. “the previous token”), acting like a learned
          shift register.
        </li>
        <li>
          <strong>Induction heads:</strong> a celebrated, well-studied pattern. After seeing{" "}
          <code>… A B … A</code>, an induction head makes the second <code>A</code> attend to the
          token that followed the first <code>A</code> — i.e. it predicts <code>B</code>. This is the
          mechanism behind in-context, copy-the-pattern learning, and it emerges from a pair of heads
          across two layers working together.
        </li>
        <li>
          <strong>Syntactic heads:</strong> heads that approximately track grammatical relations —
          a verb attending to its subject, a determiner to its noun, a preposition to its object.
        </li>
        <li>
          <strong>Rare / “duplicate token” and name-mover heads:</strong> documented in detailed
          circuit analyses of GPT-2, where specific heads were shown to route names between clauses.
        </li>
      </ul>
      <p>
        The honest caveat: these are tendencies, not guarantees. Heads can be polysemantic, jobs can
        be split across several heads, and many heads do something hard to name. But the high-level
        story holds — giving the model multiple independent attention subspaces lets it discover and
        dedicate machinery to distinct linguistic relationships.
      </p>

      <Callout type="history" title="Where this comes from">
        <p>
          Multi-head attention was introduced in <em>“Attention Is All You Need”</em> (Vaswani et
          al., 2017), the paper that introduced the transformer. The original base model used{" "}
          <M>{"d_{\\text{model}} = 512"}</M> with <M>{"h = 8"}</M> heads of size{" "}
          <M>{"d_k = 64"}</M>. The authors explicitly motivated heads as letting the model “jointly
          attend to information from different representation subspaces.”
        </p>
      </Callout>

      <h2>Cost is unchanged</h2>
      <p>
        Here’s the result that makes multi-head attention almost free. You might fear that running{" "}
        <M>{"h"}</M> attention operations costs <M>{"h\\times"}</M> as much. It doesn’t. The total
        work is essentially identical to a single attention over the full <M>{"d_{\\text{model}}"}</M>.
        The reason: each head is <M>{"h"}</M> times <em>smaller</em>, so the savings cancel the
        multiplication exactly.
      </p>
      <p>
        Count the score-and-blend FLOPs. A single big head over dimension <M>{"d_{\\text{model}}"}</M>{" "}
        computes a <M>{"T \\times T"}</M> score matrix where each entry is a dot product of length{" "}
        <M>{"d_{\\text{model}}"}</M>: that’s <M>{"\\propto T^2\\, d_{\\text{model}}"}</M>. With{" "}
        <M>{"h"}</M> heads, each head computes a <M>{"T \\times T"}</M> matrix with dot products of
        length <M>{"d_k = d_{\\text{model}}/h"}</M>, and there are <M>{"h"}</M> of them:
      </p>
      <MB>{String.raw`h \times \underbrace{T^2 d_k}_{\text{per head}} = h \cdot T^2 \cdot \frac{d_{\text{model}}}{h} = T^2\, d_{\text{model}}.`}</MB>
      <p>
        The <M>{"h"}</M> cancels. Same for the value aggregation. The only genuinely new parameters
        and FLOPs come from the output projection <M>{"W_O"}</M>, which is a single{" "}
        <M>{"d_{\\text{model}} \\times d_{\\text{model}}"}</M> matrix — a small, fixed addition
        independent of <M>{"h"}</M>. So you get the expressive power of many attention patterns at
        the price of (essentially) one.
      </p>

      <Callout type="tip" title="Free expressiveness">
        <p>
          Multi-head attention is one of those rare wins where you gain a lot (parallel relational
          views) and pay almost nothing (the same matmul cost, plus one extra projection). That’s
          why <em>every</em> production transformer uses many heads — GPT-2 small uses 12, GPT-3
          uses 96.
        </p>
      </Callout>

      <h2>Implementing it from scratch</h2>
      <p>
        Below is a complete, correct multi-head attention module in PyTorch. It does the full
        pipeline: project to Q/K/V, reshape to <M>{"(B, h, T, d_k)"}</M>, run batched scaled
        dot-product attention with a causal mask, transpose and reshape back to{" "}
        <M>{"(B, T, d_{\\text{model}})"}</M>, and apply the output projection. This is, modulo
        engineering details, the attention block inside a real GPT.
      </p>

      <CodeBlock language="python" filename="multihead_attention.py" highlight={[24, 25, 38]}>
{`import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads          # subspace size per head
        self.causal = causal

        # One fused projection produces Q, K, V (3 * d_model) in a single matmul.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)   # W_O

    def forward(self, x):
        B, T, C = x.shape                      # (batch, seq_len, d_model)

        # Project, then split into Q, K, V — each (B, T, d_model).
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        # Reshape (B, T, d_model) -> (B, n_heads, T, d_k) so heads are batched.
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores for every head at once: (B, n_heads, T, T).
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)       # rows sum to 1, per head
        out = attn @ v                         # (B, n_heads, T, d_k)

        # Concatenate heads: (B, n_heads, T, d_k) -> (B, T, d_model).
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)                  # apply W_O, return (B, T, d_model)


# --- sanity check ---
if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(2, 16, 512)                # (B=2, T=16, d_model=512)
    y = mha(x)
    print(y.shape)                             # torch.Size([2, 16, 512])`}
      </CodeBlock>

      <p>
        Read the two highlighted reshape lines slowly — they are the entire trick. The expensive part
        (<code>q @ k.transpose</code>, the softmax, <code>attn @ v</code>) is literally the
        single-head code from last chapter, now with an extra <M>{"h"}</M> axis that the GPU
        parallelizes over. In production you’d swap the manual score/softmax for{" "}
        <code>F.scaled_dot_product_attention</code> (FlashAttention under the hood), but the shapes
        and the meaning are exactly these.
      </p>

      <Callout type="pitfall" title="Don’t forget the .contiguous()">
        <p>
          After <code>transpose(1, 2)</code> the tensor is non-contiguous in memory, and{" "}
          <code>.view()</code> will throw. The <code>.contiguous()</code> before the final{" "}
          <code>.view(B, T, C)</code> physically reorders the data so the concatenation is correct.
          A subtle bug here silently scrambles which dimensions belong to which head.
        </p>
      </Callout>

      <h2>What just happened, and where we go next</h2>
      <p>
        We turned one attention pattern into <M>{"h"}</M> of them by partitioning the model’s width
        into independent subspaces, running scaled dot-product attention in each, concatenating, and
        mixing the results with <M>{"W_O"}</M> — all for essentially the same cost as a single head.
        The model can now track coreference, syntax, and position <em>simultaneously</em>, and we saw
        that trained heads really do specialize into recognizable jobs.
      </p>
      <p>
        But attention — single- or multi-head — has a quiet, glaring blind spot. It treats its input
        as an unordered <em>set</em>: shuffle the tokens and the outputs shuffle identically, with no
        sense that “the cat” is different from “cat the.” Word order is the bedrock of language, and
        so far our model is completely blind to it. In the{" "}
        <a href="/chapter/positional-encoding" className="prose-link">next chapter</a> we fix
        that — injecting position into the model with sinusoidal encodings, learned embeddings, and
        the rotary embeddings (RoPE) that power today’s frontier models.
      </p>
    </>
  );
}
