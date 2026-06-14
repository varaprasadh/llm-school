import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import AttentionExplorer from "../components/viz/AttentionExplorer";

export default function Chapter() {
  return (
    <>
      <p>
        Self-attention is the beating heart of the transformer — the mechanism that lets every
        token <em>look at</em> every other token and decide what’s relevant. If you understand this
        one chapter deeply, the rest of the architecture is bookkeeping. We’ll build it from the
        ground up: the intuition, the exact formula, a live visualizer, and a from-scratch
        implementation.
      </p>

      <h2>The problem attention solves</h2>
      <p>
        Consider the sentence <em>“The cat sat on the mat because it was tired.”</em> To represent
        the word <em>“it”</em> usefully, the model must figure out what “it” refers to — the cat,
        not the mat. That information lives several tokens away. Earlier architectures (RNNs) passed
        information along a chain, step by step, and struggled to connect distant words. Attention
        throws out the chain: it lets <em>“it”</em> reach back and pull information directly from{" "}
        <em>“cat”</em>, no matter the distance, in a single step.
      </p>

      <Callout type="key" title="One-line intuition">
        <p>
          Self-attention computes, for every token, a <em>weighted average of all the other tokens’
          information</em>, where the weights are learned and depend on how relevant each token is to
          the current one.
        </p>
      </Callout>

      <h2>Queries, keys, and values</h2>
      <p>
        The cleanest mental model is a soft dictionary lookup. Every token produces three vectors by
        multiplying its embedding <M>{"x"}</M> by three learned weight matrices:
      </p>
      <ul>
        <li>
          <strong>Query</strong> <M>{"q = x W_Q"}</M> — “what am I looking for?”
        </li>
        <li>
          <strong>Key</strong> <M>{"k = x W_K"}</M> — “what do I contain / advertise?”
        </li>
        <li>
          <strong>Value</strong> <M>{"v = x W_V"}</M> — “what information will I hand over if
          attended to?”
        </li>
      </ul>
      <p>
        To decide how much token <M>{"i"}</M> should attend to token <M>{"j"}</M>, we compare token{" "}
        <M>{"i"}</M>’s query with token <M>{"j"}</M>’s key using a dot product. A large dot product
        means “these match — pay attention.” We then take a weighted sum of the <em>values</em>,
        using those match scores (after a softmax) as weights.
      </p>

      <Callout type="industry" title="The database analogy">
        <p>
          In a normal hash map you look up a key and get exactly one value. Attention is a{" "}
          <em>soft</em> version: your query is compared against <em>every</em> key, and you get a
          blended mixture of <em>all</em> the values, weighted by how well each key matched. It’s
          differentiable, so the matrices <M>{"W_Q, W_K, W_V"}</M> can be learned by gradient
          descent.
        </p>
      </Callout>

      <h2>Scaled dot-product attention</h2>
      <p>
        Stacking the per-token queries, keys and values into matrices{" "}
        <M>{"Q, K, V \\in \\mathbb{R}^{n \\times d_k}"}</M> (one row per token), the entire operation
        is a single, famous formula:
      </p>
      <MB>{String.raw`\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V`}</MB>
      <p>Let’s dissect it left to right, because every piece earns its place:</p>
      <ol>
        <li>
          <M>{"Q K^{\\top}"}</M> — an <M>{"n \\times n"}</M> matrix of <em>scores</em>. Entry{" "}
          <M>{"(i, j)"}</M> is the dot product of query <M>{"i"}</M> with key <M>{"j"}</M>: how much
          token <M>{"i"}</M> finds token <M>{"j"}</M> relevant.
        </li>
        <li>
          <M>{"\\div \\sqrt{d_k}"}</M> — the <em>scaling</em>. Without it, dot products grow with
          dimension <M>{"d_k"}</M>, pushing the softmax into a regime where one entry saturates to 1
          and gradients vanish. Dividing by <M>{"\\sqrt{d_k}"}</M> keeps the variance ~1.
        </li>
        <li>
          <M>{"\\text{softmax}(\\cdot)"}</M> — turns each row of scores into a probability
          distribution that sums to 1: the <strong>attention weights</strong>.
        </li>
        <li>
          <M>{"\\times V"}</M> — uses those weights to take a weighted average of the value vectors.
          The result, for each token, is a blend of information gathered from the tokens it attended
          to.
        </li>
      </ol>

      <Callout type="math" title="Why √dₖ exactly?">
        <p>
          If the components of <M>{"q"}</M> and <M>{"k"}</M> are independent with mean 0 and variance
          1, their dot product <M>{"q\\cdot k = \\sum_{i=1}^{d_k} q_i k_i"}</M> has variance{" "}
          <M>{"d_k"}</M> and standard deviation <M>{"\\sqrt{d_k}"}</M>. Dividing by{" "}
          <M>{"\\sqrt{d_k}"}</M> renormalizes the scores back to unit variance, regardless of head
          size — keeping the softmax in its sensitive, high-gradient range.
        </p>
      </Callout>

      <h2>See it work</h2>
      <p>
        Enough symbols. Below is a live attention map over our example sentence. Pick a{" "}
        <strong>head</strong> (each head learns a different relationship), click any token to make it
        the <span className="text-brand-300">query</span>, and watch the weights form. Notice how
        the <em>Coreference</em> head makes <em>“it”</em> attend strongly to <em>“cat”</em> — the
        model has learned what “it” points to.
      </p>

      <Figure
        n="8.1"
        title="Self-attention, live"
        caption="Each row of the matrix is one query token's probability distribution over keys. Real models learn dozens of such patterns per layer. Toggle the causal mask to see how a decoder is forbidden from attending to future tokens."
      >
        <AttentionExplorer />
      </Figure>

      <h2>Causal masking: don’t peek at the future</h2>
      <p>
        We are building a model that <em>generates</em> text left to right. When predicting token{" "}
        <M>{"t"}</M>, it must only use tokens <M>{"1 \\ldots t"}</M> — letting it see token{" "}
        <M>{"t+1"}</M> would be cheating, and would make generation impossible (that token doesn’t
        exist yet at inference time). We enforce this with a <strong>causal mask</strong>: before the
        softmax, set every score where <M>{"j > i"}</M> to <M>{"-\\infty"}</M>, so those positions
        receive exactly zero weight.
      </p>
      <MB>{String.raw`\text{scores}_{ij} \;=\; \begin{cases} \dfrac{q_i \cdot k_j}{\sqrt{d_k}} & j \le i \\[1.2em] -\infty & j > i \end{cases}`}</MB>
      <p>
        Toggle the mask in the visualizer above: with it on, the attention matrix becomes lower-
        triangular — every token can attend to itself and the past, never the future.
      </p>

      <h2>Implementing it from scratch</h2>
      <p>
        Here is the complete operation in PyTorch. This exact function (just vectorized over batches
        and heads) is what runs inside every LLM you’ve used.
      </p>

      <CodeBlock language="python" filename="attention.py">
{`import torch
import torch.nn.functional as F

def self_attention(x, W_q, W_k, W_v, causal=True):
    # x: (T, d_model)  — T tokens, each a d_model vector
    T, d_model = x.shape
    Q = x @ W_q                 # (T, d_k)  queries
    K = x @ W_k                 # (T, d_k)  keys
    V = x @ W_v                 # (T, d_v)  values
    d_k = Q.shape[-1]

    # 1) raw compatibility scores between every pair of tokens
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5     # (T, T)

    # 2) causal mask: forbid attending to future positions
    if causal:
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    # 3) softmax over the key dimension -> attention weights
    attn = F.softmax(scores, dim=-1)                  # (T, T), rows sum to 1

    # 4) weighted sum of values
    out = attn @ V                                    # (T, d_v)
    return out, attn`}
      </CodeBlock>

      <p>
        Four lines of real work. The genius of the transformer is realizing that this — repeated and
        stacked — is enough to model language.
      </p>

      <Callout type="pitfall" title="Quadratic cost">
        <p>
          The score matrix <M>{"Q K^{\\top}"}</M> is <M>{"n \\times n"}</M>: attention is{" "}
          <M>{"O(n^2)"}</M> in sequence length, in both time and memory. Doubling the context
          quadruples the cost. This single fact drives an entire research field — FlashAttention,
          sparse attention, linear attention — which we revisit in{" "}
          <a href="/chapter/inference-optimization" className="prose-link">
            Chapter 22
          </a>
          .
        </p>
      </Callout>

      <h2>What just happened, in one paragraph</h2>
      <p>
        Each token asked a question (its query), compared it against what every other token
        advertised (their keys), and used the match scores to pull a weighted blend of their
        information (values) into itself. No recurrence, no convolution — just three matrix
        multiplies and a softmax, fully parallel across all positions. Next we’ll see why one
        attention pattern isn’t enough, and how <strong>multiple heads</strong> let the model track
        many relationships at once.
      </p>
    </>
  );
}
