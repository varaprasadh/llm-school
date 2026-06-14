import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import BlockAnatomy from "../components/viz/transformer-block/BlockAnatomy";

export default function Chapter() {
  return (
    <>
      <p>
        The previous chapters gave attention all the glory — and rightly so, it is the one place
        tokens are allowed to talk to each other. But a transformer block has a second half that is
        just as essential and holds <em>more</em> of the parameters: the feed-forward network, the
        residual connections that wrap it, and the normalization that keeps the whole deep stack
        trainable. This is the <strong>non-attention</strong> machinery. If{" "}
        <a href="/chapter/self-attention" className="prose-link">
          attention (Ch. 8)
        </a>{" "}
        is how the block <em>gathers</em> information, this chapter is how it <em>thinks</em> about
        what it gathered — and how the gradients survive the trip down a hundred layers.
      </p>
      <p>
        We pick up exactly where{" "}
        <a href="/chapter/transformer-overview" className="prose-link">
          the overview (Ch. 7)
        </a>{" "}
        left off. There you saw the two-line block; here we open the second line, fill in every
        piece, and finally assemble a complete, runnable block that you could drop into a real
        model.
      </p>

      <h2>Attention mixes, the FFN thinks</h2>
      <p>
        A transformer block does two different jobs, and it is worth being precise about the
        division of labor because beginners routinely conflate them. <strong>Attention</strong>{" "}
        moves information <em>across</em> tokens: the representation of <em>"it"</em> can pull in
        facts stored at <em>"cat"</em> several positions away. It is fundamentally a{" "}
        <em>communication</em> operation, and every output position depends on many input positions.
      </p>
      <p>
        The <strong>feed-forward network</strong> (FFN, also called the MLP or the position-wise
        network) does the opposite. It processes <em>each token independently</em> — the vector at
        position 5 is transformed without ever looking at positions 4 or 6. Run the exact same small
        neural network on every position in parallel, sharing one set of weights. No mixing, no
        communication: pure per-token computation.
      </p>

      <Callout type="key" title="Mix, then think — repeated">
        <p>
          A transformer block alternates two complementary operations.{" "}
          <strong>Attention</strong> mixes information <em>between</em> tokens (every position reads
          from many others); the <strong>FFN</strong> transforms <em>each</em> token on its own (no
          cross-token interaction at all). Stacking <em>mix → think → mix → think</em> is the entire
          recipe. Attention decides <em>what to combine</em>; the FFN decides{" "}
          <em>what to make of it</em>.
        </p>
      </Callout>

      <p>
        Why split the work this way? Because the two jobs have genuinely different shapes. Mixing
        across a length-<M>{"T"}</M> sequence is inherently a relational operation —{" "}
        <M>{"O(T^2)"}</M> interactions — and attention is the cheap, learnable way to do it.
        Transforming a single vector into a richer one is a classic job for a multilayer perceptron,
        and it parallelizes trivially across positions. Interleaving them lets each layer first
        decide which other tokens are relevant, then chew on the gathered context. A growing body of
        interpretability work even suggests the FFN layers act as a kind of{" "}
        <em>key–value memory</em>: certain hidden units fire for specific input patterns and write
        specific facts back into the residual stream. The FFN is, loosely, where a lot of the
        model's <em>stored knowledge</em> lives.
      </p>

      <h2>The feed-forward network</h2>
      <p>
        Mechanically, the FFN is the simplest thing in the whole architecture: two linear layers
        with a nonlinearity between them. Take a token's <M>{"d"}</M>-dimensional vector, project it
        up to a larger hidden dimension <M>{"d_{\\text{ff}}"}</M>, apply a nonlinear activation, then
        project back down to <M>{"d"}</M>:
      </p>
      <MB label="11.1">{String.raw`\text{FFN}(x) = W_2\,\text{act}\!\big(W_1 x + b_1\big) + b_2,\qquad W_1 \in \mathbb{R}^{d_{\text{ff}}\times d},\ \ W_2 \in \mathbb{R}^{d\times d_{\text{ff}}}.`}</MB>
      <p>
        The hidden dimension is conventionally <strong>four times</strong> the model width,{" "}
        <M>{"d_{\\text{ff}} = 4d"}</M>. So the data flow inside the FFN is a little diamond:{" "}
        <M>{"d \\to 4d \\to d"}</M>. The middle is deliberately wide. That <M>{"4\\times"}</M>{" "}
        expansion gives the network room to compute a rich set of nonlinear features about the token
        before compressing the useful ones back down into the residual stream. (The factor 4 is a
        well-worn default from the original transformer, not a law of nature — some models tune it,
        and gated variants use a smaller multiplier to keep the parameter count fixed, as we will
        see.)
      </p>
      <p>
        Here is the part that surprises people: this humble two-layer MLP is where{" "}
        <strong>most of the block's parameters live</strong>. Count them. The two attention-style
        weight matrices in the FFN are <M>{"W_1"}</M> of size <M>{"4d \\times d"}</M> and{" "}
        <M>{"W_2"}</M> of size <M>{"d \\times 4d"}</M>, for a total of{" "}
        <M>{"4d^2 + 4d^2 = 8d^2"}</M> parameters. Multi-head attention, by contrast, uses four{" "}
        <M>{"d \\times d"}</M> matrices (<M>{"W_Q, W_K, W_V, W_O"}</M>) for <M>{"4d^2"}</M>. So in a
        standard block the FFN holds twice as many parameters as attention —{" "}
        <M>{"8d^2"}</M> of the <M>{"12d^2"}</M> total, almost exactly two-thirds.
      </p>

      <Callout type="industry" title="Two-thirds of the weights, per token">
        <p>
          For a typical pre-norm block, the parameter split is{" "}
          <M>{"4d^2"}</M> (attention) <M>{"+\\ 8d^2"}</M> (FFN) <M>{"=\\ 12d^2"}</M>. The
          feed-forward network is roughly <strong>two-thirds of every block's weights</strong>, and
          therefore two-thirds of the model. When people say large language models "store knowledge
          in their weights," a great deal of that storage is the FFN. It is also why techniques that
          shrink or sparsify the FFN — like mixture-of-experts, which swaps one big FFN for many
          smaller ones and routes each token to a few — have such an outsized effect on model size
          and cost.
        </p>
      </Callout>

      <p>
        Note again that <M>{"x"}</M> here is a <em>single</em> token's vector. In code we apply the
        same <M>{"W_1, W_2"}</M> to every position of the <M>{"(B, T, d)"}</M> tensor at once, but
        conceptually each token marches through the FFN alone. The shape comes out exactly as it
        went in — <M>{"(B, T, d)"}</M> — which is precisely what lets us drop the result back onto
        the residual stream.
      </p>

      <h2>Activations: GELU and SwiGLU</h2>
      <p>
        The nonlinearity in the middle of the FFN matters more than its simplicity suggests — it is
        the <em>only</em> source of nonlinearity in the entire per-token pathway. Without it, two
        stacked linear layers would collapse into a single linear map, and the FFN could compute
        nothing a plain matrix could not.
      </p>
      <p>
        The classic choice is <strong>ReLU</strong>, <M>{"\\text{ReLU}(x) = \\max(0, x)"}</M> — cheap
        and effective, but it has a hard kink at zero and kills the gradient entirely for any
        negative input (the "dying ReLU" problem). Modern transformers overwhelmingly prefer{" "}
        <strong>GELU</strong> (Gaussian Error Linear Unit), a smooth approximation that weights each
        input by the probability a standard Gaussian falls below it:
      </p>
      <MB>{String.raw`\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \tfrac{1}{2}\!\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right],`}</MB>
      <p>
        where <M>{"\\Phi"}</M> is the standard normal CDF. Intuitively, GELU is a "soft" gate: small
        negative inputs are not slammed to exactly zero but gently attenuated, and the curve is
        smooth everywhere, so gradients flow even on the negative side. GPT-2, GPT-3, and BERT all
        use GELU; it consistently trains a touch better than ReLU at no real cost.
      </p>
      <p>
        The current state of the art goes one step further with a <strong>gated</strong> activation,
        most popularly <strong>SwiGLU</strong> (used in LLaMA, PaLM, Mistral, and most recent open
        models). A gated FFN splits the up-projection into <em>two</em> parallel branches and uses
        one to gate the other elementwise:
      </p>
      <MB>{String.raw`\text{SwiGLU}(x) = \big(\,\text{Swish}(W_1 x)\;\odot\;(W_3 x)\,\big)\,W_2,\qquad \text{Swish}(z)=z\cdot\sigma(z),`}</MB>
      <p>
        where <M>{"\\odot"}</M> is elementwise multiplication and <M>{"\\sigma"}</M> is the sigmoid.
        The intuition is that the <M>{"W_3 x"}</M> branch learns a data-dependent <em>gate</em> that
        decides, per hidden unit, how much of the <M>{"\\text{Swish}(W_1 x)"}</M> signal to let
        through — a more expressive nonlinearity than a single fixed curve. The catch is the extra
        matrix <M>{"W_3"}</M>: a gated FFN has three weight matrices instead of two. To keep the
        parameter budget unchanged, gated models shrink the hidden dimension, typically from{" "}
        <M>{"4d"}</M> to about <M>{"\\tfrac{8}{3}d"}</M>, so that{" "}
        <M>{"3 \\times \\tfrac{8}{3}d \\cdot d = 8d^2"}</M> matches the ungated{" "}
        <M>{"2 \\times 4d \\cdot d"}</M>.
      </p>

      <Callout type="history" title="From ReLU to GELU to gates">
        <p>
          The 2017 transformer used ReLU. GPT-2/BERT switched to <strong>GELU</strong> (2018) for a
          smoother gradient. Around 2020 Noam Shazeer's "GLU Variants Improve Transformer" showed
          that <strong>gated</strong> activations — GEGLU, SwiGLU — beat their plain counterparts,
          and they have since become standard in frontier open models (LLaMA, Mistral, Gemma). If
          you are building a model today, GELU is the safe default and SwiGLU is the modern upgrade;
          either is correct.
        </p>
      </Callout>

      <h2>Residual connections: the gradient highway</h2>
      <p>
        Now the single most important structural idea for making deep networks trainable — more
        load-bearing, in a sense, than attention itself. Instead of <em>replacing</em> a layer's
        input with its output, we <em>add</em> the output back onto the input:
      </p>
      <MB>{String.raw`y = x + f(x).`}</MB>
      <p>
        That little <M>{"+\\,x"}</M> is a <strong>residual</strong> (or skip, or shortcut)
        connection, and it changes everything. The sub-layer <M>{"f"}</M> — attention or the FFN —
        no longer has to reproduce its input and edit it; it only has to compute the{" "}
        <em>change</em>, the residual <M>{"f(x) = y - x"}</M>. If the best thing a layer can do is
        nothing, it can simply learn <M>{"f(x) \\approx 0"}</M> and pass <M>{"x"}</M> through
        untouched. Learning a small correction is far easier than learning an entire transformation
        from scratch, which is why residual networks (ResNets, 2015) first made it possible to train
        networks hundreds of layers deep.
      </p>
      <p>
        The deeper reason is about <strong>gradients</strong>. Consider the problem residuals solve.
        In a plain deep stack <M>{"x_{\\ell} = f_{\\ell}(x_{\\ell-1})"}</M>, the gradient that
        reaches the bottom layer is a long product of every layer's Jacobian. If each of those
        factors has magnitude a little below 1, the product shrinks geometrically and the gradient{" "}
        <strong>vanishes</strong> before it reaches the early layers; a little above 1 and it{" "}
        <strong>explodes</strong>. Either way the bottom of a deep network barely learns. This{" "}
        <em>vanishing/exploding gradient</em> problem is exactly what kept networks shallow for
        decades.
      </p>
      <p>
        A residual connection cuts a clean path through it. Differentiate <M>{"y = x + f(x)"}</M>:
      </p>
      <MB>{String.raw`\frac{\partial y}{\partial x} = \underbrace{I}_{\text{identity path}} + \frac{\partial f}{\partial x}.`}</MB>
      <p>
        That identity term <M>{"I"}</M> means the gradient always has a route that passes through{" "}
        <em>unchanged</em>, no matter what <M>{"f"}</M> does to its part. Chain a hundred of these
        together and the gradient at the bottom is the top gradient plus a sum of branch
        contributions — it never has to survive a hundredfold product. The residual stream is, quite
        literally, a <strong>highway for gradients</strong> running straight from the loss to the
        embeddings.
      </p>

      <Callout type="key" title="Residuals are the gradient highway">
        <p>
          Writing each sub-layer as <M>{"y = x + f(x)"}</M> gives every layer a direct{" "}
          <strong>identity path</strong> to the loss: <M>{"\\partial y / \\partial x = I + \\partial f/\\partial x"}</M>.
          Gradients flow to the earliest layers without being attenuated by a long product of
          Jacobians, which is what defeats the vanishing-gradient problem and makes 50-, 100-, even
          1000-layer transformers trainable. Forward, the same path lets information{" "}
          <em>accumulate</em> rather than be overwritten — the residual stream we met in Chapter 7.
        </p>
      </Callout>

      <p>
        Toggle the <strong>"Why residuals"</strong> view in the figure below and drag the depth
        slider. Without the skip connection, a representative deep stack attenuates the signal layer
        by layer until it vanishes; with the residual, the magnitude holds steady at any depth. That
        contrast, in miniature, is why your 96-layer model trains at all.
      </p>

      <Figure
        n="11.1"
        title="Anatomy of a pre-norm block — and why residuals matter"
        caption="Left view: one pre-norm transformer block. Click any node to see its formula and tensor shape; the two blue dashed arrows are the residual skips — the identity path. Right view: drag the depth slider to watch a signal vanish through a plain stack (rose) while the residual stream (cyan) stays near 1.0 at any depth."
      >
        <BlockAnatomy />
      </Figure>

      <h2>Normalization</h2>
      <p>
        Residuals keep gradients flowing, but they do not, on their own, keep the{" "}
        <em>activations</em> in a sane numeric range — and a stack that keeps <em>adding</em> to a
        running vector can let its scale drift. Normalization is the cure. The idea: before feeding a
        vector into a sub-layer, rescale it so its entries have a controlled mean and variance, which
        keeps every layer operating in the well-behaved part of its nonlinearities and gives gradient
        descent a smoother surface to descend.
      </p>
      <p>
        <strong>LayerNorm</strong> is the transformer's workhorse. For a single token's vector{" "}
        <M>{"x \\in \\mathbb{R}^{d}"}</M>, it normalizes <em>across the feature dimension</em>:
        compute that vector's own mean and variance over its <M>{"d"}</M> entries, standardize, then
        rescale and shift by learned per-feature parameters <M>{"\\gamma"}</M> and <M>{"\\beta"}</M>:
      </p>
      <MB label="11.2">{String.raw`\mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}\big(x_i-\mu\big)^2,\qquad \text{LN}(x) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\,\odot\,\gamma + \beta.`}</MB>
      <p>
        The small <M>{"\\epsilon"}</M> (typically <M>{"10^{-5}"}</M>) just guards against dividing by
        zero. Crucially, LayerNorm acts <em>per token, per example</em>, using only that one vector's
        statistics — it does not mix information across the batch or across positions, which is why
        it works identically at training time (big batches) and at inference (one token at a time).
        That independence from batch statistics is exactly why transformers use LayerNorm rather than
        the BatchNorm common in vision.
      </p>
      <p>
        Modern models often simplify LayerNorm to <strong>RMSNorm</strong> (root-mean-square norm),
        which drops the mean-subtraction and the <M>{"\\beta"}</M> shift entirely, keeping only the
        scaling by the root-mean-square of the entries:
      </p>
      <MB>{String.raw`\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}\,\odot\,\gamma,\qquad \text{i.e.}\quad \frac{x}{\text{RMS}(x)}\,\odot\,\gamma.`}</MB>
      <p>
        RMSNorm asks: why bother re-centering at all? Empirically the mean-subtraction contributes
        little, so removing it (and the bias) saves compute and parameters while training just as
        well. LLaMA and most recent open models use RMSNorm; GPT-2 and the original transformer use
        full LayerNorm. Both serve the same purpose — pin the scale of the vector entering each
        sub-layer.
      </p>

      <Callout type="math" title="Why normalization stabilizes training">
        <p>
          Two effects. First, it bounds the <em>scale</em> of activations entering each sub-layer, so
          a deep stack cannot let magnitudes drift toward overflow or collapse — the inputs to every
          attention and FFN are kept unit-ish regardless of depth. Second, by fixing the variance it
          makes the loss surface better-conditioned: the learned <M>{"\\gamma"}</M> (and{" "}
          <M>{"\\beta"}</M>) let each layer choose the scale it actually wants, decoupled from the
          raw magnitude of whatever flowed in. The practical payoff is that you can use a larger
          learning rate and far less hand-holding to train a deep model.
        </p>
      </Callout>

      <h2>Pre-norm vs. post-norm</h2>
      <p>
        Where exactly the normalization sits, relative to the residual add, turns out to matter a
        great deal. There are two arrangements, and the field decisively switched from one to the
        other.
      </p>
      <p>
        The original 2017 transformer used <strong>post-norm</strong>: apply the sub-layer, add the
        residual, <em>then</em> normalize the sum.
      </p>
      <MB>{String.raw`\text{post-norm:}\qquad x \;\leftarrow\; \text{LN}\big(x + f(x)\big).`}</MB>
      <p>
        This works, but notice that the normalization sits <em>on</em> the residual path — every
        trip up the stack passes through a LayerNorm, so the clean identity highway is interrupted at
        every layer. Deep post-norm models are finicky: they need careful learning-rate warmup and
        are prone to instability, because the gradient still has to thread through all those norms.
      </p>
      <p>
        Modern models use <strong>pre-norm</strong>: normalize <em>first</em>, run the sub-layer on
        the normalized vector, and add the result back onto the <em>un-normalized</em> input.
      </p>
      <MB>{String.raw`\text{pre-norm:}\qquad x \;\leftarrow\; x + f\big(\text{LN}(x)\big).`}</MB>
      <p>
        Here the normalization lives <em>inside the residual branch</em>. The <M>{"+\\,x"}</M> path
        is now a completely clean, un-normalized shortcut from the bottom of the network to the top —
        the gradient highway is uninterrupted. That single change is what lets very deep transformers
        train stably without elaborate warmup schedules, and it is why essentially every LLM since
        GPT-2 — GPT-2/3, LLaMA, Mistral, and the rest — is pre-norm. (One small bookkeeping detail:
        because nothing normalizes the stream on the way out, pre-norm models add one{" "}
        <strong>final LayerNorm</strong> after the last block, which you met as <M>{"\\text{ln\\_f}"}</M>{" "}
        in Chapter 7.)
      </p>

      <Callout type="pitfall" title="Don't normalize the skip path">
        <p>
          The whole point of pre-norm is that the residual shortcut stays clean. A tempting bug is to
          write <M>{"x \\leftarrow \\text{LN}(x) + f(\\text{LN}(x))"}</M> — normalizing the skip term
          too. That reintroduces a normalization on the identity path and throws away the very
          stability you switched to pre-norm to get. The skip must carry the{" "}
          <em>raw</em> <M>{"x"}</M>: <M>{"x \\leftarrow x + f(\\text{LN}(x))"}</M>. Likewise, do not
          forget the final norm after the last block — without it the output of a pre-norm stack is
          never normalized at all, and the LM head sees an un-scaled vector.
        </p>
      </Callout>

      <h2>Dropout & regularization</h2>
      <p>
        One more ingredient rounds out the block: <strong>dropout</strong>, a simple and effective
        regularizer. During training, dropout randomly zeroes a fraction <M>{"p"}</M> of the
        activations (and rescales the survivors by <M>{"1/(1-p)"}</M> so the expected magnitude is
        unchanged). This forces the network not to rely on any single unit, acting like training an
        ensemble of thinned subnetworks and reducing overfitting. At <em>inference</em> time dropout
        is turned off entirely — every unit is used.
      </p>
      <p>
        In a transformer block, dropout is usually applied in a few places: on the attention weights,
        and on the output of each sub-layer just before it is added back to the residual stream
        (often called <em>residual dropout</em>). GPT-2 uses <M>{"p = 0.1"}</M>. That said, dropout
        matters most when you risk overfitting — i.e., when the model is large relative to the data.
        The largest modern LLMs, trained on enormous corpora for roughly a single pass, frequently
        set dropout to <strong>zero</strong>, because with that much fresh data the model never
        sees the same example twice and there is little to overfit to. Weight decay (an{" "}
        <M>{"L_2"}</M> penalty on the weights) typically does the regularization heavy-lifting
        instead. We cover the full optimization recipe — learning rate, warmup, weight decay — in{" "}
        <a href="/chapter/training-loop" className="prose-link">
          the training chapter
        </a>
        .
      </p>

      <h2>Assembling the full block</h2>
      <p>
        We now have every piece. Stack them in the pre-norm arrangement and a complete transformer
        block is exactly two residual updates — communication, then computation:
      </p>
      <MB>{String.raw`\begin{aligned} x &\;\leftarrow\; x + \text{MultiHeadAttention}\big(\text{LayerNorm}(x)\big) &&\text{(mix across tokens)}\\[0.5em] x &\;\leftarrow\; x + \text{FeedForward}\big(\text{LayerNorm}(x)\big) &&\text{(think per token)} \end{aligned}`}</MB>
      <p>
        Read it against the diagram: the stream enters, a LayerNorm and attention compute a
        contribution that is added back, a second LayerNorm and the FFN compute another contribution
        that is added back, and the stream leaves with the same <M>{"(B, T, d)"}</M> shape it
        arrived with. Because the shape is preserved, you can stack <M>{"N"}</M> of these identical
        blocks — the depth dial of the whole model.
      </p>
      <p>
        Here is the entire block in PyTorch, with a from-scratch <code>RMSNorm</code> module beside
        the standard <code>nn.LayerNorm</code> so you can see both. This is real, runnable code — the
        production version of the skeleton stubbed in Chapter 7.
      </p>

      <CodeBlock language="python" filename="block.py" highlight={[31, 32, 33, 34]}>
{`import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm: scale by RMS over the feature dim, no
    mean-subtraction and no bias. Used by LLaMA-style models."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # learned per-feature gain

    def forward(self, x):                          # x: (B, T, d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.gamma                # normalize, then rescale


class FeedForward(nn.Module):
    """Position-wise FFN: d -> 4d -> d with a GELU nonlinearity."""
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, mult * d_model)   # expand   d  -> 4d
        self.fc2  = nn.Linear(mult * d_model, d_model)   # project  4d -> d
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                          # x: (B, T, d)
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """One pre-norm transformer block. Shape in == shape out: (B, T, d)."""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)                # pre-attention norm
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)                # pre-FFN norm
        self.ffn  = FeedForward(d_model, mult=4, dropout=dropout)
        self.drop = nn.Dropout(dropout)                  # residual dropout

    def forward(self, x, attn_mask=None):                # x: (B, T, d)
        # --- sub-layer 1: communication (attention) ---
        h = self.ln1(x)                                  # pre-norm
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,   # self-attention
                         need_weights=False)
        x = x + self.drop(a)                             # residual add (keep raw x!)
        # --- sub-layer 2: computation (FFN) ---
        x = x + self.ffn(self.ln2(x))                    # pre-norm + residual add
        return x                                          # (B, T, d), unchanged shape`}
      </CodeBlock>

      <p>
        Read the two highlighted lines of <code>forward</code> as the heart of it: each sub-layer
        reads a normalized copy of the stream, computes a contribution, and adds it straight back
        onto the raw <M>{"x"}</M>. The <code>attn_mask</code> threads the causal mask through to the
        attention call so the block stays a decoder. Swap <code>nn.LayerNorm</code> for{" "}
        <code>RMSNorm</code> and <code>F.gelu</code> for a SwiGLU and you have a modern LLaMA-style
        block; the structure is identical.
      </p>

      <Callout type="tip" title="The whole block, distilled">
        <p>
          Two pre-norm sub-layers, two residual adds:{" "}
          <code>x = x + attn(ln1(x))</code> then <code>x = x + ffn(ln2(x))</code>. Attention is the
          only cross-token step; the FFN (<M>{"d \\to 4d \\to d"}</M>) is the per-token step and
          holds ~2/3 of the parameters; residuals give gradients an identity path; pre-norm keeps
          that path clean. Memorize those four facts and you understand the block.
        </p>
      </Callout>

      <h2>Where we go next</h2>
      <p>
        The architecture is now complete. You have walked the whole stack — embeddings lift token IDs
        into vectors, positional information tells them where they sit, attention lets every token
        gather context from the others, the feed-forward network thinks about each token on its own,
        and residual connections wrapped in normalization let you stack dozens of these blocks into a
        deep, trainable network. A final norm and an unembedding turn the top of the stack into
        next-token logits. Nothing about the model is mysterious anymore; it is matrices, a softmax,
        and a great deal of careful plumbing.
      </p>
      <p>
        But a freshly initialized transformer is just random weights — it produces gibberish. The
        next part of the book breathes life into it. We define the{" "}
        <a href="/chapter/training-objective" className="prose-link">
          training objective
        </a>{" "}
        that turns next-token prediction into a single number to minimize, then build the{" "}
        <a href="/chapter/training-loop" className="prose-link">
          training loop
        </a>{" "}
        — the optimizer, the learning-rate schedule, the gradient updates — that drives those weights
        from noise toward a model that can actually write. The blueprint is finished; time to build.
      </p>
    </>
  );
}
