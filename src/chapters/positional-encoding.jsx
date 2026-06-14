import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import PositionalViz from "../components/viz/positional-encoding/PositionalViz";

export default function Chapter() {
  return (
    <>
      <p>
        We just gave the transformer many ways to look at its tokens. But there is a hole at the
        center of everything we’ve built: <strong>attention doesn’t know what order the tokens are
        in.</strong> To it, <em>“dog bites man”</em> and <em>“man bites dog”</em> are the same bag of
        words. That’s a catastrophe for language, where order is meaning. This chapter is about
        plugging that hole — injecting <em>position</em> into the model. We’ll go from the classic
        sinusoidal encodings, through the learned embeddings GPT-2 uses, to <strong>Rotary Position
        Embeddings (RoPE)</strong>, the scheme that powers essentially every frontier model today.
      </p>

      <h2>Attention forgets order</h2>
      <p>
        Let’s prove the claim, because it’s the entire motivation. Recall self-attention computes,
        for the stack of token vectors <M>{"X"}</M>:
      </p>
      <MB>{String.raw`\text{Attention}(X) = \text{softmax}\!\left(\frac{(XW_Q)(XW_K)^{\top}}{\sqrt{d_k}}\right)(XW_V).`}</MB>
      <p>
        Now imagine permuting the input rows — shuffling which token sits in which position — with a
        permutation matrix <M>{"P"}</M> (a matrix that just reorders rows). Feed in <M>{"PX"}</M>{" "}
        instead of <M>{"X"}</M>. Every projection sees the rows in the new order, the score matrix
        gets both its rows <em>and</em> columns permuted (<M>{"P\\,\\text{scores}\\,P^{\\top}"}</M>),
        the softmax is computed row-wise so it commutes with the row permutation, and the final
        weighted sum comes out reordered the same way. The upshot:
      </p>
      <MB>{String.raw`\text{Attention}(PX) = P\,\text{Attention}(X).`}</MB>
      <p>
        Read that carefully: shuffling the inputs simply shuffles the outputs <em>identically</em>.
        The function is <strong>permutation-equivariant</strong> — it operates on the tokens as an
        unordered <em>set</em>. Token <M>{"i"}</M>’s output depends on <em>which</em> tokens are
        present and their content, never on <em>where</em> they sit. Nothing inside attention can
        distinguish the first word from the fifth. Position information must be added from the
        outside.
      </p>

      <Callout type="key" title="The whole problem">
        <p>
          Self-attention is a <em>set</em> operation: permute the inputs and the outputs permute the
          same way (<M>{"\\text{Attention}(PX) = P\\,\\text{Attention}(X)"}</M>). It is blind to
          word order. We must explicitly inject each token’s position, or the model literally cannot
          tell <em>“dog bites man”</em> from <em>“man bites dog.”</em>
        </p>
      </Callout>

      <Callout type="pitfall" title="The MLP and softmax don’t save you">
        <p>
          You might hope the per-token feed-forward layers or the softmax break the symmetry. They
          don’t. The MLP is applied to each position <em>independently</em> with shared weights, so
          it’s also permutation-equivariant. The softmax is row-wise, so it commutes with row
          permutations too. <em>Every</em> component of the transformer is order-agnostic unless we
          add positional information. There is no accidental back door.
        </p>
      </Callout>

      <h2>Sinusoidal positional encoding</h2>
      <p>
        The original transformer’s answer: build a fixed vector for each position and{" "}
        <strong>add</strong> it to the token embedding before the first layer. The clever part is{" "}
        <em>how</em> those vectors are built — out of sines and cosines at geometrically spaced
        frequencies. For position <M>{"pos"}</M> and embedding dimension index <M>{"i"}</M> (with
        embedding size <M>{"d"}</M>):
      </p>
      <MB>{String.raw`\begin{aligned}
PE_{(pos,\,2k)} &= \sin\!\left(\frac{pos}{10000^{\,2k/d}}\right) \\[0.6em]
PE_{(pos,\,2k+1)} &= \cos\!\left(\frac{pos}{10000^{\,2k/d}}\right)
\end{aligned}`}</MB>
      <p>
        Dimensions come in <M>{"(\\sin, \\cos)"}</M> pairs. The index <M>{"k"}</M> runs over those
        pairs, and the term <M>{"10000^{2k/d}"}</M> sets each pair’s <strong>wavelength</strong>. For
        the first pairs the denominator is small, so the angle <M>{"pos / 10000^{2k/d}"}</M> sweeps
        fast — these are short-wavelength, fast-oscillating dimensions. For the last pairs the
        denominator is huge (up to <M>{"\\approx 10000"}</M>), so the angle barely moves across the
        whole sequence — long, slow oscillations. Wavelengths form a geometric series from{" "}
        <M>{"2\\pi"}</M> up to about <M>{"2\\pi \\cdot 10000"}</M>.
      </p>
      <p>
        The intuition is a <strong>multi-frequency clock</strong>. A clock tells time with hands at
        different speeds: the second hand pins down fine detail, the hour hand the coarse range. Read
        together, the hands give an unambiguous timestamp. Likewise, the fast dimensions distinguish
        nearby positions sharply while the slow dimensions disambiguate far-apart positions — and the
        full vector is a unique fingerprint for each position. Toggle to the <em>Sinusoidal
        heatmap</em> below: each row is one position’s fingerprint, left columns are the fast hands,
        right columns the slow ones.
      </p>

      <Figure
        n="10.1"
        title="Positional encodings, computed live"
        caption="Left view: the sinusoidal PE matrix (rows = positions, columns = dimensions) straight from the formula — note the fast oscillations on the left fading to slow waves on the right. Right view: RoPE, where rotating two token vectors by an angle proportional to position makes their dot product depend only on the gap between them."
      >
        <PositionalViz />
      </Figure>

      <Callout type="math" title="Why sinusoids? Relative position is a linear map">
        <p>
          Sinusoids have a magic property: a shift in position is a <em>rotation</em>. For any fixed
          offset <M>{"\\Delta"}</M>, there is a matrix (independent of <M>{"pos"}</M>) that maps{" "}
          <M>{"PE_{pos}"}</M> to <M>{"PE_{pos+\\Delta}"}</M> — because{" "}
          <M>{"\\sin(\\theta+\\phi)"}</M> and <M>{"\\cos(\\theta+\\phi)"}</M> are linear combinations
          of <M>{"\\sin\\theta, \\cos\\theta"}</M>. This lets the model learn to attend by{" "}
          <em>relative</em> offset, and lets the encodings extrapolate smoothly to positions longer
          than any seen in training. This same rotation idea, applied directly to Q and K, becomes
          RoPE.
        </p>
      </Callout>

      <h2>Learned positional embeddings</h2>
      <p>
        There’s a blunter alternative: don’t engineer anything — just <strong>learn</strong> a
        position vector the same way you learn a token vector. Allocate a trainable lookup table{" "}
        <M>{"E_{\\text{pos}} \\in \\mathbb{R}^{T_{\\max} \\times d_{\\text{model}}}"}</M> with one row
        per position up to a maximum length <M>{"T_{\\max}"}</M>, and add row <M>{"t"}</M> to the
        token at position <M>{"t"}</M>. Gradient descent figures out what each position vector should
        be.
      </p>
      <p>
        This is what <strong>GPT-2</strong> (and the original GPT) does. The final input to the
        first transformer block is simply the sum of a token embedding and a learned positional
        embedding:
      </p>
      <MB>{String.raw`\text{input}_t = E_{\text{token}}[\,x_t\,] + E_{\text{pos}}[\,t\,].`}</MB>
      <p>
        Learned embeddings are dead simple and let the model use positions however it likes. The
        cost: they add parameters (<M>{"T_{\\max}\\cdot d_{\\text{model}}"}</M> of them) and they{" "}
        <strong>cannot generalize beyond <M>{"T_{\\max}"}</M></strong>. There is no row for position
        2048 if you only trained up to 1024 — the model has literally never built a representation
        for that slot. This hard context wall is a real operational limit of GPT-2-style models.
      </p>

      <Callout type="industry" title="Absolute encodings in the wild">
        <p>
          The original transformer and BERT use <em>sinusoidal absolute</em> encodings; GPT-1/GPT-2
          use <em>learned absolute</em> embeddings. Both add a position signal once, at the input. In
          practice the two perform comparably at fixed length — the decisive difference shows up at
          the edges: extrapolation to longer contexts and the way position interacts with attention,
          which is exactly what the next approaches improve.
        </p>
      </Callout>

      <h2>The problem with absolute positions</h2>
      <p>
        Both schemes above encode <em>absolute</em> position — “you are token number 5.” But what
        attention usually cares about is <em>relative</em> position — “the token three steps back.”
        The phrase <em>“New York”</em> should behave the same whether it lands at positions 5–6 or
        500–501; what matters is that those two tokens are adjacent. Absolute encodings force the
        model to <em>infer</em> relative offsets by subtracting two absolute signals, which is
        wasteful and fragile, and they make every position past <M>{"T_{\\max}"}</M> a stranger.
      </p>
      <p>
        We want a scheme where the attention score between tokens <M>{"m"}</M> and <M>{"n"}</M>{" "}
        depends only on their content and on the <em>difference</em> <M>{"m - n"}</M>, never on the
        absolute values. That is precisely what rotary embeddings deliver.
      </p>

      <h2>Rotary embeddings (RoPE)</h2>
      <p>
        Rotary Position Embedding (RoPE), introduced in the RoFormer paper (Su et al., 2021), is a
        beautifully direct idea: <strong>don’t add anything to the embeddings — instead rotate the
        query and key vectors by an angle proportional to their position.</strong> The dot product
        of two rotated vectors then depends only on how much they were rotated <em>relative</em> to
        each other, i.e. on the position difference.
      </p>
      <p>
        Work in 2-D first. Take one pair of features <M>{"(x_1, x_2)"}</M> from a token at position{" "}
        <M>{"m"}</M>. RoPE multiplies it by the standard 2-D rotation matrix through angle{" "}
        <M>{"m\\theta"}</M>:
      </p>
      <MB>{String.raw`R_{m\theta}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \phantom{-}\cos m\theta \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}.`}</MB>
      <p>
        Apply <M>{"R_{m\\theta}"}</M> to the query at position <M>{"m"}</M> and{" "}
        <M>{"R_{n\\theta}"}</M> to the key at position <M>{"n"}</M>. Now use the key fact that
        rotations compose by adding angles and that{" "}
        <M>{"R_a^{\\top} R_b = R_{b-a}"}</M>. The attention logit — their dot product — becomes:
      </p>
      <MB>{String.raw`(R_{m\theta}\,q)^{\top}(R_{n\theta}\,k) = q^{\top} R_{m\theta}^{\top} R_{n\theta}\, k = q^{\top} R_{(n-m)\theta}\, k.`}</MB>
      <p>
        The absolute positions <M>{"m"}</M> and <M>{"n"}</M> have vanished — only the difference{" "}
        <M>{"n - m"}</M> survives, baked right into the angle of a single rotation. The score is
        automatically a function of <em>relative</em> position, with no extra parameters and no terms
        added to the values. The <em>RoPE</em> view in the figure above makes this tangible: rotate
        both token arrows by their positions and the wedge between them — hence their dot product —
        depends only on the gap.
      </p>
      <p>
        For the full <M>{"d_k"}</M>-dimensional head, RoPE simply does this independently on each
        consecutive pair of dimensions, giving each pair its <em>own</em> frequency{" "}
        <M>{"\\theta_k = 10000^{-2k/d_k}"}</M> — the very same geometric frequency ladder as the
        sinusoidal scheme. Fast-rotating pairs encode fine relative offsets; slow-rotating pairs
        encode coarse ones.
      </p>

      <Callout type="key" title="Why RoPE became the standard">
        <p>
          RoPE is <strong>relative</strong> (scores depend on <M>{"n-m"}</M>, the thing attention
          actually wants), adds <strong>zero parameters</strong> (it’s a fixed rotation, not a
          learned table), is applied <strong>only to Q and K</strong> (values are untouched), and{" "}
          <strong>extrapolates</strong> gracefully to longer contexts — often with a small tweak. For
          these reasons it powers LLaMA, GPT-NeoX, PaLM, Mistral, Qwen, and most modern open models.
        </p>
      </Callout>

      <Callout type="math" title="Rotations are norm-preserving">
        <p>
          A rotation matrix is orthogonal: it changes a vector’s <em>direction</em> but never its{" "}
          <em>length</em>. So RoPE doesn’t inflate or shrink the magnitude of Q or K — it can’t
          accidentally blow up the attention logits’ scale. It re-aims the vectors so their{" "}
          <em>alignment</em> carries relative-position information, while leaving everything else
          about their geometry intact.
        </p>
      </Callout>

      <h2>Long-context &amp; extrapolation</h2>
      <p>
        Modern LLMs advertise context windows of tens or hundreds of thousands of tokens, far beyond
        their original training length. RoPE makes stretching feasible. Because positions enter as
        rotation <em>angles</em>, you can change the angles to remap a long sequence onto the range
        the model already understands:
      </p>
      <ul>
        <li>
          <strong>Position interpolation (NTK / linear scaling):</strong> divide every position by a
          factor <M>{"s"}</M> before computing angles, squeezing, say, 8× the tokens into the
          original angular range. Combined with a little fine-tuning, this cheaply extends a model
          trained at 4k tokens to 32k+ — the technique behind many long-context releases.
        </li>
        <li>
          <strong>NTK-aware / “YaRN” scaling:</strong> scale the low (fast) and high (slow)
          frequencies <em>unequally</em>, preserving fine local resolution while stretching the
          coarse ranges — extrapolating further with less quality loss than naive linear scaling.
        </li>
      </ul>
      <p>
        A different philosophy is <strong>ALiBi</strong> (Attention with Linear Biases, Press et
        al., 2021): use <em>no</em> positional embeddings at all. Instead, add a static penalty to
        each attention score that grows linearly with the distance between query and key, with a
        head-specific slope:
      </p>
      <MB>{String.raw`\text{score}_{m,n} \;=\; \frac{q_m \cdot k_n}{\sqrt{d_k}} \;-\; \lambda\,|m - n|.`}</MB>
      <p>
        The bias makes distant tokens cheaper to ignore and, crucially, depends only on the
        distance, so ALiBi extrapolates to far longer sequences than it trained on essentially for
        free. RoPE and ALiBi share a goal — relative position and length generalization — and modern
        practice leans heavily on RoPE (often with scaling), with ALiBi a lightweight alternative
        used in models like BLOOM.
      </p>

      <Callout type="tip" title="What to reach for today">
        <p>
          Building a transformer in 2024-and-after? Use <strong>RoPE</strong>. It’s the default in
          essentially every strong open model, integrates cleanly with FlashAttention, and gives you
          relative positions and a clear path to long context via scaling. Learned absolute
          embeddings remain fine for fixed, short context lengths; sinusoidal encodings are a great
          teaching tool and still appear in encoder models.
        </p>
      </Callout>

      <h2>Implementing it from scratch</h2>
      <p>
        Two snippets. First, the classic sinusoidal table, built once and added to embeddings.
        Second, <code>apply_rotary_emb()</code>, which rotates Q and K in place — the function that
        runs inside a modern attention block, right after the Q/K projections and before the dot
        product.
      </p>

      <CodeBlock language="python" filename="positional.py" highlight={[7, 8, 9, 10]}>
{`import torch


def sinusoidal_pe(seq_len, d_model):
    """Fixed sinusoidal positional encodings -> (seq_len, d_model)."""
    pos = torch.arange(seq_len).unsqueeze(1)            # (seq_len, 1)
    k = torch.arange(0, d_model, 2)                     # pair indices: 0,2,4,...
    div = torch.pow(10000.0, k / d_model)              # 10000^(2k/d)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / div)                  # even dims -> sin
    pe[:, 1::2] = torch.cos(pos / div)                  # odd dims  -> cos
    return pe


# Usage: add to token embeddings at the input.
#   x = token_emb(ids) + sinusoidal_pe(T, d_model).to(x.device)`}
      </CodeBlock>

      <CodeBlock language="python" filename="rope.py" highlight={[18, 19, 20, 21]}>
{`import torch


def rope_freqs(seq_len, d_k, base=10000.0):
    """Precompute the cos/sin tables for RoPE. d_k must be even."""
    # One frequency per dimension *pair*: theta_k = base^(-2k/d_k).
    k = torch.arange(0, d_k, 2).float()                 # 0, 2, ..., d_k-2
    inv_freq = 1.0 / (base ** (k / d_k))                # (d_k/2,)
    pos = torch.arange(seq_len).float()                 # (seq_len,)
    angles = torch.outer(pos, inv_freq)                 # (seq_len, d_k/2)
    # Duplicate each angle so it lines up with both halves of the vector.
    angles = torch.cat([angles, angles], dim=-1)        # (seq_len, d_k)
    return angles.cos(), angles.sin()


def _rotate_half(x):
    # Maps (x1, x2) pairs to (-x2, x1): the 90-degree part of the rotation.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """Rotate q and k by their positions. q, k: (B, H, T, d_k).

    Implements R_{pos*theta} @ x  via  x*cos + rotate_half(x)*sin,
    the standard, dot-product-friendly form of the 2-D rotation.
    """
    cos = cos[None, None, :, :]                         # broadcast over (B, H)
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# Inside attention, after projecting q, k and before scores = q @ k^T:
#   cos, sin = rope_freqs(T, d_k)
#   q, k = apply_rotary_emb(q, k, cos.to(q.device), sin.to(q.device))`}
      </CodeBlock>

      <p>
        The <code>apply_rotary_emb</code> identity{" "}
        <M>{"R_{m\\theta}x = x\\cos(m\\theta) + \\text{rotate\\_half}(x)\\sin(m\\theta)"}</M> is just
        the 2-D rotation written so it vectorizes over all dimension-pairs at once — it’s
        algebraically identical to multiplying each pair by its <M>{"2\\times 2"}</M> rotation matrix.
        Notice RoPE touches <strong>only q and k</strong>; the values <code>v</code> flow through
        untouched, exactly as the math promised.
      </p>

      <Callout type="pitfall" title="RoPE goes on Q and K, inside each head">
        <p>
          A frequent bug is rotating the values, or rotating before the per-head reshape. RoPE must
          be applied to <strong>q and k only</strong>, on the <strong>per-head</strong> tensors of
          shape <M>{"(B, H, T, d_k)"}</M>, so each head rotates within its own subspace with{" "}
          <M>{"d_k"}</M> (even!) dimensions. Rotating <code>v</code>, or applying it to the full{" "}
          <M>{"d_{\\text{model}}"}</M> before splitting heads, quietly corrupts the relative-position
          guarantee.
        </p>
      </Callout>

      <h2>Where we go next</h2>
      <p>
        Order is no longer invisible. Whether by adding a sinusoidal or learned vector at the input,
        or — better — by rotating queries and keys so attention scores read relative position
        directly, the model can finally tell <em>“dog bites man”</em> from <em>“man bites dog.”</em>{" "}
        Together with multi-head attention, we now have the two ideas that make the attention sublayer
        complete.
      </p>
      <p>
        A transformer block is more than attention, though. Each block pairs the attention sublayer
        with a position-wise feed-forward network, and stitches the two together with{" "}
        <strong>residual connections</strong> and <strong>layer normalization</strong> — the
        plumbing that lets us stack dozens of these blocks into a deep, trainable network. That
        assembly is the subject of the chapters ahead, where we put attention, position, the MLP, and
        the normalization machinery together into the full transformer block.
      </p>
    </>
  );
}
