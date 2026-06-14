import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import EmbeddingSpace from "../components/viz/embeddings/EmbeddingSpace";

export default function Chapter() {
  return (
    <>
      <p>
        We left the last chapter with a sequence of integers — token IDs in{" "}
        <M>{"[0, V)"}</M> — and a complaint: <strong>integers carry no meaning</strong>. ID 4087
        isn’t “closer” to ID 4088 than to ID 12; the numbering is arbitrary, an accident of how the
        tokenizer happened to sort its vocabulary. A neural network multiplies and adds, so feeding
        it raw IDs would force it to learn that <code>327</code> and <code>328</code> are unrelated
        despite being neighbours, and that two synonyms with distant IDs have nothing in common.
        This chapter fixes that. We give every token a <strong>learnable vector</strong>, turning
        discrete symbols into points in a continuous space where <em>geometry encodes meaning</em>.
      </p>

      <h2>The embedding table</h2>
      <p>
        An <strong>embedding</strong> is just a lookup table. We allocate one row of numbers per
        token in the vocabulary and call the whole thing the embedding matrix{" "}
        <M>{"E \\in \\mathbb{R}^{V \\times d}"}</M>, where <M>{"V"}</M> is the vocabulary size and{" "}
        <M>{"d"}</M> is the <strong>embedding dimension</strong> (also written <M>{"d_{model}"}</M>)
        — the length of each vector. To embed token with ID <M>{"i"}</M>, you read row <M>{"i"}</M>:
      </p>
      <MB>{String.raw`\mathbf{e}_i \;=\; E[i, :] \;\in\; \mathbb{R}^{d}`}</MB>
      <p>
        That’s it — a single array index. For a sequence of <M>{"T"}</M> token IDs we gather{" "}
        <M>{"T"}</M> rows and get a matrix <M>{"X \\in \\mathbb{R}^{T \\times d}"}</M>: the input to
        the rest of the model. With a vocabulary of <M>{"V = 50{,}000"}</M> and{" "}
        <M>{"d = 768"}</M> (GPT-2 small), the table holds{" "}
        <M>{"50{,}000 \\times 768 \\approx 38\\text{M}"}</M> learnable numbers — often the single
        largest parameter block in a small model.
      </p>

      <Callout type="math" title="Lookup is a one-hot matrix multiply">
        <p>
          A row lookup is secretly a matrix multiplication. Let <M>{"\\mathbf{o}_i"}</M> be the{" "}
          <strong>one-hot</strong> vector for ID <M>{"i"}</M> — a length-<M>{"V"}</M> vector that is
          1 at position <M>{"i"}</M> and 0 everywhere else. Then
        </p>
        <MB>{String.raw`\mathbf{o}_i^{\top} E \;=\; E[i, :] \;=\; \mathbf{e}_i,`}</MB>
        <p>
          because multiplying picks out exactly row <M>{"i"}</M>. This equivalence matters for two
          reasons: it shows the embedding is a perfectly ordinary linear layer (so gradients flow
          through it like any other weight), and it explains the efficient implementation — nobody
          actually builds the giant one-hot vector or does the multiply; you just index the row.
        </p>
      </Callout>

      <p>
        The rows start as small random numbers and are <strong>trained by gradient descent</strong>{" "}
        alongside everything else. There is no separate “learn the embeddings” phase — every time the
        model makes a prediction and the loss flows backward, the rows for the tokens involved get
        nudged. Over billions of tokens, this carves a richly structured space out of what began as
        noise.
      </p>

      <h2>Why vectors, not integers?</h2>
      <p>
        Suppose we skipped embeddings and fed the model the ID directly as a single number. Three
        things break:
      </p>
      <ul>
        <li>
          <strong>No notion of similarity.</strong> The model can only see “how far apart on the
          number line.” But adjacent IDs are unrelated and distant IDs are often synonyms — the one
          dimension it has carries no usable signal.
        </li>
        <li>
          <strong>A false ordering.</strong> Treating IDs as magnitudes implies{" "}
          <code>cat</code> &lt; <code>dog</code> &lt; <code>elephant</code> means something. It
          doesn’t. The model would waste capacity unlearning this fiction.
        </li>
        <li>
          <strong>One number can’t describe a word.</strong> “King” is royal, male, human, singular,
          a noun. That’s many independent attributes; you need many dimensions to place a word along
          each of them.
        </li>
      </ul>
      <p>
        A vector solves all three. With <M>{"d"}</M> dimensions, each token gets <M>{"d"}</M>{" "}
        coordinates the network can set freely, so it can place <em>related tokens near each other</em>{" "}
        and unrelated ones far apart. Crucially, the space is <strong>continuous</strong>: a tiny
        change to a vector is a tiny change in meaning. That smoothness is exactly what gradient
        descent needs — it can take small steps and improve, learning smooth functions of meaning
        rather than memorizing a lookup of discrete symbols.
      </p>

      <Callout type="key" title="The whole idea">
        <p>
          An embedding turns a meaningless integer ID into a point in <M>{"\\mathbb{R}^d"}</M> whose{" "}
          <em>location is learned</em>. After training, <strong>distance and direction in that space
          encode meaning</strong>: similar tokens cluster together, and consistent semantic
          relationships become consistent geometric directions.
        </p>
      </Callout>

      <h2>The geometry of meaning</h2>
      <p>
        Once tokens are vectors, “similar” becomes a measurable geometric quantity. The standard
        measure is <strong>cosine similarity</strong> — the cosine of the angle between two vectors,
        which ignores their lengths and asks only whether they point the same way:
      </p>
      <MB>{String.raw`\cos(\mathbf{u}, \mathbf{v}) \;=\; \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert} \;=\; \frac{\sum_{k=1}^{d} u_k v_k}{\sqrt{\sum_k u_k^2}\,\sqrt{\sum_k v_k^2}}`}</MB>
      <p>
        It ranges from <M>{"+1"}</M> (identical direction — very similar) through <M>{"0"}</M>{" "}
        (orthogonal — unrelated) to <M>{"-1"}</M> (opposite). We prefer cosine over raw Euclidean
        distance because, in high dimensions, the <em>direction</em> a vector points turns out to be
        a cleaner carrier of meaning than its magnitude (length often correlates with uninteresting
        things like word frequency).
      </p>
      <p>
        Plotted, a trained embedding space is full of structure. Words cluster by <strong>topic</strong>{" "}
        (all the animals near each other, all the country names elsewhere) and by{" "}
        <strong>part of speech</strong> (verbs occupy a different region than nouns). The figure below
        is a hand-built 2-D toy — a real table has hundreds of dimensions we can’t draw — but the
        clustering it shows is exactly what emerges in practice. Hover any word to see its nearest
        neighbours.
      </p>

      <Figure
        n="6.1"
        title="An embedding space you can explore"
        caption="A 2-D stand-in for a d-dimensional table. In Explore mode, hovering a word reveals its nearest neighbours — always from the same topical cluster. Switch to Analogy mode to do vector arithmetic on meaning."
      >
        <EmbeddingSpace />
      </Figure>

      <Callout type="pitfall" title="Two dimensions is a lie (a useful one)">
        <p>
          Real embeddings live in <M>{"d = 768"}</M>, <M>{"4096"}</M>, or more. To draw them we
          <em>project</em> down to 2-D with tools like PCA, t-SNE or UMAP — and projection always
          distorts. Two words that look adjacent on a t-SNE plot may not be true neighbours in the
          full space, and global distances are especially unreliable. Treat 2-D pictures as
          intuition, never as measurement. The honest comparison is cosine similarity in the original{" "}
          <M>{"d"}</M> dimensions.
        </p>
      </Callout>

      <h2>Directions are meaning: analogies</h2>
      <p>
        Here is the result that made embeddings famous. Not only do similar words cluster — consistent
        relationships become consistent <strong>directions</strong>. The vector from{" "}
        <code>man</code> to <code>woman</code> points in roughly the same direction as the vector from{" "}
        <code>king</code> to <code>queen</code>: both encode “make it feminine.” So you can do
        arithmetic on meaning:
      </p>
      <MB>{String.raw`\mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}} \;\approx\; \mathbf{e}_{\text{queen}}`}</MB>
      <p>
        Read it as: start at <em>king</em>, subtract the “maleness” you can read off from{" "}
        <M>{"\\mathbf{e}_{\\text{man}}"}</M>, add the “femaleness” from{" "}
        <M>{"\\mathbf{e}_{\\text{woman}}"}</M>, and you land essentially on <em>queen</em>. Whole
        families of relations show up as stable offset vectors:
      </p>
      <ul>
        <li>
          <strong>Gender:</strong> <M>{"\\text{actor} \\to \\text{actress}"}</M>,{" "}
          <M>{"\\text{uncle} \\to \\text{aunt}"}</M>.
        </li>
        <li>
          <strong>Plurality:</strong> <M>{"\\text{cat} \\to \\text{cats}"}</M>,{" "}
          <M>{"\\text{mouse} \\to \\text{mice}"}</M>.
        </li>
        <li>
          <strong>Verb tense:</strong> <M>{"\\text{walk} \\to \\text{walked}"}</M>,{" "}
          <M>{"\\text{run} \\to \\text{ran}"}</M>.
        </li>
        <li>
          <strong>Capitals:</strong> <M>{"\\text{France} \\to \\text{Paris}"}</M>,{" "}
          <M>{"\\text{Japan} \\to \\text{Tokyo}"}</M>.
        </li>
      </ul>
      <p>
        Switch the figure above to <strong>Analogy</strong> mode and pick A, B, C (try{" "}
        <code>man</code>, <code>woman</code>, <code>king</code>). It draws the offset{" "}
        <M>{"B - A"}</M>, re-applies it at <M>{"C"}</M>, and snaps to the nearest real word — landing
        on <em>queen</em>.
      </p>

      <Callout type="history" title="Where this came from">
        <p>
          The “king − man + woman ≈ queen” trick was popularized by <strong>word2vec</strong>{" "}
          (Mikolov et al., 2013) and <strong>GloVe</strong> (2014) — methods that learned a{" "}
          <em>single static vector per word</em> from co-occurrence statistics, with no neural model
          on top. Modern LLM embeddings are learned <strong>end-to-end</strong>: the table is trained
          jointly with the transformer to minimize next-token loss. Nobody hand-designs a “gender
          axis” — these directions <em>emerge</em> because they help the model predict text. The
          geometry is a discovered byproduct, not a goal.
        </p>
      </Callout>

      <h2>Static vs contextual embeddings</h2>
      <p>
        The embedding table has one fatal limitation, and naming it sets up the entire rest of the
        book. The lookup is <strong>static</strong>: token <code>bank</code> always retrieves the{" "}
        <em>same</em> row, whether the sentence is “river <code>bank</code>” or “savings{" "}
        <code>bank</code>.” A single vector cannot be both “sloping land beside water” and “financial
        institution” at once — so what does it store? A blurry average of all of <code>bank</code>’s
        meanings, the best compromise no-context guess.
      </p>
      <p>
        Resolving the ambiguity is the transformer’s job. As that initial static vector flows up
        through the stack of attention and MLP layers (
        <a href="/chapter/self-attention" className="prose-link">Chapter 8</a> onward), it gets{" "}
        <em>mixed with its neighbours</em> and becomes <strong>contextual</strong>. The two
        <code>bank</code>s start identical at the embedding layer and <strong>diverge</strong> as
        they rise: the river one pulls in “water” and “flow,” the money one pulls in “loan” and
        “account,” until by the top layer they occupy entirely different regions of the space.
      </p>

      <Callout type="key" title="Static in, contextual out">
        <p>
          The embedding table gives each token a fixed, context-free starting vector. The
          transformer’s entire purpose is to turn those static vectors into{" "}
          <strong>context-dependent</strong> ones — representations of a token <em>in this specific
          sentence</em>. Embeddings are where meaning enters; attention is where it gets refined.
        </p>
      </Callout>

      <h2>Weight tying</h2>
      <p>
        At the <em>output</em>, the model does the mirror operation. To predict the next token it maps
        the final hidden vector <M>{"h \\in \\mathbb{R}^{d}"}</M> back to a score (logit) for every
        token in the vocabulary, using an output projection{" "}
        <M>{"W_{out} \\in \\mathbb{R}^{V \\times d}"}</M>:
      </p>
      <MB>{String.raw`\text{logits} \;=\; h \, W_{out}^{\top} \;\in\; \mathbb{R}^{V}, \qquad p \;=\; \text{softmax}(\text{logits}).`}</MB>
      <p>
        Notice <M>{"W_{out}"}</M> and the embedding table <M>{"E"}</M> have the <em>same shape</em>,{" "}
        <M>{"V \\times d"}</M>. A widely used trick — <strong>weight tying</strong> (Press &amp; Wolf,
        2017; Inan et al., 2017) — is to make them the <em>same matrix</em>:{" "}
        <M>{"W_{out} = E"}</M>. Two benefits:
      </p>
      <ul>
        <li>
          <strong>Fewer parameters.</strong> You drop a whole <M>{"V \\times d"}</M> block — for GPT-2
          small that’s ~38M parameters saved, a meaningful fraction of a small model.
        </li>
        <li>
          <strong>A shared input/output space.</strong> The dot product{" "}
          <M>{"h \\cdot \\mathbf{e}_i"}</M> now measures how well the hidden state aligns with token{" "}
          <M>{"i"}</M>’s <em>embedding</em>. Reading and writing a token use one consistent vector, so
          a word that is “close” on the way in is also “close” on the way out.
        </li>
      </ul>

      <Callout type="industry" title="Tied by default — with caveats">
        <p>
          GPT-2 and many open models tie weights; it’s a sensible default, especially when the
          embedding dominates the parameter count. But it isn’t universal: very large models (where
          the embedding is a tiny fraction of total parameters) sometimes <em>untie</em> for a little
          extra quality, and when <M>{"d"}</M> differs between the embedding and the LM head you’ll
          need an extra projection. If you tie, the embedding and head must share the same{" "}
          <M>{"d"}</M> and vocabulary.
        </p>
      </Callout>

      <h2>In code</h2>
      <p>
        All of this is a few lines of PyTorch. <code>nn.Embedding</code> is the lookup table;
        below we also confirm the one-hot equivalence, compute cosine nearest-neighbours, and tie the
        output head to the input embedding.
      </p>

      <CodeBlock language="python" filename="embeddings.py">
{`import torch
import torch.nn as nn
import torch.nn.functional as F

V, d = 50_000, 768                       # vocab size, embedding dim
tok_emb = nn.Embedding(V, d)             # the table E: (V, d), learnable

# --- Lookup: integer IDs -> dense vectors ---
ids = torch.tensor([[2025, 318, 281]])   # (batch=1, T=3) token IDs
x = tok_emb(ids)                         # (1, 3, 768) — one row gathered per ID

# --- The one-hot equivalence (never do this for real; it's the definition) ---
onehot = F.one_hot(ids, V).float()       # (1, 3, V)
x2 = onehot @ tok_emb.weight             # (1, 3, 768) — same result as the lookup
assert torch.allclose(x, x2, atol=1e-5)  # row lookup == one-hot @ E

# --- Cosine similarity & top-k nearest neighbours of a token ---
def nearest(token_id, k=5):
    E = F.normalize(tok_emb.weight, dim=1)   # unit-length rows
    sims = E @ E[token_id]                    # (V,) cosine sim to every token
    sims[token_id] = -1.0                     # exclude the word itself
    vals, idx = sims.topk(k)
    return idx, vals                          # neighbour IDs and their cosines

# --- Weight tying: share the input embedding with the output projection ---
class TinyLM(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)
        self.lm_head = nn.Linear(d, V, bias=False)   # output projection W_out
        self.lm_head.weight = self.tok_emb.weight    # <-- tie: one matrix, two uses

    def forward(self, ids):
        h = self.tok_emb(ids)            # (B, T, d): embed
        # ... the transformer would transform h here (Chapters 7-11) ...
        return self.lm_head(h)           # (B, T, V): logits over the vocabulary`}
      </CodeBlock>

      <p>
        Because <code>lm_head.weight</code> and <code>tok_emb.weight</code> are the{" "}
        <em>same</em> tensor object, a single gradient update improves both the input and output
        roles at once — and the model has one fewer giant matrix to store.
      </p>

      <h2>Where we are, and what’s missing</h2>
      <p>
        Every token is now a learnable vector, and we’ve built the intuition for the space they live
        in: distance is similarity, direction is relationship, and the table is just{" "}
        <M>{"V"}</M> rows trained end-to-end. But two things are still missing before the model can
        actually <em>think</em>. First, these embeddings are <strong>static</strong> — every{" "}
        <code>bank</code> looks identical until context reshapes it. Second, and more immediately:
        the sequence <M>{"X \\in \\mathbb{R}^{T \\times d}"}</M> we built is just an unordered{" "}
        <em>set</em> of vectors. Nothing in it records that token 1 came before token 2 — and as
        we’ll see, attention is <strong>permutation-invariant</strong>, so it can’t tell either.{" "}
        <em>“dog bites man”</em> and <em>“man bites dog”</em> would look the same.
      </p>
      <p>
        So next we inject order. <a href="/chapter/positional-encoding" className="prose-link">
        Chapter 10</a> adds <strong>positional information</strong> — sinusoidal, learned, and rotary
        encodings — so the model knows <em>where</em> each token sits. Then{" "}
        <a href="/chapter/self-attention" className="prose-link">Chapter 8</a> turns these static
        vectors contextual through attention, and the geometry we sketched here comes alive.
      </p>
    </>
  );
}
