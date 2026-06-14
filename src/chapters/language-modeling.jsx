import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import NextTokenDemo from "../components/viz/NextTokenDemo";
import ChainRule from "../components/viz/language-modeling/ChainRule";

export default function Chapter() {
  return (
    <>
      <p>
        In <a href="/chapter/what-is-an-llm" className="prose-link">Chapter 1</a> we said an LLM
        is “a probability distribution over text” and left it at that. Now we make that sentence
        exact. By the end of this chapter you will know precisely what mathematical object we are
        training, what the “correct answer” is for every training example, and why a single,
        almost embarrassingly simple objective — <em>predict the next token</em> — is enough to
        learn grammar, facts, translation and reasoning. Everything else in this book (attention,
        optimization, alignment) is in service of estimating this one distribution well.
      </p>

      <h2>Language as probability</h2>
      <p>
        A <strong>language model</strong> is a function that assigns a probability to any sequence
        of tokens. Give it the sentence <em>“the cat sat on the mat”</em> and it returns a single
        number — how plausible that string is as a piece of the language it learned. Fluent,
        grammatical, true-sounding text gets high probability; word salad like{" "}
        <em>“mat the cat the sat”</em> gets low probability.
      </p>
      <p>
        Writing the sequence as <M>{"x = (x_1, x_2, \\ldots, x_T)"}</M> — a list of{" "}
        <M>{"T"}</M> tokens drawn from a fixed vocabulary <M>{"\\mathcal{V}"}</M> — the model is a
        distribution
      </p>
      <MB>{String.raw`P_\theta(x_1, x_2, \ldots, x_T)`}</MB>
      <p>
        parameterized by the network’s weights <M>{"\\theta"}</M>. Because it is a probability
        distribution over <em>all</em> possible sequences, it must satisfy the usual rules:
        every <M>{"P_\\theta(x) \\ge 0"}</M>, and summing over every conceivable string gives 1.
        That second constraint is enormous — there are vastly more possible sentences than atoms
        in the universe — which is exactly why we cannot store this as a table and must{" "}
        <em>learn</em> it as a compact function instead.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          A language model is one object that does two jobs. <strong>Scoring:</strong> given a
          sequence, return its probability — useful for ranking, spell-check, and evaluation.{" "}
          <strong>Generating:</strong> sample new sequences from that same distribution — useful
          for writing, answering, and coding. They are not two models. They are two ways of
          querying the <em>same</em> <M>{"P_\\theta"}</M>.
        </p>
      </Callout>

      <p>
        Hold onto that duality. Scoring asks “how likely is this text?”; generating asks “give me
        likely text.” Once you can do the first, the second comes almost for free — and the bridge
        between them is the chain rule.
      </p>

      <h2>The chain rule of language</h2>
      <p>
        A joint distribution over a six-token sentence looks hopeless: it lives in a space of{" "}
        <M>{"|\\mathcal{V}|^{6}"}</M> possibilities. The trick that makes language modeling
        tractable is a piece of basic probability you already know — the{" "}
        <strong>chain rule</strong>, which factorizes any joint distribution into a product of
        conditionals:
      </p>
      <MB>{String.raw`P_\theta(x_1, x_2, \ldots, x_T) \;=\; \prod_{t=1}^{T} P_\theta\!\left(x_t \mid x_1, x_2, \ldots, x_{t-1}\right)`}</MB>
      <p>
        Using the shorthand <M>{"x_{<t} = (x_1, \\ldots, x_{t-1})"}</M> for “everything before
        position <M>{"t"}</M>,” this compresses to:
      </p>
      <MB>{String.raw`P_\theta(x_{1:T}) \;=\; \prod_{t=1}^{T} P_\theta\!\left(x_t \mid x_{<t}\right)`}</MB>
      <p>
        Read the right-hand side aloud: <em>“the probability of token <M>{"x_t"}</M> given all the
        tokens before it,”</em> multiplied together across every position. This is an exact
        identity — no approximation, no assumption. It holds for any distribution over sequences.
        And it transforms one impossible problem (model the joint over all of language) into{" "}
        <M>{"T"}</M> copies of one merely-hard problem (predict the next token given a prefix).
        The first factor <M>{"P_\\theta(x_1)"}</M> has an empty context; in practice we prepend a
        special <code>&lt;bos&gt;</code> (beginning-of-sequence) token so even position 1 has
        something to condition on.
      </p>

      <Callout type="math" title="Why multiply, not add?">
        <p>
          Probabilities of independent events multiply, and the chain rule’s conditionals are
          constructed to be exactly the pieces that compose back into the joint:{" "}
          <M>{"P(A, B) = P(A)\\,P(B \\mid A)"}</M>, applied recursively. Because each factor is at
          most 1, the product <em>shrinks</em> as the sequence grows — a 20-token sentence is far
          less probable than a 3-token one. That is correct and expected; we almost always work
          with the <em>log</em> of this product, turning the runaway multiplication into a stable
          sum: <M>{"\\log P_\\theta(x_{1:T}) = \\sum_t \\log P_\\theta(x_t \\mid x_{<t})"}</M>.
        </p>
      </Callout>

      <p>
        The figure below makes the factorization tangible. Hover any token to spotlight the
        context it conditions on, and flip between a <em>bigram</em> (condition on one previous
        token), a <em>trigram</em> (two), and <em>full history</em> to see how much each
        prediction is allowed to look at. Watch the running product (the sequence probability) and
        the running sum of <M>{"-\\log p"}</M> (the total “surprise”) update as you go.
      </p>

      <Figure
        n="3.1"
        title="The chain rule, factor by factor"
        caption="P(the cat sat on the mat) decomposed into six next-token predictions. More context (trigram → full history) generally raises each factor and lowers the total surprise — the exact advantage a neural model buys by conditioning on the whole prefix. Probabilities shown are illustrative."
      >
        <ChainRule />
      </Figure>

      <h2>The n-gram baseline</h2>
      <p>
        The chain rule is exact but leaves one term unspecified: how do we actually compute{" "}
        <M>{"P_\\theta(x_t \\mid x_{<t})"}</M>? The oldest answer predates neural networks by
        decades. Make a <strong>Markov assumption</strong>: pretend the next token depends only on
        the last <M>{"n-1"}</M> tokens, throwing away the rest of the history.
      </p>
      <MB>{String.raw`P(x_t \mid x_{<t}) \;\approx\; P(x_t \mid x_{t-n+1}, \ldots, x_{t-1})`}</MB>
      <p>
        With <M>{"n=2"}</M> (a <strong>bigram</strong> model) the next token depends only on the
        single previous token; with <M>{"n=3"}</M> (a <strong>trigram</strong>) on the previous
        two. The beauty is that you can <em>estimate</em> these conditionals by pure counting. Walk
        through a giant corpus, tally how often each context is followed by each token, and
        normalize:
      </p>
      <MB>{String.raw`P(x_t \mid x_{t-1}) \;=\; \frac{\text{count}(x_{t-1},\, x_t)}{\text{count}(x_{t-1})} \;=\; \frac{\#\,\text{times the pair appeared}}{\#\,\text{times the context appeared}}`}</MB>
      <p>
        No gradient descent, no GPUs — just a dictionary of counts. For a long time this was the
        state of the art, and n-gram models still power autocomplete in constrained settings. But
        they hit four walls, and understanding <em>why</em> motivates everything that follows.
      </p>
      <ul>
        <li>
          <strong>Sparsity &amp; zero-counts.</strong> Most valid token sequences never appear in
          any finite corpus. If <M>{"\\text{count}(x_{t-1}, x_t) = 0"}</M>, the model declares the
          continuation <em>impossible</em> — probability exactly zero — which makes the whole
          sequence probability zero and the log-probability <M>{"-\\infty"}</M>. A single unseen
          bigram poisons an otherwise perfectly fluent sentence. Practitioners patch this with{" "}
          <em>smoothing</em> (Kneser–Ney, add-one), but it’s a band-aid on a structural wound.
        </li>
        <li>
          <strong>No generalization.</strong> An n-gram model has no notion that{" "}
          <em>“cat”</em> and <em>“dog”</em> are similar. Having seen <em>“the cat sat”</em> a
          thousand times tells it nothing about <em>“the dog sat,”</em> because the strings are
          literally different keys in the table. It memorizes; it cannot transfer.
        </li>
        <li>
          <strong>Exploding table size.</strong> The number of possible contexts is{" "}
          <M>{"|\\mathcal{V}|^{\\,n-1}"}</M>. With a realistic vocabulary of 50,000 tokens, a
          trigram table has <M>{"50{,}000^2 = 2.5 \\times 10^{9}"}</M> contexts; a 5-gram table
          has <M>{"50{,}000^4 \\approx 6 \\times 10^{18}"}</M> — more entries than you could ever
          store, almost all of them empty. Longer context is precisely what helps, and it is
          precisely what n-grams cannot afford.
        </li>
        <li>
          <strong>Hard context limit.</strong> A trigram fundamentally cannot connect{" "}
          <em>“it”</em> back to a <em>“cat”</em> mentioned ten words earlier. The information
          needed is outside the window, gone.
        </li>
      </ul>

      <Callout type="pitfall" title="The zero-frequency catastrophe">
        <p>
          Because the sequence probability is a <em>product</em>, one zero anywhere makes the
          entire thing zero. An n-gram model that has never seen the exact pair{" "}
          <em>“purple hydrogen”</em> assigns probability 0 to any sentence containing it — even if
          every other word is perfect. Neural models avoid this entirely: a softmax over the
          vocabulary gives <em>every</em> token a strictly positive probability, always.
        </p>
      </Callout>

      <p>
        Here is a complete, runnable bigram model: count the pairs in a tiny corpus, normalize to
        conditional probabilities, then sample a sentence by walking the table. It is correct and
        only a few lines — and its limitation is written right into the loop.
      </p>

      <CodeBlock language="python" filename="bigram.py">
{`import random
from collections import Counter, defaultdict

corpus = """the cat sat on the mat
the dog sat on the rug
the cat ran on the mat""".split()

# 1) count every (previous_word, word) pair
counts = defaultdict(Counter)              # counts[w_prev][w] = how many times w followed w_prev
for w_prev, w in zip(corpus, corpus[1:]):
    counts[w_prev][w] += 1

# 2) normalize counts into conditional probabilities P(w | w_prev)
def probs(w_prev):
    total = sum(counts[w_prev].values())
    return {w: c / total for w, c in counts[w_prev].items()}

# 3) generate by sampling the next word from P(w | w_prev), one step at a time
def generate(start="the", n=6, seed=0):
    rng = random.Random(seed)              # seeded -> reproducible
    out = [start]
    for _ in range(n - 1):
        dist = probs(out[-1])
        if not dist:                       # unseen context -> dead end (no smoothing here)
            break
        words, weights = zip(*dist.items())
        out.append(rng.choices(words, weights=weights)[0])
    return " ".join(out)

print(generate())                          # e.g. "the cat sat on the mat"
# Limitation: "dog" and "cat" share no information — seeing "cat sat"
# teaches this model nothing about "dog sat". No generalization.`}
      </CodeBlock>

      <h2>Neural language models to the rescue</h2>
      <p>
        Every n-gram weakness traces to the same root: the model treats each token as an opaque,
        unrelated symbol and each context as a distinct lookup key. A <strong>neural language
        model</strong> fixes this by replacing the count table with a parameterized{" "}
        <em>function</em> <M>{"f_\\theta"}</M> that <em>computes</em> the conditional
        distribution:
      </p>
      <MB>{String.raw`P_\theta(x_t \mid x_{<t}) \;=\; f_\theta\!\left(x_{<t}\right)_{x_t}`}</MB>
      <p>
        Two ideas make this dramatically better than counting:
      </p>
      <ul>
        <li>
          <strong>Embeddings.</strong> Each token is mapped to a dense vector. The model learns to
          place <em>“cat”</em> and <em>“dog”</em> near each other in this space, so what it learns
          about one automatically transfers to the other. Similarity becomes geometry — the
          subject of <a href="/chapter/embeddings" className="prose-link">Chapter 6</a>.
        </li>
        <li>
          <strong>Shared parameters.</strong> The same weights <M>{"\\theta"}</M> are reused for
          every context and every position. The model isn’t storing answers; it’s storing a
          compressed <em>procedure</em> for producing them. That is why a network with a few
          billion parameters can model a distribution with astronomically more configurations than
          it has weights — and assign sensible probability to sentences it has never seen.
        </li>
      </ul>
      <p>
        Because <M>{"f_\\theta"}</M> ends in a softmax over the vocabulary, every continuation gets
        strictly positive probability — no zeros, no <M>{"-\\infty"}</M>, no smoothing hacks. And
        because the context is processed by a network (a transformer, in our case), it can attend
        to the entire prefix, not just the last <M>{"n-1"}</M> tokens. The Markov assumption is
        gone.
      </p>

      <Callout type="industry" title="Why this framing won">
        <p>
          The label for position <M>{"t"}</M> is simply the token that actually came next — already
          present in the text, no annotation required. This <em>self-supervised</em> setup is why
          LLMs could train on trillions of tokens scraped from the web: the supervision was free
          and effectively unlimited. A model that just predicts the next token, at scale, ends up
          having to learn syntax, world facts, arithmetic, and code to do it well.
        </p>
      </Callout>

      <h2>The prediction setup</h2>
      <p>
        Let’s nail down the exact computation for one prediction. The model conditions on a
        bounded slice of recent tokens called the <strong>context window</strong> (or block size){" "}
        <M>{"-"}</M> say the last <M>{"T"}</M> positions. Those token IDs flow through the network,
        which emits a vector of <strong>logits</strong>, one real number per vocabulary entry:
      </p>
      <MB>{String.raw`z = f_\theta(x_{<t}) \in \mathbb{R}^{|\mathcal{V}|}`}</MB>
      <p>
        Logits are unnormalized scores — they can be any real number, positive or negative. The{" "}
        <strong>softmax</strong> turns them into a proper probability distribution that sums to 1:
      </p>
      <MB>{String.raw`P_\theta(x_t = v \mid x_{<t}) \;=\; \frac{e^{\,z_v}}{\sum_{w \in \mathcal{V}} e^{\,z_w}}`}</MB>
      <p>
        That single softmax vector <em>is</em> the conditional <M>{"P_\\theta(x_t \\mid x_{<t})"}</M>{" "}
        from the chain rule. The whole pipeline for one step is:
      </p>
      <MB>{String.raw`\underbrace{x_{<t}}_{\text{context}} \;\longrightarrow\; \underbrace{f_\theta}_{\text{network}} \;\longrightarrow\; \underbrace{z}_{\text{logits}} \;\xrightarrow{\;\text{softmax}\;}\; \underbrace{P_\theta(x_t \mid x_{<t})}_{\text{distribution over } \mathcal{V}}`}</MB>
      <p>
        Crucially, <em>every position in a training sequence is its own training example</em>. A
        single document of length <M>{"T"}</M> yields <M>{"T"}</M> next-token predictions at once
        — at position 1 predict <M>{"x_2"}</M>, at position 2 predict <M>{"x_3"}</M>, and so on.
        The causal mask in the transformer (from{" "}
        <a href="/chapter/self-attention" className="prose-link">Chapter 8</a>) guarantees each
        prediction only sees its own past, so all <M>{"T"}</M> predictions can be computed in a
        single parallel forward pass. That parallelism is what makes pretraining feasible at all.
      </p>

      <p>
        The demo below is that loop made interactive. It holds a tiny hand-built distribution, but
        the mechanics are identical to a frontier model: read the distribution over the next token,
        pick one (greedily or by sampling), append it, feed the longer sequence back in. The
        <strong> temperature</strong> knob reshapes the softmax — we return to it shortly.
      </p>

      <Figure
        n="3.2"
        title="Next-token prediction and sampling, live"
        caption="Generation is the chain rule run forward: predict P(x_t | x_<t), choose a token, append, repeat. Greedy always takes the argmax; Sample draws from the distribution. Temperature sharpens (→0) or flattens (→∞) the softmax before sampling."
      >
        <NextTokenDemo />
      </Figure>

      <h2>Teacher forcing</h2>
      <p>
        Here is a subtlety that trips up almost everyone. During <strong>training</strong>, when
        we score position <M>{"t"}</M>, what do we feed as the context <M>{"x_{<t}"}</M> — the
        model’s own previous predictions, or the real tokens from the corpus? The answer is the{" "}
        <em>real</em> tokens, always. This is called <strong>teacher forcing</strong>.
      </p>
      <p>
        The reasoning is both practical and statistical. Practically, feeding the ground-truth
        prefix lets us evaluate all <M>{"T"}</M> positions of a sequence in one parallel pass — if
        each step depended on the model’s own previous output, we’d have to generate token by token
        and training would crawl. Statistically, the chain rule we are fitting{" "}
        <em>conditions on the true history</em> <M>{"x_{<t}"}</M>, so to estimate{" "}
        <M>{"P_\\theta(x_t \\mid x_{<t})"}</M> correctly we must condition on exactly that. The
        “teacher” supplies the correct prefix at every step so an early mistake never derails the
        rest of the sequence during learning.
      </p>
      <p>
        At <strong>inference</strong>, there is no teacher. The model runs{" "}
        <strong>autoregressively</strong>: it feeds its <em>own</em> sampled tokens back in as
        context, because the true continuation is exactly what we’re trying to produce. Contrast
        the two regimes:
      </p>
      <table>
        <thead>
          <tr>
            <th>Aspect</th>
            <th>Training (teacher forcing)</th>
            <th>Inference (autoregressive)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Context fed at step <M>{"t"}</M></td>
            <td>true tokens <M>{"x_{<t}"}</M> from the corpus</td>
            <td>the model’s own generated tokens <M>{"\\hat{x}_{<t}"}</M></td>
          </tr>
          <tr>
            <td>Computation</td>
            <td>all <M>{"T"}</M> positions in parallel (one pass)</td>
            <td>strictly sequential, one token at a time</td>
          </tr>
          <tr>
            <td>Purpose</td>
            <td>compute the loss, update <M>{"\\theta"}</M></td>
            <td>produce new text</td>
          </tr>
        </tbody>
      </table>

      <Callout type="warning" title="Exposure bias">
        <p>
          Because the model is only ever trained on <em>true</em> prefixes but must run on its{" "}
          <em>own</em> prefixes at inference, it can drift into states it never saw during
          training — small errors compound. This gap is called <strong>exposure bias</strong>. In
          practice, large-scale pretraining plus the alignment stages of Part&nbsp;V keep it mild,
          but it is the fundamental reason a model can start a paragraph beautifully and then
          wander off the rails.
        </p>
      </Callout>

      <h2>Sampling vs. scoring (and how we measure it)</h2>
      <p>
        We can now state the scoring/generating duality from the opening precisely, because both
        operations read off the same per-step distributions.
      </p>
      <ul>
        <li>
          <strong>Scoring</strong> a known sequence: run one teacher-forced forward pass, read off{" "}
          <M>{"P_\\theta(x_t \\mid x_{<t})"}</M> at the <em>actual</em> next token for every
          position, and sum the logs. The result, <M>{"\\sum_t \\log P_\\theta(x_t \\mid x_{<t})"}</M>,
          is the sequence log-probability — no randomness involved.
        </li>
        <li>
          <strong>Generating</strong> a new sequence: at each step, <em>draw</em> a token from{" "}
          <M>{"P_\\theta(x_t \\mid x_{<t})"}</M> (optionally reshaped by temperature, top-k, or
          top-p), append it, and continue. You first met these decoding controls — temperature,
          top-k and top-p — in <a href="/chapter/what-is-an-llm" className="prose-link">Chapter 1</a>.
        </li>
      </ul>
      <p>
        How do we judge whether <M>{"P_\\theta"}</M> is any good? With <strong>perplexity</strong>,
        the standard intrinsic metric for language models. It is the exponentiated average
        negative log-likelihood on held-out text,
      </p>
      <MB>{String.raw`\text{Perplexity}(x_{1:T}) \;=\; \exp\!\left(-\frac{1}{T} \sum_{t=1}^{T} \log P_\theta\!\left(x_t \mid x_{<t}\right)\right),`}</MB>
      <p>
        and it has a lovely interpretation: the model’s effective <em>branching factor</em>, i.e.
        how many equally-likely options it is, on average, choosing between at each step. A
        perplexity of 1 means perfect certainty; a perplexity of 50,000 (an untrained model over a
        50k vocabulary) means it is guessing uniformly at random. Lower is better. We derive
        perplexity and its relationship to the cross-entropy loss in full in{" "}
        <a href="/chapter/training-objective" className="prose-link">Chapter 12</a> — for now, just
        know that minimizing the training loss <em>is</em> minimizing perplexity, which{" "}
        <em>is</em> sharpening the distribution you sample from.
      </p>

      <Callout type="note" title="One objective, three views">
        <p>
          The next-token loss, perplexity, and the chain-rule log-probability are the same quantity
          wearing different hats. Training minimizes per-token cross-entropy; perplexity is its
          exponential; and the sum of per-token log-probs is the sequence score. Improve one and
          you improve all three.
        </p>
      </Callout>

      <h2>Bridge: from objective to architecture</h2>
      <p>
        We now have a precise target. A language model is <M>{"P_\\theta(x_{1:T})"}</M>; the chain
        rule factorizes it into next-token predictions <M>{"P_\\theta(x_t \\mid x_{<t})"}</M>; a
        neural network computes each one as a softmax over logits; we fit it by teacher forcing and
        measure it with perplexity; and we generate by sampling the same distribution
        autoregressively. That is the entire objective of pretraining, stated end to end.
      </p>
      <p>
        Three questions remain, and they shape the rest of Part&nbsp;II. <em>What is a token</em>,
        exactly, and how do we choose the vocabulary <M>{"\\mathcal{V}"}</M> — the job of{" "}
        <a href="/chapter/tokenization" className="prose-link">Chapter 5</a>. How do we turn token
        IDs into the dense vectors that let the model generalize — embeddings, in{" "}
        <a href="/chapter/embeddings" className="prose-link">Chapter 6</a>. And what network{" "}
        <M>{"f_\\theta"}</M> is powerful enough to map a long context to a good
        next-token distribution — the transformer, whose engine, self-attention, we open in{" "}
        <a href="/chapter/self-attention" className="prose-link">Chapter 8</a>. The objective is
        settled; now we build the machine that meets it.
      </p>
    </>
  );
}
