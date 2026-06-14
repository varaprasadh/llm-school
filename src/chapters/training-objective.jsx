import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LineChart from "../components/viz/LineChart";
import CrossEntropyViz from "../components/viz/training-objective/CrossEntropyViz";

export default function Chapter() {
  return (
    <>
      <p>
        We have a model that turns a context into a distribution over the next token, and we have a
        mountain of text. What we are still missing is a single number that says{" "}
        <em>how wrong the model currently is</em> — a number we can differentiate and drive toward
        zero. That number is the <strong>loss</strong>, and for language models it is almost always
        the same one: <strong>cross-entropy</strong>, equivalently the{" "}
        <strong>negative log-likelihood</strong> of the true next token. This chapter derives it from
        first principles, shows off its astonishingly clean gradient, and connects it to{" "}
        <strong>perplexity</strong> and <strong>bits-per-token</strong> — the units everyone reports.
        By the end you will know the <em>exact</em> objective that every LLM, from GPT-2 to the
        frontier, minimizes.
      </p>

      <h2>From logits to probabilities</h2>
      <p>
        Recall the end of the forward pass from{" "}
        <a href="/chapter/language-modeling" className="prose-link">Chapter 3</a>. The network reads a
        context <M>{"x_{<t}"}</M> and emits a vector of <strong>logits</strong> — one raw, unbounded
        real number per vocabulary entry:
      </p>
      <MB>{String.raw`z = f_\theta(x_{<t}) \in \mathbb{R}^{|\mathcal{V}|}.`}</MB>
      <p>
        Logits are not probabilities: they can be negative, they need not sum to anything in
        particular. The <strong>softmax</strong> is the function that maps them onto the probability
        simplex — non-negative entries that sum to 1:
      </p>
      <MB>{String.raw`p_v \;=\; \text{softmax}(z)_v \;=\; \frac{e^{\,z_v}}{\sum_{w \in \mathcal{V}} e^{\,z_w}}.`}</MB>
      <p>
        Exponentiating guarantees positivity; dividing by the sum guarantees normalization. The vector{" "}
        <M>{"p = (p_1, \\ldots, p_{|\\mathcal{V}|})"}</M> <em>is</em> the model's predicted
        distribution <M>{"P_\\theta(x_t \\mid x_{<t})"}</M>. Two properties matter for what follows:
        softmax is <em>shift-invariant</em> (adding a constant to every logit leaves <M>{"p"}</M>
        unchanged, which is why stable implementations subtract <M>{"\\max_w z_w"}</M> first), and it
        is smooth, so we can take gradients through it.
      </p>

      <Callout type="note" title="Temperature, for later">
        <p>
          You will sometimes see <M>{"\\text{softmax}(z / \\tau)"}</M> with a{" "}
          <strong>temperature</strong> <M>{"\\tau"}</M>. During <em>training</em> the temperature is
          fixed at <M>{"\\tau = 1"}</M> — it is purely a <em>decoding</em>-time knob for reshaping the
          distribution when we generate (covered in{" "}
          <a href="/chapter/evaluation" className="prose-link">Chapter 21</a>). The loss below always
          uses the bare softmax.
        </p>
      </Callout>

      <h2>Cross-entropy loss</h2>
      <p>
        For a single position, the dataset hands us the <em>correct</em> next token. Call its index{" "}
        <M>{"y"}</M>. The model put probability <M>{"p_y"}</M> on it. A good model makes{" "}
        <M>{"p_y"}</M> large; a perfect model makes it 1. So we want a penalty that is small when{" "}
        <M>{"p_y \\to 1"}</M> and large when <M>{"p_y \\to 0"}</M>. The choice is{" "}
        <M>{"-\\log p_y"}</M>, the <strong>negative log-likelihood</strong> of the true token. The
        full <strong>cross-entropy</strong> form writes the target as a one-hot vector{" "}
        <M>{"y \\in \\{0,1\\}^{|\\mathcal{V}|}"}</M> (a 1 at the true token, 0 elsewhere) and sums:
      </p>
      <MB>{String.raw`\ell(z, y) \;=\; -\sum_{v \in \mathcal{V}} y_v \log p_v \;=\; -\log p_{\,y_{\text{true}}}.`}</MB>
      <p>
        Because the one-hot target zeroes out every term except the true one, the sum{" "}
        <em>collapses</em> to a single <M>{"-\\log p"}</M>. That is the whole loss for one token. Now
        average over an entire corpus (or minibatch) of <M>{"N"}</M> positions:
      </p>
      <MB>{String.raw`\mathcal{L}(\theta) \;=\; -\frac{1}{N} \sum_{t=1}^{N} \log P_\theta\!\left(x_t \mid x_{<t}\right).`}</MB>
      <p>
        This is <strong>the</strong> objective. Every training step computes this scalar and nudges{" "}
        <M>{"\\theta"}</M> to lower it. Notice it is exactly the per-token average that appears inside
        the perplexity formula from{" "}
        <a href="/chapter/language-modeling" className="prose-link">Chapter 3</a> — we will close that
        loop shortly.
      </p>

      <h3>Deriving it from maximum likelihood</h3>
      <p>
        Cross-entropy is not an arbitrary choice; it falls out of asking for the single most
        principled thing. <strong>Maximum likelihood estimation</strong> says: pick the parameters{" "}
        <M>{"\\theta"}</M> that make the observed data as probable as possible. Using the chain-rule
        factorization, the likelihood of the corpus is the product of every next-token probability:
      </p>
      <MB>{String.raw`\theta^\star \;=\; \arg\max_{\theta} \; \prod_{t=1}^{N} P_\theta\!\left(x_t \mid x_{<t}\right).`}</MB>
      <p>
        Products of thousands of sub-1 numbers underflow to zero, and sums are friendlier to
        optimize, so we take the logarithm (monotonic, so the <M>{"\\arg\\max"}</M> is unchanged) and
        flip the sign to turn maximization into minimization:
      </p>
      <MB>{String.raw`\theta^\star \;=\; \arg\min_{\theta} \; \left[-\frac{1}{N}\sum_{t=1}^{N} \log P_\theta\!\left(x_t \mid x_{<t}\right)\right] \;=\; \arg\min_{\theta}\; \mathcal{L}(\theta).`}</MB>
      <p>
        That bracket is precisely the cross-entropy loss. So <em>minimizing cross-entropy is
        identical to maximizing the likelihood of the training text</em> — they are the same
        objective wearing two hats. The <M>{"1/N"}</M> is cosmetic (it makes the number comparable
        across batch sizes) and does not move the optimum.
      </p>

      <Callout type="math" title="Why the negative log?">
        <p>
          The logarithm earns its place three times over. It turns the runaway <em>product</em> of
          probabilities into a stable <em>sum</em>; it converts maximization into the minimization our
          optimizers expect (via the minus sign); and — as the next section shows — it gives the loss
          a precise meaning in <strong>bits</strong>, the currency of information theory. The negative
          sign simply makes the quantity non-negative, since <M>{"\\log p \\le 0"}</M> for{" "}
          <M>{"p \\le 1"}</M>.
        </p>
      </Callout>

      <h2>Why this loss? Surprise, in bits and nats</h2>
      <p>
        There is a second, equally illuminating origin story. Information theory defines the{" "}
        <strong>surprise</strong> (or "surprisal", or "self-information") of an event with probability{" "}
        <M>{"p"}</M> as <M>{"-\\log p"}</M>. A certain event (<M>{"p = 1"}</M>) carries zero surprise;
        a one-in-a-million event carries a lot. Cross-entropy is nothing but the model's{" "}
        <em>average surprise at the actual next token</em>. Training the model is teaching it to
        stop being surprised by real text.
      </p>
      <p>
        The base of the logarithm sets the unit:
      </p>
      <ul>
        <li>
          <strong>Natural log</strong> (<M>{"\\ln"}</M>, base <M>{"e"}</M>) gives the loss in{" "}
          <strong>nats</strong>. This is what PyTorch's <code>cross_entropy</code> returns and what
          you watch tick down during training.
        </li>
        <li>
          <strong>Log base 2</strong> gives the loss in <strong>bits</strong> — the average number of
          yes/no questions needed to pin down the next token. Conversion is a constant factor:{" "}
          <M>{"\\text{bits} = \\text{nats} / \\ln 2 \\approx 1.4427 \\times \\text{nats}"}</M>.
        </li>
      </ul>
      <p>
        Connecting to entropy: if text were drawn from a true distribution <M>{"P^\\star"}</M>, the
        cross-entropy <M>{"H(P^\\star, P_\\theta) = -\\mathbb{E}_{P^\\star}[\\log P_\\theta]"}</M>{" "}
        decomposes as <M>{"H(P^\\star) + D_{\\mathrm{KL}}(P^\\star \\,\\|\\, P_\\theta)"}</M>. The first
        term, the language's intrinsic entropy, is a floor we cannot beat; the second, the
        Kullback–Leibler divergence, is the <em>excess</em> surprise from our model being wrong, and
        it is exactly what training drives toward zero. The model can never reach zero loss — natural
        language has irreducible entropy — but it can close the gap to it.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          LLMs minimize the <strong>average negative log-probability of the true next token</strong>,{" "}
          <M>{"\\mathcal{L} = -\\frac{1}{N}\\sum_t \\log p_{\\text{true}}"}</M>. That single scalar is
          simultaneously the cross-entropy, the negative log-likelihood, and the model's average
          surprise. Lower it and you sharpen the very distribution you will later sample from.
        </p>
      </Callout>

      <h2>The beautiful gradient</h2>
      <p>
        Here is the payoff that makes softmax + cross-entropy the canonical pairing in deep learning.
        We need the gradient of the loss with respect to the <em>logits</em> <M>{"z"}</M> (the thing
        the network actually outputs and backpropagates through). You might brace for a mess of
        exponentials from differentiating the softmax. Instead, almost everything cancels, and what
        remains is breathtakingly simple:
      </p>
      <MB>{String.raw`\frac{\partial \ell}{\partial z_v} \;=\; p_v - y_v \qquad\Longleftrightarrow\qquad \nabla_{z}\, \ell \;=\; \underbrace{\text{softmax}(z)}_{\text{prediction}} \;-\; \underbrace{y}_{\text{one-hot target}}.`}</MB>
      <p>
        Read it component by component. For the <strong>true</strong> token, <M>{"y_v = 1"}</M>, so
        the gradient is <M>{"p_{\\text{true}} - 1"}</M> — negative whenever the model is not fully
        confident, which (after a gradient-descent step subtracts it) <em>pushes that logit up</em>.
        For every <strong>wrong</strong> token, <M>{"y_v = 0"}</M>, so the gradient is <M>{"p_v"}</M> —
        positive in proportion to how much probability the model wrongly spent there, which{" "}
        <em>pushes those logits down</em>. The update is a tug-of-war that drains probability from the
        impostors and pours it onto the truth, with a force proportional to the error.
      </p>

      <Callout type="math" title="Where the cancellation comes from">
        <p>
          The softmax Jacobian is <M>{"\\partial p_i / \\partial z_j = p_i(\\delta_{ij} - p_j)"}</M>.
          The loss is <M>{"\\ell = -\\sum_i y_i \\log p_i"}</M>, so by the chain rule{" "}
          <M>{"\\partial \\ell / \\partial z_j = -\\sum_i (y_i / p_i)\\, p_i(\\delta_{ij} - p_j) = -\\sum_i y_i(\\delta_{ij} - p_j) = -y_j + p_j \\sum_i y_i"}</M>.
          Since the one-hot target sums to one, <M>{"\\sum_i y_i = 1"}</M>, leaving the elegant{" "}
          <M>{"p_j - y_j"}</M>. No surviving exponentials, no division — this numerical grace is a big
          part of why the pairing is universal, and why frameworks fuse <code>log_softmax</code> with
          the NLL into one stable <code>cross_entropy</code> op.
        </p>
      </Callout>

      <p>
        Drag the slider below to <em>feel</em> the loss. A small vocabulary of six words is shown with
        one true next token. The knob sets how much probability the model assigns to that true token;
        the rest spreads over the impostors. Watch <M>{"-\\log p"}</M> stay gentle near{" "}
        <M>{"p = 1"}</M> and detonate as <M>{"p \\to 0"}</M> — and watch perplexity, the effective
        number of choices, track it. The lesson lives in your fingertips: being{" "}
        <strong>confident and correct</strong> costs almost nothing; being{" "}
        <strong>confident and wrong</strong> is catastrophic.
      </p>

      <Figure
        n="12.1"
        title="Cross-entropy, felt as a slider"
        caption="One training example. Loss = −log p(true). As the probability on the correct token falls, the loss climbs along the −log curve and perplexity = exp(loss) = 1/p blows up. The asymmetry — cheap to be right, ruinous to be confidently wrong — is the entire shape of the objective."
      >
        <CrossEntropyViz />
      </Figure>

      <h2>Perplexity: the effective branching factor</h2>
      <p>
        The loss is in nats, which is hard to feel. <strong>Perplexity</strong> (PPL) re-expresses the
        very same quantity on a far more intuitive scale by exponentiating it:
      </p>
      <MB>{String.raw`\text{PPL} \;=\; \exp\!\left(\mathcal{L}\right) \;=\; \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N} \log P_\theta\!\left(x_t \mid x_{<t}\right)\right).`}</MB>
      <p>
        Perplexity is the model's <strong>effective branching factor</strong>: the number of
        equally-likely options it is, on average, choosing between at each step. If a model is
        perfectly certain of every token, <M>{"\\mathcal{L} = 0"}</M> and <M>{"\\text{PPL} = 1"}</M>{" "}
        (one choice — no perplexity at all). A model that guesses uniformly over a vocabulary of size{" "}
        <M>{"V"}</M> has loss <M>{"\\ln V"}</M> and perplexity exactly <M>{"V"}</M> — it is "perplexed"
        among all <M>{"V"}</M> tokens. Real models live in between, and lower is better:
      </p>
      <table>
        <thead>
          <tr>
            <th>Model / regime</th>
            <th>Loss (nats)</th>
            <th>Perplexity</th>
            <th>Reading</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Untrained, vocab 50k</td>
            <td><M>{"\\ln 50000 \\approx 10.8"}</M></td>
            <td><M>{"\\approx 50{,}000"}</M></td>
            <td>uniform random guessing</td>
          </tr>
          <tr>
            <td>GPT-2 on its own eval set</td>
            <td><M>{"\\approx 3.0\\text{–}3.4"}</M></td>
            <td><M>{"\\approx 20\\text{–}30"}</M></td>
            <td>~25 plausible continuations</td>
          </tr>
          <tr>
            <td>Strong modern LLM (held-out web)</td>
            <td><M>{"\\approx 2.0\\text{–}2.7"}</M></td>
            <td><M>{"\\approx 8\\text{–}15"}</M></td>
            <td>much sharper predictions</td>
          </tr>
          <tr>
            <td>Perfect model</td>
            <td><M>{"0"}</M></td>
            <td><M>{"1"}</M></td>
            <td>certainty (unreachable)</td>
          </tr>
        </tbody>
      </table>
      <p>
        Because <M>{"\\text{PPL} = e^{\\mathcal{L}}"}</M> is monotonic in the loss, every gradient step
        that lowers the loss lowers perplexity — they rise and fall together. A loss drop from 3.4 to
        3.0 looks tiny but is a perplexity drop from ~30 to ~20: the model went from juggling thirty
        options to twenty.
      </p>

      <Callout type="pitfall" title="Perplexity is only comparable under a fixed tokenizer">
        <p>
          Perplexity depends on what a "token" <em>is</em>. A model with a coarse vocabulary has fewer,
          higher-information tokens and will show a <em>higher</em> per-token perplexity than a
          fine-grained one on the same text — without being worse. Never compare perplexities across
          different tokenizers or context lengths. For cross-tokenizer comparison, normalize by raw
          text instead, with <strong>bits-per-byte</strong> (below).
        </p>
      </Callout>

      <h2>Bits-per-token, bits-per-byte, and label smoothing</h2>
      <p>
        Two practical footnotes round out the objective.
      </p>
      <p>
        <strong>Bits-per-token and bits-per-byte.</strong> Convert the loss to base 2 and you get{" "}
        <strong>bits-per-token</strong>, <M>{"\\text{bpt} = \\mathcal{L} / \\ln 2"}</M> — the average
        information content of each token under the model. To compare models with{" "}
        <em>different</em> tokenizers fairly, divide the total bits by the number of raw{" "}
        <em>bytes</em> of text rather than tokens, giving <strong>bits-per-byte</strong> (BPB):
      </p>
      <MB>{String.raw`\text{BPB} \;=\; \frac{1}{\ln 2}\cdot\frac{\sum_t -\ln P_\theta(x_t \mid x_{<t})}{\#\,\text{bytes of text}} \;=\; \frac{\text{total nats}}{(\ln 2)\,(\#\text{bytes})}.`}</MB>
      <p>
        BPB is tokenizer-agnostic — it measures how well the model compresses the <em>actual
        characters</em> — which is why scaling-law papers and cross-model leaderboards report it.
        Strong models reach roughly <M>{"0.6\\text{–}0.9"}</M> bits per byte on English text, versus 8
        bits for a naïve byte and about <M>{"1.0\\text{–}1.3"}</M> for classic compressors like gzip.
      </p>
      <p>
        <strong>Label smoothing.</strong> A one-hot target asks the model to drive <M>{"p_{\\text{true}}"}</M>
        all the way to 1, which pushes logits toward <M>{"\\pm\\infty"}</M> and can make the model
        over-confident and poorly calibrated. <strong>Label smoothing</strong> softens the target:
        instead of a hard 1, the true token gets <M>{"1 - \\varepsilon"}</M> and the remaining{" "}
        <M>{"\\varepsilon"}</M> is spread uniformly over the other <M>{"V - 1"}</M> tokens (typically{" "}
        <M>{"\\varepsilon = 0.1"}</M>). The loss becomes a blend of the true-token NLL and a uniform
        term:
      </p>
      <MB>{String.raw`\ell_{\text{LS}} \;=\; (1-\varepsilon)\,\bigl(-\log p_{\text{true}}\bigr) \;-\; \frac{\varepsilon}{V}\sum_{v} \log p_v.`}</MB>
      <p>
        It improves calibration and is standard in machine translation and vision, but most
        large-scale autoregressive LLM pretraining runs use <em>plain</em> cross-entropy{" "}
        (<M>{"\\varepsilon = 0"}</M>) — the regularization that matters most at that scale comes from
        the sheer breadth of the data. Know the tool; reach for it deliberately.
      </p>

      <Callout type="industry" title="What you actually watch during a run">
        <p>
          In practice you stare at two curves: <strong>training loss</strong> (cross-entropy in nats
          on the current batch) and <strong>validation loss</strong> (the same on held-out text). A
          healthy run shows both gliding down and tracking each other; a growing gap means
          overfitting; a flat or rising train loss means your learning rate, data, or numerics are
          broken. Many teams plot the exponential of these as perplexity for intuition, and report
          BPB for the paper.
        </p>
      </Callout>

      <p>
        The figure below sketches a typical pretraining loss curve and its perplexity twin, so you can
        recognize a healthy run at a glance.
      </p>

      <Figure
        n="12.2"
        title="A healthy loss curve and its perplexity twin"
        caption="Cross-entropy (nats, left axis intuition) falls fast then flattens toward the language's entropy floor; perplexity = exp(loss) is the same story on a branching-factor scale. Validation tracks training — the sign of a run that is learning, not memorizing. Values illustrative."
      >
        <LineChart
          height={300}
          xLabel="training step"
          yLabel="value"
          xScale="log"
          series={[
            {
              label: "train loss (nats)",
              color: "#22d3ee",
              points: [
                [10, 9.6], [50, 7.1], [200, 5.2], [800, 3.9], [2000, 3.3],
                [6000, 2.9], [15000, 2.6], [40000, 2.45], [100000, 2.38],
              ],
            },
            {
              label: "val loss (nats)",
              color: "#fb7185",
              dashed: true,
              points: [
                [10, 9.7], [50, 7.3], [200, 5.4], [800, 4.1], [2000, 3.5],
                [6000, 3.1], [15000, 2.8], [40000, 2.66], [100000, 2.6],
              ],
            },
            {
              label: "perplexity = exp(loss)",
              color: "#a855f7",
              points: [
                [10, Math.exp(9.6)], [50, Math.exp(7.1)], [200, Math.exp(5.2)],
                [800, Math.exp(3.9)], [2000, Math.exp(3.3)], [6000, Math.exp(2.9)],
                [15000, Math.exp(2.6)], [40000, Math.exp(2.45)], [100000, Math.exp(2.38)],
              ],
            },
          ]}
          fmtX={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0))}
          fmtY={(v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))}
        />
      </Figure>

      <h2>The loss in code</h2>
      <p>
        Here is the objective exactly as you will write it. Two versions: the one-liner everyone uses,{" "}
        <code>F.cross_entropy</code>, and a manual <code>softmax → gather → −log → mean</code> that
        reproduces it number-for-number. Seeing them agree demystifies the library call — it is doing
        precisely the math above, just fused and numerically stabilized.
      </p>

      <CodeBlock language="python" filename="loss.py" highlight={[10, 24]}>
{`import torch
import torch.nn.functional as F

# A toy batch: B=2 positions, V=5-word vocabulary.
# logits are the raw network outputs (pre-softmax), one row per position.
logits  = torch.tensor([[2.0, 0.5, 1.0, -1.0, 0.2],
                         [0.1, 3.0, 0.4,  0.0, 1.2]])   # (B, V)
targets = torch.tensor([0, 1])                          # true next-token ids, one per row

# --- the one-liner everyone actually uses -----------------------------------
# F.cross_entropy fuses log_softmax + negative-log-likelihood, numerically stable,
# and averages over the batch by default (reduction='mean').
loss = F.cross_entropy(logits, targets)
print(loss.item())            # e.g. 0.4585  (nats per token)

# --- the same thing, by hand, to prove there is no magic --------------------
probs    = F.softmax(logits, dim=-1)                    # (B, V), rows sum to 1
true_p   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # p assigned to the true token
manual   = -torch.log(true_p).mean()                    # -(1/N) * sum log p(true)
print(manual.item())          # 0.4585  -> identical

assert torch.allclose(loss, manual, atol=1e-6)

# --- perplexity is just the exponentiated loss ------------------------------
ppl = loss.exp()
print(ppl.item())             # e.g. 1.5817  (effective branching factor)

# In a real training step, 'logits' has shape (B, T, V) for B sequences of length T.
# Flatten the batch and time dims so every position is one classification example:
#   loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
# Use ignore_index=-100 to skip padding / prompt tokens you do not want to score.`}
      </CodeBlock>

      <Callout type="warning" title="Feed logits, not probabilities">
        <p>
          <code>F.cross_entropy</code> expects <strong>raw logits</strong> and applies{" "}
          <code>log_softmax</code> internally. A classic bug is to softmax first and pass{" "}
          <em>probabilities</em> in — the loss is then silently wrong (a double softmax) and training
          stalls. If you have already softmaxed, use <code>F.nll_loss(torch.log(probs), targets)</code>{" "}
          instead. When in doubt, pass logits.
        </p>
      </Callout>

      <h2>Bridge: from objective to optimization</h2>
      <p>
        We now hold the complete objective of pretraining. The model emits logits; softmax turns them
        into a distribution; cross-entropy — the negative log-likelihood of the true next token,
        averaged over the corpus — measures the average surprise; perplexity and bits-per-byte
        re-express that same number as a branching factor and a compression rate; and the gradient
        with respect to the logits is the wonderfully clean <M>{"\\text{softmax}(z) - y"}</M>.
      </p>
      <p>
        One question remains: knowing the loss and its gradient, <em>how do we actually take the
        steps</em> that lower it across hundreds of billions of parameters and tokens? That is the
        craft of <strong>optimization</strong> — the optimizers, learning rates, and schedules that
        turn a gradient into a trained model — which we take up next in{" "}
        <a href="/chapter/optimization" className="prose-link">Chapter 13</a>.
      </p>
    </>
  );
}
