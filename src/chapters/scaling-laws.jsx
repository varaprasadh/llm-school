import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LineChart from "../components/viz/LineChart";
import ChinchillaAllocator from "../components/viz/scaling-laws/ChinchillaAllocator";

// ---- Figure 17.1 data: reducible test loss as a power law in compute --------
// We plot the REDUCIBLE loss  L(C) - E  against compute C on log-log axes, where
// it is a straight descending line:  (L - E) = k * C^(-alpha_C).  Using a
// Chinchilla-flavoured exponent alpha_C ≈ 0.072 and a constant chosen so the
// curve passes through realistic compute-optimal losses. Generated, not faked.
const ALPHA_C = 0.072;
const K_C = 15.23; // sets the height of the line (reducible loss at C=1)
const lossCurve = [];
for (let e = 18; e <= 26; e += 0.25) {
  const C = Math.pow(10, e);
  lossCurve.push([C, K_C * Math.pow(C, -ALPHA_C)]);
}

// Real-ish training-compute landmarks (order-of-magnitude, in FLOPs).
const MODELS = [
  { C: 1.5e21, label: "GPT-2 (1.5B)", color: "#f59e0b" },
  { C: 3.1e23, label: "GPT-3 (175B)", color: "#fb7185" },
  { C: 5.8e23, label: "Chinchilla (70B)", color: "#34d399" },
];

function flopsTick(v) {
  const e = Math.round(Math.log10(v));
  return `10^${e}`;
}

export default function Chapter() {
  return (
    <>
      <p>
        Most of engineering is unpredictable: you build the thing, then find out if it works. Large
        language models are the rare exception. Before spending ten million dollars and three months
        of a GPU cluster, you can predict — to within a few percent — how good the resulting model
        will be. That predictability is not a lucky accident; it is an empirical law as clean as
        anything in physics, and it is the single fact that turned LLMs from a research curiosity into
        a multi-billion-dollar industry. This chapter is about those <strong>scaling laws</strong>:
        what they say, why <M>{"C \\approx 6ND"}</M>, and how to spend a fixed compute budget so you
        get the best model for the money.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          Test loss falls as a <em>power law</em> in compute, data, and parameters — a straight line
          on a log-log plot. Because the curve is smooth and extrapolable, you can run a few small
          experiments, fit the line, and forecast the loss of a model 1000× larger before you train
          it. Spending the budget well then reduces to one optimization: pick the model size{" "}
          <M>{"N"}</M> and token count <M>{"D"}</M> that sit at the bottom of the curve.
        </p>
      </Callout>

      <h2>Loss is predictable</h2>
      <p>
        Take a transformer, train it well, and measure its test loss (the cross-entropy from{" "}
        <a href="/chapter/training-objective" className="prose-link">
          Chapter 12
        </a>{" "}
        — average negative log-likelihood per token). Now vary one ingredient at a time and re-train.
        What you find, over <em>seven orders of magnitude</em>, is that the loss obeys a power law in
        each of the three resources, as long as the other two aren't the bottleneck:
      </p>
      <MB>{String.raw`L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}`}</MB>
      <p>
        Here <M>{"N"}</M> is the number of (non-embedding) parameters, <M>{"D"}</M> the number of
        training tokens, and <M>{"C"}</M> the total training compute in FLOPs. The exponents are
        small positive numbers — in the original measurements{" "}
        <M>{"\\alpha_N \\approx 0.076"}</M>, <M>{"\\alpha_D \\approx 0.095"}</M>,{" "}
        <M>{"\\alpha_C \\approx 0.050"}</M> — which is exactly why scaling is a long, patient grind:
        to <em>halve</em> the reducible loss you must scale the resource by{" "}
        <M>{"2^{1/\\alpha}"}</M>, which for <M>{"\\alpha \\approx 0.08"}</M> is a factor of roughly{" "}
        <M>{"5000\\times"}</M>.
      </p>
      <p>
        Take the logarithm of a power law and it becomes a straight line:{" "}
        <M>{"\\log L = \\alpha\\,\\log N_c - \\alpha\\,\\log N"}</M>. That is why every scaling-laws
        paper is a gallery of straight lines on log-log axes. Figure 17.1 shows the compute version:
        the reducible test loss sliding down a ruler-straight line as you pour in more FLOPs, with a
        few real model scales marked for orientation.
      </p>

      <Figure
        n="17.1"
        title="Test loss is a power law in compute"
        caption="Reducible loss (total loss minus the irreducible entropy floor of language) versus training compute, on log-log axes. A power law L ∝ C^(−α) is a straight line here. Hover to read values; the markers show where a few well-known training runs sit. Doubling-down on compute buys a predictable, but shrinking, slice of loss."
      >
        <LineChart
          height={340}
          xScale="log"
          yScale="log"
          xLabel="training compute  C  (FLOPs, log scale)"
          yLabel="reducible loss  L − E  (log)"
          series={[{ label: "L(C) ∝ C^(−0.072)", color: "#22d3ee", points: lossCurve }]}
          annotations={MODELS.map((m) => ({ x: m.C, label: m.label, color: m.color }))}
          fmtX={flopsTick}
          fmtY={(v) => v.toFixed(2)}
        />
      </Figure>

      <Callout type="math" title="Why subtract a floor?">
        <p>
          Language has an <em>irreducible</em> entropy: even a perfect model can't predict the next
          token with certainty, so loss can never reach zero. A more honest fit adds a constant{" "}
          <M>{"E"}</M>, the floor: <M>{"L(C) = E + k\\,C^{-\\alpha_C}"}</M>. The{" "}
          <em>reducible</em> part <M>{"L - E"}</M> is the true power law and the straight line you can
          extrapolate. The total loss <M>{"L"}</M> bends and flattens toward <M>{"E"}</M> — a first
          hint that scaling has a ceiling.
        </p>
      </Callout>

      <h2>The compute budget: <M>{"C \\approx 6ND"}</M></h2>
      <p>
        Compute is the currency you actually spend, so we need to relate it to <M>{"N"}</M> and{" "}
        <M>{"D"}</M>. The estimate is beautifully simple. A dense transformer with <M>{"N"}</M>{" "}
        parameters does, to leading order, two floating-point operations per parameter for every token
        it processes in the forward pass (each weight participates in one multiply and one add — a
        multiply-accumulate). The backward pass costs about twice the forward pass, because for each
        weight you must compute <em>two</em> gradients: one with respect to the layer's input (to keep
        backpropagating) and one with respect to the weight itself. That is the famous factor of six:
      </p>
      <MB>{String.raw`C \;\approx\; \underbrace{2ND}_{\text{forward}} \;+\; \underbrace{4ND}_{\text{backward}} \;=\; 6ND \quad \text{FLOPs}`}</MB>
      <p>
        Two for the forward pass, four for the backward pass, per parameter per token. Multiply by{" "}
        <M>{"N"}</M> parameters and <M>{"D"}</M> tokens and you have the cost of the whole training
        run. It is an approximation — it ignores attention's <M>{"O(n^2)"}</M> term (small until
        context gets very long), layer norm, the softmax, and activation recomputation — but it is
        astonishingly accurate in practice, usually within 10–20%.
      </p>

      <Callout type="industry" title="A worked example">
        <p>
          GPT-3 has <M>{"N \\approx 175 \\times 10^{9}"}</M> parameters and was trained on{" "}
          <M>{"D \\approx 300 \\times 10^{9}"}</M> tokens. So{" "}
          <M>{"C \\approx 6 \\times 175\\text{e}9 \\times 300\\text{e}9 \\approx 3.15 \\times 10^{23}"}</M>{" "}
          FLOPs. On an A100 delivering ~150 teraFLOP/s of <em>useful</em> throughput, one GPU would
          take <M>{"3.15\\text{e}23 / 1.5\\text{e}14 \\approx 2.1 \\times 10^{9}"}</M> seconds — about{" "}
          <strong>66 years</strong>. Split across 1024 GPUs at, say, 40% utilization, that becomes a
          few weeks. The factor of six is how you turn a model spec into a cluster reservation.
        </p>
      </Callout>

      <h2>Compute-optimal training</h2>
      <p>
        Here is the central tension. For a <em>fixed</em> compute budget <M>{"C"}</M>, the constraint{" "}
        <M>{"C \\approx 6ND"}</M> means <M>{"N"}</M> and <M>{"D"}</M> trade off:{" "}
        <M>{"D = C / (6N)"}</M>. Spend the budget on a <em>bigger</em> model and you can afford{" "}
        <em>fewer</em> tokens; spend it on <em>more data</em> and the model must be{" "}
        <em>smaller</em>. Both extremes waste money:
      </p>
      <ul>
        <li>
          <strong>Model too big</strong> (<M>{"N"}</M> large, <M>{"D"}</M> small): the model is{" "}
          <em>under-trained</em>. It has the capacity to learn but never sees enough tokens to fill
          that capacity — you paid for parameters you couldn't feed.
        </li>
        <li>
          <strong>Model too small</strong> (<M>{"N"}</M> small, <M>{"D"}</M> large): the model is{" "}
          <em>over-trained</em>. It saturates — extra tokens stop helping because the model lacks the
          capacity to absorb them — and you burned compute pushing data through a model that was full.
        </li>
      </ul>
      <p>
        Somewhere between those failures is a sweet spot: the <strong>compute-optimal</strong>{" "}
        <M>{"(N^\\star, D^\\star)"}</M> that minimizes loss for the budget. Plot loss against{" "}
        <M>{"N"}</M> while holding <M>{"C"}</M> fixed (so <M>{"D"}</M> slides the other way) and you
        get a <strong>U-shaped curve</strong>: a clear bottom flanked by the two wastelands. The
        allocator below lets you feel it. Pick a budget, read off the optimal split, then deliberately
        choose a too-big or too-small model and watch the loss climb.
      </p>

      <Figure
        n="17.2"
        title="The compute allocator"
        caption="For a fixed budget, loss vs. model size is a U. The green line marks the compute-optimal point (~20 tokens per parameter); the purple dot is your choice. Slide the budget and the whole U glides — but its bottom always sits at the same token-to-parameter ratio. That budget-independence is the Chinchilla result."
      >
        <ChinchillaAllocator />
      </Figure>

      <Callout type="math" title="The optimum, in one line of calculus">
        <p>
          Minimize <M>{"L(N, D) = E + A/N^{\\alpha} + B/D^{\\beta}"}</M> subject to{" "}
          <M>{"6ND = C"}</M>. Substituting <M>{"D = C/(6N)"}</M> and setting{" "}
          <M>{"dL/dN = 0"}</M> gives a power-law solution{" "}
          <M>{"N^\\star \\propto C^{a}"}</M>, <M>{"D^\\star \\propto C^{b}"}</M> with exponents{" "}
          <M>{"a, b"}</M> that depend only on <M>{"\\alpha, \\beta"}</M>. The empirical finding is
          that <M>{"a \\approx b \\approx 0.5"}</M>: both grow as the <em>square root</em> of the
          budget, so their <em>ratio</em> <M>{"D^\\star/N^\\star"}</M> stays constant. That constant
          is about 20.
        </p>
      </Callout>

      <h2>The Chinchilla result</h2>
      <p>
        The story has two acts. In 2020, Kaplan and colleagues at OpenAI published the first careful
        scaling laws and concluded something specific: given more compute, you should make the{" "}
        <em>model</em> much bigger and increase the data only modestly. In their compute-optimal
        analysis, parameters scaled roughly as <M>{"N \\propto C^{0.73}"}</M> and data as only{" "}
        <M>{"D \\propto C^{0.27}"}</M>. The whole field took this to heart, and a parade of enormous,
        relatively data-starved models followed: GPT-3 (175B params on 300B tokens), Gopher (280B on
        300B), Megatron-Turing NLG (530B). The implicit rule was <em>“make it bigger.”</em>
      </p>
      <p>
        In 2022, Hoffmann and colleagues at DeepMind revisited the question with a more thorough
        experimental design — over 400 models, three independent estimation methods, and crucially a{" "}
        <em>separate learning-rate schedule tuned to each token budget</em> (a methodological flaw in
        the earlier work had biased it toward big models). Their conclusion overturned the consensus:{" "}
        <M>{"N"}</M> and <M>{"D"}</M> should scale <strong>roughly equally</strong>,{" "}
        <M>{"N^\\star \\propto C^{0.5}"}</M> and <M>{"D^\\star \\propto C^{0.5}"}</M>. The
        compute-optimal ratio is about <strong>20 training tokens per parameter</strong>.
      </p>
      <p>
        To prove it, they trained <strong>Chinchilla</strong>: 70B parameters on 1.4 trillion tokens
        — four times smaller than their own 280B Gopher, but trained on more than four times the data,
        at the <em>same</em> compute budget. Chinchilla beat Gopher across the board. The lesson
        landed hard: most of the giant models of 2020–2021 were badly <em>under-trained</em>, sitting
        on the wrong (too-big) wall of the U.
      </p>

      <Callout type="history" title="Kaplan 2020 vs. Chinchilla 2022">
        <p>
          Both papers are correct power-law science; they disagreed on the <em>exponents</em> because
          of methodology. Kaplan et al. used a fixed learning-rate schedule and a fixed (large) number
          of steps, which under-counts how much a small model improves when its schedule is matched to
          its data — systematically favoring bigger models. Chinchilla’s per-budget tuning removed
          that bias and revealed the near-equal scaling. The phrase{" "}
          <strong>“Chinchilla-optimal”</strong> — about 20 tokens per parameter — is now the default
          mental model. Be precise when you cite it: <em>Kaplan said big; Chinchilla said balanced.</em>
        </p>
      </Callout>

      <p>
        The fit Chinchilla published is worth memorizing as the canonical form of a loss surface:
      </p>
      <MB>{String.raw`L(N, D) \;=\; \underbrace{1.69}_{E,\ \text{irreducible}} \;+\; \frac{406.4}{N^{0.34}} \;+\; \frac{410.7}{D^{0.28}}`}</MB>
      <p>
        Three terms: a floor you can never cross, a parameter term that shrinks as you add capacity,
        and a data term that shrinks as you add tokens. Minimizing it under <M>{"6ND = C"}</M> is
        exactly the curve in Figure 17.2.
      </p>

      <h2>Why people still “over-train” small models</h2>
      <p>
        If Chinchilla is optimal, why did Meta train LLaMA-7B on <strong>one trillion</strong> tokens
        (about 140 tokens per parameter, 7× past the Chinchilla ratio), and LLaMA-3-8B on{" "}
        <strong>fifteen trillion</strong>? Because Chinchilla optimizes the wrong thing for a model
        you intend to <em>deploy</em>. It minimizes <em>training</em> compute. But a model that ships
        is run billions of times, and <strong>inference cost scales with <M>{"N"}</M></strong>, not
        with how long you trained it.
      </p>
      <p>
        The trade is this: a <em>smaller</em> model trained on <em>extra</em> data reaches the same
        loss as a larger Chinchilla-optimal model, but is permanently cheaper to serve — fewer
        parameters means less memory, lower latency, and fewer FLOPs per generated token, forever.
        You pay a one-time premium in training compute to save on every single inference call. When a
        model will answer trillions of queries, that one-time premium is trivial.
      </p>

      <Callout type="industry" title="Training once vs. serving forever">
        <p>
          Chinchilla answers <em>“cheapest way to reach loss <M>{"L"}</M> if I only pay to train.”</em>{" "}
          Production answers <em>“cheapest way to reach loss <M>{"L"}</M> if I also pay to serve.”</em>{" "}
          The second pushes you toward smaller <M>{"N"}</M> and larger <M>{"D"}</M> — the right wall of
          the U, on purpose. LLaMA, Mistral, Qwen, and Gemma all deliberately over-train. The
          allocator’s “too small” region isn’t a mistake to avoid; it’s where most <em>shipped</em>{" "}
          models live, traded against the inference bill the U-curve doesn’t show.
        </p>
      </Callout>

      <h2>Emergent abilities</h2>
      <p>
        Loss is smooth and predictable. <em>Capabilities</em> sometimes are not. As models scale, some
        skills — multi-step arithmetic, following a chain of logic, answering certain benchmark
        questions — appear to switch on <strong>abruptly</strong>: near-zero accuracy across a range
        of sizes, then a sharp jump at some scale. The 2022 paper that popularized the term called
        these <strong>emergent abilities</strong>: capabilities <em>“not present in smaller models but
        present in larger ones,”</em> seemingly unpredictable from the loss curve alone.
      </p>
      <p>
        It is a genuinely exciting and slightly unnerving idea — that scale alone can conjure new
        behaviors. But you should hold it with care, because a chunk of apparent emergence is a{" "}
        <em>measurement artifact</em>.
      </p>

      <Callout type="pitfall" title="Emergence can be an artifact of the metric">
        <p>
          Many “emergent” jumps were measured with <strong>discontinuous metrics</strong> — exact-match
          accuracy on a multi-step task, where you score 1 only if <em>every</em> step is right.
          Under such a metric, a model whose <em>per-token</em> loss is improving smoothly will show
          near-zero accuracy until the per-step probability crosses a threshold, then leap. A 2023
          paper showed that when you re-measure the same models with a <em>smooth</em> metric (token
          edit distance, Brier score, or just the loss), the cliff dissolves into a gentle, continuous
          slope. The capability was improving all along; the harsh metric hid it. <em>Emergence is
          sometimes real, but “sharp jump” often means “sharp metric.”</em> Always ask how a claimed
          emergent ability was scored.
        </p>
      </Callout>

      <p>
        The honest summary: per-token loss scales predictably; <em>downstream task scores</em> are
        derived from loss through a non-linear, sometimes thresholded, lens. Some discontinuities are
        artifacts of that lens; some appear to be real phase-transition-like effects that researchers
        are still characterizing. Treat dramatic “it suddenly could do X” claims as hypotheses to
        check, not laws.
      </p>

      <h2>Limits of scaling</h2>
      <p>
        Scaling laws are powerful but they are not infinite, and three forces bend the curve:
      </p>
      <ul>
        <li>
          <strong>The data wall.</strong> Chinchilla-optimal training at frontier scale needs tens of
          trillions of high-quality tokens, and the supply of good public text is finite. Estimates
          put the exhaustion of high-quality human-written text somewhere in the late 2020s. You can’t
          maintain <M>{"D \\propto C^{0.5}"}</M> forever if <M>{"D"}</M> runs out — hence the rush
          toward synthetic data, multi-epoch training, and squeezing more from each token.
        </li>
        <li>
          <strong>Diminishing returns.</strong> The exponents are small, so every further halving of
          loss costs exponentially more compute, and the total loss is bending toward its irreducible
          floor <M>{"E"}</M>. The next order of magnitude of scale buys less than the last.
        </li>
        <li>
          <strong>Loss is a proxy, not the goal.</strong> Lower next-token loss correlates with “more
          capable,” but it is not the same as “more helpful, honest, and safe.” A base model with
          superb loss can still be useless as an assistant and confidently wrong.
        </li>
      </ul>
      <p>
        This is why the frontier has shifted from <em>just scale the pretraining</em> toward{" "}
        <strong>data quality</strong> (better filtering and deduplication beats raw token count — see{" "}
        <a href="/chapter/data-curation" className="prose-link">
          Chapter 4
        </a>
        ) and toward <strong>post-training</strong>: the alignment and reasoning techniques that
        extract far more usefulness per pretraining-FLOP than another turn of the scaling crank.
      </p>

      <p>
        Here is the planning math in code — the same two functions every lab uses to turn a budget
        into a model spec, and a spec back into a FLOP estimate.
      </p>

      <CodeBlock language="python" filename="compute_optimal.py">
{`def compute_optimal(C, tokens_per_param=20.0):
    """Chinchilla-optimal (N, D) for a training budget C (FLOPs).

    Constraint:   C = 6 * N * D          (forward + backward FLOPs)
    Heuristic:    D / N = tokens_per_param  (~20, the Chinchilla ratio)

    Solve the two equations together:
        C = 6 * N * (r * N) = 6 * r * N**2   =>   N = sqrt(C / (6 * r))
    """
    r = tokens_per_param
    N = (C / (6.0 * r)) ** 0.5     # optimal parameter count
    D = r * N                      # optimal token count  (so 6*N*D == C)
    return N, D


def training_flops(N, D):
    """Estimate FLOPs for a dense-transformer training run.

    ~2 FLOPs/param/token forward + ~4 backward = 6 * N * D.
    Ignores attention's O(seq_len^2) term, which is minor until
    context length gets very large.
    """
    return 6.0 * N * D


if __name__ == "__main__":
    # Budget the size of GPT-3's training run.
    C = 3.1e23
    N, D = compute_optimal(C)
    print(f"budget        : {C:.2e} FLOPs")
    print(f"optimal params: {N/1e9:6.1f} B")
    print(f"optimal tokens: {D/1e12:6.2f} T")
    print(f"check  6*N*D  : {training_flops(N, D):.2e} FLOPs")   # ~= C
    # -> ~51 B params on ~1.0 T tokens: far smaller, far more data
    #    than the actual 175 B-on-300 B-token GPT-3, which was
    #    badly under-trained by Chinchilla's standard.`}
      </CodeBlock>

      <Callout type="tip" title="A back-of-the-envelope you'll use constantly">
        <p>
          Three numbers unlock most capacity planning: <M>{"C = 6ND"}</M> (budget),{" "}
          <M>{"D/N \\approx 20"}</M> (Chinchilla split), and FLOP/s × seconds × utilization (your
          cluster’s delivered compute). Set the cluster compute equal to <M>{"6ND"}</M> and you can
          solve for the largest compute-optimal model your hardware and timeline allow — before
          writing a line of training code.
        </p>
      </Callout>

      <h2>Bridge to post-training</h2>
      <p>
        Scaling laws give you a <em>base model</em>: a next-token predictor whose loss you forecast and
        whose size you chose deliberately, balancing the cost of training against the cost of serving.
        But a base model is raw. It completes text; it does not yet follow instructions, refuse harmful
        requests, or hold a conversation. The remaining usefulness — the difference between a smart
        autocomplete and an assistant you’d actually ship — comes not from more pretraining FLOPs but
        from <strong>post-training</strong>. In{" "}
        <a href="/chapter/supervised-finetuning" className="prose-link">
          Part V
        </a>{" "}
        we turn this carefully-scaled predictor into something helpful: supervised fine-tuning,
        reward modeling, and reinforcement learning from human feedback. The scaling is done; the
        alignment begins.
      </p>
    </>
  );
}
