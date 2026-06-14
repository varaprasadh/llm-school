import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import BenchmarkBoard from "../components/viz/evaluation/BenchmarkBoard";

export default function Chapter() {
  return (
    <>
      <p>
        You trained a model. Loss went down. Is it <em>good</em>? That question is far harder than it
        sounds, and it is where most teams quietly fool themselves. A model that scores 90% on a
        famous benchmark can be useless in your product; a model that "feels" great in a demo can be
        a regression you can't see. This chapter is about measuring capability honestly — the metrics,
        the benchmarks and <em>what each one actually tests</em>, the clever trick of using one model
        to grade another, and the traps (contamination, leaderboard overfitting, Goodhart's law) that
        turn impressive numbers into lies.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          No single number captures "good." Evaluation is a <em>portfolio</em> of measurements, each
          probing a different capability, each with its own failure modes. Your job is to assemble a
          suite that correlates with the thing you actually care about — and to stay paranoid about
          the gap between a score and real usefulness.
        </p>
      </Callout>

      <h2>Intrinsic vs. extrinsic evaluation</h2>
      <p>
        There are two fundamentally different ways to score a language model.
      </p>
      <p>
        <strong>Intrinsic</strong> evaluation measures the model on its own training objective:
        how well does it predict held-out text? The canonical metric is{" "}
        <strong>perplexity</strong> — the exponentiated average negative log-likelihood per token,{" "}
        <M>{"\\text{PPL} = \\exp\\!\\big(-\\frac{1}{N}\\sum_{i} \\log p(x_i \\mid x_{<i})\\big)"}</M>.
        Intuitively, perplexity is the model's average "branching factor": a perplexity of 12 means
        the model is, on average, as uncertain as if it were choosing uniformly among 12 tokens. Lower
        is better. We derive it fully in{" "}
        <a href="/chapter/training-objective" className="prose-link">
          Chapter 12
        </a>
        ; here we just note its limits.
      </p>
      <p>
        <strong>Extrinsic</strong> evaluation measures the model on a <em>downstream task</em> you
        care about: answering questions, writing code, summarizing documents, following instructions.
        This is what users actually experience, and it is what benchmarks try to capture.
      </p>

      <Callout type="pitfall" title="Perplexity is necessary, not sufficient">
        <p>
          Lower perplexity correlates with better models — but only <em>within</em> the same
          tokenizer and data distribution. You cannot compare perplexity across different tokenizers
          (a model with a bigger vocabulary splits text into fewer, "easier" tokens and looks better
          for free). And a model can have excellent perplexity yet be terrible at instruction
          following, reasoning, or staying truthful. Perplexity measures <em>compression of text</em>,
          not <em>usefulness to a person</em>.
        </p>
      </Callout>

      <h2>The benchmark zoo: what each one measures</h2>
      <p>
        A benchmark is a fixed dataset of tasks with a scoring rule. The crowded modern landscape can
        feel like alphabet soup, so here is the map — grouped by the capability each one is built to
        probe. Knowing <em>what a benchmark tests</em> is more important than memorizing leaderboard
        numbers.
      </p>
      <ul>
        <li>
          <strong>Knowledge &amp; reasoning — MMLU</strong> (Massive Multitask Language
          Understanding): 57 subjects from elementary math to law, medicine and ethics, as
          multiple-choice questions. It is the standard proxy for "how much does this model know and
          can it reason over it." MMLU-Pro is a harder, cleaner successor.
        </li>
        <li>
          <strong>Commonsense — HellaSwag, ARC, WinoGrande</strong>. HellaSwag asks the model to pick
          the plausible ending of a everyday situation (trivial for humans, once hard for models). ARC
          (AI2 Reasoning Challenge) is grade-school science questions; WinoGrande tests pronoun
          resolution requiring world knowledge.
        </li>
        <li>
          <strong>Math reasoning — GSM8K, MATH</strong>. GSM8K is 8.5k grade-school word problems
          needing multi-step arithmetic; MATH is competition-level problems (algebra, geometry,
          number theory). These reward genuine multi-step reasoning, not recall.
        </li>
        <li>
          <strong>Code — HumanEval, MBPP</strong>. The model is given a function signature and
          docstring and must write a correct implementation, checked by running <em>hidden unit
          tests</em>. Scored with <strong>pass@k</strong> (below). This is one of the few benchmarks
          that is automatically and unambiguously gradable.
        </li>
        <li>
          <strong>Truthfulness — TruthfulQA</strong>. Questions engineered to elicit popular human
          misconceptions ("What happens if you crack your knuckles a lot?"). It measures whether a
          model parrots common falsehoods or resists them — a different axis entirely from raw
          knowledge.
        </li>
        <li>
          <strong>Chat quality — MT-Bench, Chatbot Arena</strong>. Open-ended, multi-turn
          conversation. MT-Bench uses a strong model as a judge over 80 curated prompts; Chatbot
          Arena collects millions of <em>human</em> pairwise votes and ranks models by Elo. These
          target the fuzzy thing benchmarks usually miss: is it actually pleasant and helpful to talk
          to?
        </li>
      </ul>

      <Callout type="industry" title="Read the benchmark, not just the score">
        <p>
          Before you trust a number, ask: is it <strong>multiple-choice</strong> (the model only has
          to rank options — easier, and gameable by elimination) or <strong>generative</strong> (it
          must produce the full answer)? Is it <strong>zero-shot</strong> or{" "}
          <strong>5-shot</strong>? Does "MMLU 86%" mean the same protocol both papers used? Tiny
          differences in prompting and answer-extraction can swing scores by several points, which is
          why cross-paper comparisons are so often apples-to-oranges.
        </p>
      </Callout>

      <h2>Few-shot &amp; in-context evaluation</h2>
      <p>
        Base models aren't chatbots — you can't just ask them a question. The trick that unlocked
        benchmark evaluation of raw language models is <strong>few-shot prompting</strong>: you show
        the model a handful of worked examples in the prompt, then a new question, and read off its
        completion. With <M>{"k"}</M> examples it's called <M>{"k"}</M>-shot;{" "}
        <M>{"k=0"}</M> is zero-shot.
      </p>
      <p>
        Because the model learns the task <em>format</em> from the examples (this is "in-context
        learning"), the number and choice of shots materially changes the score. A good eval fixes
        the shots, the ordering, and the exact answer-extraction rule (e.g. "take the first capital
        letter A–D") so the measurement is reproducible. For chat models, you instead use the
        instruction format the model was tuned on. Mismatched formatting is one of the most common
        reasons a model "underperforms" its true ability on a benchmark.
      </p>

      <h2>Scoring code: pass@k</h2>
      <p>
        Code is special: correctness is checkable by execution. For a problem, you sample{" "}
        <M>{"n"}</M> independent completions, run the hidden tests on each, and count how many pass.
        The metric <strong>pass@k</strong> estimates the probability that <em>at least one</em> of{" "}
        <M>{"k"}</M> samples is correct. The naive estimator (generate <M>{"k"}</M>, check if any
        pass) is high-variance, so the HumanEval paper uses an unbiased estimator: generate{" "}
        <M>{"n \\ge k"}</M> samples, observe <M>{"c"}</M> correct ones, and compute
      </p>
      <MB>{String.raw`\text{pass@}k \;=\; \mathbb{E}\Big[\, 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \,\Big]`}</MB>
      <p>
        The fraction <M>{"\\binom{n-c}{k} / \\binom{n}{k}"}</M> is the probability that a random draw
        of <M>{"k"}</M> from the <M>{"n"}</M> samples contains <em>zero</em> correct ones; one minus
        that is the chance of catching at least one. Here's the standard, numerically-stable
        implementation.
      </p>

      <CodeBlock language="python" filename="pass_at_k.py">
{`import numpy as np

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k (Chen et al., 2021).

    n: total samples generated for a problem
    c: number of those samples that passed all tests
    k: the k in pass@k (k <= n)
    """
    if n - c < k:
        # fewer than k failures => every draw of k must contain a pass
        return 1.0
    # P(no correct in a random k-subset) = C(n-c, k) / C(n, k)
    # Computed as a product to avoid huge binomials / overflow.
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

# Aggregate over a benchmark: average pass@k across all problems.
def humaneval_score(results, k=1):
    # results: list of (n_samples, n_correct) per problem
    return float(np.mean([pass_at_k(n, c, k) for n, c in results]))

# Example: 200 samples/problem, this problem got 24 right.
print(round(pass_at_k(200, 24, 1), 3))    # 0.12   (pass@1)
print(round(pass_at_k(200, 24, 10), 3))   # 0.73   (pass@10)`}
      </CodeBlock>

      <p>
        Notice the gap: a model can have a low <M>{"\\text{pass@}1"}</M> (rarely right on the first
        try) but a high <M>{"\\text{pass@}10"}</M> (usually right if you let it try ten times). That
        gap matters: pass@1 reflects a one-shot assistant, while pass@100 reflects what's possible
        with a verifier or test-driven loop on top.
      </p>

      <h2>LLM-as-judge: scaling evaluation with a model</h2>
      <p>
        Generative quality — "is this summary good?", "which of these two answers is better?" — has no
        cheap automatic metric. Human grading is the gold standard but is slow and expensive. The
        modern workaround is <strong>LLM-as-judge</strong>: prompt a strong model (often a frontier
        model) to score or compare outputs. It is fast, cheap, and correlates surprisingly well with
        human preference — when done carefully. The catch is that judges have systematic{" "}
        <em>biases</em> you must defend against:
      </p>
      <ul>
        <li>
          <strong>Position bias</strong>: in a pairwise A-vs-B comparison, judges tend to prefer
          whichever answer is presented <em>first</em> (or sometimes second). Fix: evaluate both
          orderings and average, or randomize and require consistency.
        </li>
        <li>
          <strong>Verbosity bias</strong>: judges reward longer, more elaborate answers even when the
          extra words add nothing. Fix: control for length, or instruct the judge to penalize
          padding.
        </li>
        <li>
          <strong>Self-preference &amp; style bias</strong>: a judge tends to favour outputs that
          match its own style or that it generated. Fix: use a different model family as judge, and
          anchor with a small human-labeled calibration set.
        </li>
      </ul>

      <Callout type="warning" title="A judge is a model, with all that implies">
        <p>
          An LLM judge can be jailbroken, can hallucinate a rubric it wasn't given, and can be wrong
          confidently. Always pin the judge to a <em>specific model version</em> (judges drift when
          the provider updates the model), give it an explicit rubric, ask for a short rationale
          before the score, and periodically check its agreement against human labels. Treat the
          judge's output as a noisy estimate, not ground truth.
        </p>
      </Callout>

      <h2>See it: rankings depend on the metric</h2>
      <p>
        Here is the chapter's central lesson made tangible. Below are five (invented) models scored
        across six benchmarks. Pick which benchmark to rank by and watch the leaderboard reshuffle —
        the code specialist tops HumanEval, the reasoning model tops GSM8K, the well-rounded model
        tops the human Arena, and the "safe" model tops TruthfulQA. <strong>The #1 model changes
        depending on which column you choose.</strong> Then open the score matrix: no single row is
        bright all the way across. Finally, try the LLM-as-judge tab to feel position and verbosity
        bias flip a verdict.
      </p>

      <Figure
        n="21.1"
        title="Benchmark board — no single number is 'good'"
        caption="Sort the leaderboard by different benchmarks and the ranking reorders. The score matrix shows the same data as colour; the LLM-as-judge panel lets you flip the presentation order to expose position and verbosity bias. All models and scores are fictional, chosen to illustrate the structure of the problem."
      >
        <BenchmarkBoard />
      </Figure>

      <Callout type="industry" title="What teams actually do">
        <p>
          Serious teams build a <strong>custom eval suite</strong>: a mix of public benchmarks (for
          comparability), private held-out sets (immune to contamination), and{" "}
          <strong>task-specific evals</strong> drawn from real production traffic. They track every
          number across model versions to catch regressions, gate releases on it, and — crucially —
          weight the evals by what their users care about, not by what's fashionable on the
          leaderboard.
        </p>
      </Callout>

      <h2>Human evaluation, Elo &amp; the Arena</h2>
      <p>
        When automatic metrics run out, you ask people. The most influential format is{" "}
        <strong>pairwise preference</strong>: show a rater two anonymized responses to the same
        prompt and ask which is better. Aggregate thousands of such head-to-head votes into a single
        ranking using an <strong>Elo</strong> rating system (borrowed from chess). Each model has a
        rating <M>{"R"}</M>; the predicted probability that model A beats model B is
      </p>
      <MB>{String.raw`P(A \text{ beats } B) \;=\; \frac{1}{1 + 10^{\,(R_B - R_A)/400}}`}</MB>
      <p>
        After each comparison, the winner's rating goes up and the loser's down, by an amount that
        depends on how surprising the result was. <strong>Chatbot Arena</strong> applies exactly this
        to crowd-sourced human votes across millions of battles, producing the most trusted public
        ranking of chat models — precisely because it measures real human preference on open-ended
        prompts rather than any fixed test set. The cost is that it's slow, expensive, and only
        ranks <em>relative</em> quality.
      </p>

      <h2>How benchmarks lie: contamination, Goodhart &amp; the usefulness gap</h2>
      <p>
        Now the uncomfortable part. Benchmark scores are routinely <em>too high</em>, for reasons
        that have nothing to do with capability.
      </p>
      <p>
        <strong>Contamination</strong> is the big one. Benchmarks are public, on the web — exactly
        where pretraining data is scraped from. If the test questions (or their answers) leaked into
        the training set, the model can <em>memorize</em> them, and its score reflects recall, not
        skill. A model can "ace" MMLU because it literally read the answer key. Detecting
        contamination is hard: you can search for verbatim overlap, test on freshly-written private
        questions, or check whether the model can complete a benchmark item from a partial prompt it
        should never have seen. The only real defense is a held-out set the model has provably never
        encountered.
      </p>

      <Callout type="pitfall" title="Contamination: the silent score-inflator">
        <p>
          Treat any benchmark older than your training data cutoff as <em>potentially compromised</em>
          until proven otherwise. A sudden jump on a popular public benchmark, with no jump on a fresh
          private analog of the same task, is a contamination red flag — not a breakthrough. When in
          doubt, build a new test set; don't trust the leaderboard.
        </p>
      </Callout>

      <p>
        <strong>Goodhart's law</strong>: <em>"When a measure becomes a target, it ceases to be a good
        measure."</em> Once a benchmark becomes the thing everyone optimizes, models start to fit its
        quirks rather than the underlying ability. Teams (sometimes inadvertently) tune
        hyperparameters, prompts, and data mixes to climb a specific leaderboard — <strong>overfitting
        to the benchmark</strong>. The number rises; the real capability it was supposed to track
        does not.
      </p>
      <p>
        Finally, the <strong>usefulness gap</strong>: even an honest, uncontaminated, well-chosen
        benchmark is a narrow proxy. MMLU doesn't measure whether the model follows <em>your</em>
        instructions, stays on-brand, calls tools correctly, handles your domain's jargon, or refuses
        the things you need it to refuse. A two-point benchmark difference between two models tells
        you almost nothing about which will work better in your product. The map is not the territory.
      </p>

      <Callout type="key" title="Evaluate like a scientist, not a fan">
        <p>
          Use multiple benchmarks (one number is never enough). Keep a private held-out set to dodge
          contamination. Anchor automatic metrics to a little human-labeled data. And always close the
          loop with evaluation on <em>your own task</em> — because the only score that ultimately
          matters is whether the model is genuinely useful to the people who will use it.
        </p>
      </Callout>

      <h2>Where this leads</h2>
      <p>
        Measuring quality is half the deployment story; the other half is measuring — and improving —
        speed and cost. A model that scores beautifully but answers in twelve seconds at ten cents a
        request won't survive contact with production. Next we turn to{" "}
        <a href="/chapter/inference-optimization" className="prose-link">
          inference optimization
        </a>{" "}
        — KV caching, batching, quantization and the tricks that make a good model also a{" "}
        <em>fast, cheap</em> one. And once it's deployed, we'll need to keep it healthy and safe,
        which is the subject of{" "}
        <a href="/chapter/monitoring-safety" className="prose-link">
          Chapter 24
        </a>
        .
      </p>
    </>
  );
}
