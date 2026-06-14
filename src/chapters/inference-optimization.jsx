import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LineChart from "../components/viz/LineChart";
import KVCacheViz from "../components/viz/inference-optimization/KVCacheViz";

export default function Chapter() {
  return (
    <>
      <p>
        You have trained a model. It works. Now you have to <em>run</em> it — millions of times a
        day, fast enough that a human waiting on the other end doesn’t give up, cheaply enough that
        the bill doesn’t bankrupt you. Training is a one-time capital cost; inference is the
        recurring one, and for a popular product it dwarfs training within weeks. This chapter is the
        engineering of making generation <strong>fast and cheap</strong> without retraining anything.
      </p>
      <p>
        Almost every trick here exploits one structural fact about autoregressive generation: it is
        deeply, wastefully repetitive. The same keys and values get recomputed, the same weights get
        re-read from memory, the same easy tokens get predicted the hard way. Optimization is the art
        of not doing work twice.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          Generation has two phases with opposite bottlenecks. <strong>Prefill</strong> processes the
          whole prompt at once and is <em>compute-bound</em>. <strong>Decode</strong> emits one token
          at a time and is <em>memory-bandwidth-bound</em>. Nearly every inference optimization is
          really an attack on the decode phase — caching what it would recompute, shrinking what it
          must re-read, or producing more than one token per step.
        </p>
      </Callout>

      <h2>Two phases: prefill vs. decode</h2>
      <p>
        When you send a prompt of <M>{"n"}</M> tokens and ask for a completion, the model does two
        very different things:
      </p>
      <ul>
        <li>
          <strong>Prefill</strong> — the prompt is run through the network <em>in parallel</em>, all{" "}
          <M>{"n"}</M> tokens at once, exactly like a training forward pass. This produces the first
          output token and (crucially) the keys and values for every prompt position. With <M>{"n"}</M>{" "}
          tokens flowing through dense matrix multiplies, the GPU’s arithmetic units are saturated:
          prefill is <strong>compute-bound</strong>. Its latency sets your{" "}
          <strong>time-to-first-token (TTFT)</strong>.
        </li>
        <li>
          <strong>Decode</strong> — thereafter the model generates one token per step. Each step is a
          forward pass over a <em>single</em> new token. The matrix multiplies are now tall-and-skinny
          (batch of one position), so the arithmetic units sit mostly idle while the GPU streams
          billions of weight parameters from memory just to multiply them by one vector. Decode is{" "}
          <strong>memory-bandwidth-bound</strong>. Its per-step latency sets your{" "}
          <strong>time-per-output-token (TPOT)</strong>.
        </li>
      </ul>
      <p>
        The asymmetry is stark. A 7-billion-parameter model in fp16 is ~14&nbsp;GB of weights. To
        emit <em>one</em> token in decode, the GPU must read all 14&nbsp;GB from high-bandwidth memory
        (HBM). On a card with ~2&nbsp;TB/s of bandwidth, that read alone is{" "}
        <M>{"14/2000 \\approx 7\\,\\text{ms}"}</M> — a hard floor on per-token latency, with the
        actual math taking far less time than the memory traffic. This is why the field obsesses over
        bytes moved, not FLOPs done.
      </p>

      <Callout type="note" title="Arithmetic intensity, in one sentence">
        <p>
          A kernel is compute-bound when it does many FLOPs per byte it loads, and
          memory-bound when it does few. Prefill (a big matmul) reuses each loaded weight across many
          tokens → compute-bound. Decode (a matmul against one token) uses each weight once →
          memory-bound. Batching many users’ decode steps together is how you claw the arithmetic
          intensity back up (next chapter).
        </p>
      </Callout>

      <h2>The KV cache: never recompute the past</h2>
      <p>
        Recall scaled dot-product attention from{" "}
        <a href="/chapter/self-attention" className="prose-link">Chapter 8</a>: the new token’s query{" "}
        <M>{"q_t"}</M> attends over the keys <M>{"K"}</M> and values <M>{"V"}</M> of <em>all</em>{" "}
        positions <M>{"1\\ldots t"}</M>. Here is the wasteful part: the keys and values of positions{" "}
        <M>{"1\\ldots t-1"}</M> depend only on those tokens, which haven’t changed since last step. A
        naïve loop re-projects them from scratch every step anyway — recomputing the entire growing
        prefix to add a single new token. Per step that is <M>{"O(t)"}</M> work for keys and values;
        summed over the whole sequence it is{" "}
        <M>{"\\sum_{t=1}^{n} t = \\tfrac{1}{2}n(n+1) = O(n^2)"}</M>. This is precisely the quadratic
        cost the{" "}
        <a href="/chapter/self-attention" className="prose-link">self-attention chapter</a> flagged.
      </p>
      <p>
        The fix is almost embarrassingly simple: <strong>keep the keys and values around</strong>.
        After computing <M>{"k_t, v_t"}</M> for the new token, append them to a per-layer cache. Next
        step, the new query attends over the <em>cached</em> <M>{"K, V"}</M> plus the one fresh row —
        no recomputation. Each decode step now does <M>{"O(1)"}</M> projection work and <M>{"O(t)"}</M>{" "}
        attention reads instead of <M>{"O(t)"}</M> projections; the per-token <em>compute</em> for the
        projections collapses from growing to constant, turning the cumulative{" "}
        <M>{"O(n^2)"}</M> recomputation into <M>{"O(n)"}</M>.
      </p>

      <Figure
        n="22.1"
        title="The KV cache, step by step"
        caption="Toggle the cache off and step through decoding: every step re-projects keys and values for the entire prefix (rose, ↻) — wasted work that grows each step. Toggle it on and only the new token is computed (amber, ✦) while the rest is read from cache (emerald, ✓). The running counters make the O(n²) → O(n) collapse concrete."
      >
        <KVCacheViz />
      </Figure>

      <Callout type="key" title="Why the KV cache is the load-bearing optimization">
        <p>
          Without it, generating an <M>{"n"}</M>-token response costs <M>{"O(n^2)"}</M> attention work
          and re-encodes the prompt at every step — generation would be unusably slow. With it, each
          new token only attends to stored state, so the marginal cost of a token is roughly constant.
          The KV cache is what makes streaming generation viable at all; everything else in this
          chapter optimizes <em>around</em> it.
        </p>
      </Callout>

      <h3>What the cache costs you: memory</h3>
      <p>
        The cache is not free — you trade compute for memory, and that memory can become the binding
        constraint. For one sequence you store, per layer, a key and a value vector for every position
        and every attention head. The total size is:
      </p>
      <MB>{String.raw`\text{KV bytes} \;=\; 2 \,\cdot\, n_{\text{layers}} \,\cdot\, n_{\text{heads}} \,\cdot\, d_{\text{head}} \,\cdot\, \text{seq} \,\cdot\, \text{bytes}`}</MB>
      <p>
        The leading <M>{"2"}</M> is for keys <em>and</em> values; <M>{"\\text{seq}"}</M> is the current
        sequence length; <M>{"\\text{bytes}"}</M> is the element size (2 for fp16). Plugging in a
        Llama-2-7B–shaped model — <M>{"n_{\\text{layers}}=32"}</M>, <M>{"n_{\\text{heads}}=32"}</M>,{" "}
        <M>{"d_{\\text{head}}=128"}</M>, fp16 — one token costs{" "}
        <M>{"2\\cdot 32\\cdot 32\\cdot 128\\cdot 2 = 524{,}288"}</M> bytes <M>{"\\approx 0.5\\,\\text{MB}"}</M>.
        A single 4,096-token context is therefore ~2&nbsp;GB of KV cache — <em>per user</em>. Serve a
        few dozen concurrent long-context users and the cache, not the weights, is what fills your GPU.
      </p>

      <Callout type="industry" title="Shrinking the cache: GQA and MQA">
        <p>
          That <M>{"n_{\\text{heads}}"}</M> term is why modern models share key/value heads.{" "}
          <strong>Multi-query attention (MQA)</strong> uses one K/V head for all query heads;{" "}
          <strong>grouped-query attention (GQA)</strong> uses a handful (e.g. 8 K/V heads for 64 query
          heads). Llama-2-70B and Llama-3 use GQA, cutting KV-cache memory and bandwidth by{" "}
          <M>{"n_{\\text{heads}}/n_{\\text{kv heads}}"}</M> — often 8× — with negligible quality loss.
          It’s a pure inference win baked into the architecture.
        </p>
      </Callout>

      <h3>A generation loop with a KV cache</h3>
      <p>
        Here is the decode loop the way Hugging Face <code>transformers</code> actually implements it.
        The key line is feeding <code>past_key_values</code> back in each step so only the newest
        token is processed:
      </p>

      <CodeBlock language="python" filename="generate.py" highlight={[13, 18, 24]}>
{`import torch

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, eos_id):
    # ---- PREFILL: process the whole prompt in parallel, in one pass ----
    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[:, -1, :]          # logits for the LAST prompt position
    past = out.past_key_values             # cached K/V for every prompt token

    generated = [int(logits.argmax(-1))]   # greedy 1st token (swap in sampling here)

    # ---- DECODE: one token at a time, reusing the cache ----
    for _ in range(max_new_tokens - 1):
        next_id = torch.tensor([[generated[-1]]], device=input_ids.device)

        # Feed ONLY the new token; 'past' carries the entire prefix's K/V.
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)

        logits = out.logits[:, -1, :]
        past = out.past_key_values          # cache grew by exactly one position

        tok = int(logits.argmax(-1))
        generated.append(tok)
        if tok == eos_id:
            break
    return generated`}
      </CodeBlock>
      <p>
        Notice that after prefill, every <code>model(...)</code> call passes an <code>input_ids</code>{" "}
        of length <strong>one</strong>. The growing context lives entirely inside{" "}
        <code>past_key_values</code>. That single architectural choice is the difference between
        linear and quadratic generation.
      </p>

      <h2>Quantization: store the weights in fewer bits</h2>
      <p>
        If decode is bottlenecked on reading weights from memory, the most direct attack is to make
        the weights <em>smaller</em>. <strong>Quantization</strong> represents each parameter in fewer
        bits than the fp16 it was trained in. Halving the bit-width roughly halves both the memory
        footprint and the bandwidth you must pay to read it — a near-linear latency win on a
        memory-bound workload, plus you can now fit bigger models on a given card.
      </p>
      <p>
        The standard ladder is <M>{"\\text{fp16} \\to \\text{int8} \\to \\text{int4}"}</M>. The idea is
        to map a block of high-precision weights onto a small integer grid with a per-block scale{" "}
        <M>{"s"}</M> (and sometimes a zero-point <M>{"z"}</M>):
      </p>
      <MB>{String.raw`w_q = \operatorname{round}\!\Big(\frac{w}{s}\Big) + z, \qquad \hat{w} = s\,(w_q - z) \approx w`}</MB>
      <p>
        At runtime the integers are read (cheaply) and de-quantized back to a float close to the
        original. The reconstruction error is what you trade for the savings. Two distinctions matter:
      </p>
      <ul>
        <li>
          <strong>Weight-only</strong> quantization stores weights in int4/int8 but de-quantizes to
          fp16 for the actual matmul. Since decode is memory-bound, this captures most of the speedup
          while keeping activations in full precision — the popular choice for serving.{" "}
          <strong>GPTQ</strong> and <strong>AWQ</strong> are the two dominant 4-bit weight-only
          methods. GPTQ quantizes layer by layer, using second-order (Hessian) information to choose
          rounding that minimizes the output error. AWQ observes that a small fraction of{" "}
          <em>salient</em> weight channels (identified by activation magnitude) carry most of the
          signal, and protects them with a per-channel scale so they survive 4-bit rounding.
        </li>
        <li>
          <strong>Weight + activation</strong> quantization (e.g. int8 W8A8, like SmoothQuant) also
          quantizes the activations so the matmul itself runs in low-precision integer arithmetic —
          faster on hardware with int8 tensor cores, and helpful in the compute-bound prefill, but
          harder to keep accurate because activations have nasty outliers.
        </li>
      </ul>

      <Callout type="pitfall" title="Quantization is lossy — measure it">
        <p>
          Lower precision is not free quality. int8 weight-only is usually within noise of fp16. Good
          4-bit (GPTQ/AWQ) typically costs a sliver of accuracy; naïve 4-bit, or pushing to 3-bit and
          below, degrades visibly — especially on reasoning and code. <strong>Always re-run your
          evals</strong> (perplexity is necessary but not sufficient; use task benchmarks) on the
          quantized model. Quality loss often hides in exactly the hard cases your benchmark suite is
          supposed to catch.
        </p>
      </Callout>

      <p>
        The tradeoff curve below sketches the typical shape: memory falls steeply as you drop bits,
        while quality holds nearly flat down to 4 bits and then falls off a cliff. The sweet spot for
        serving is almost always int8 or a careful int4.
      </p>

      <Figure
        n="22.2"
        title="The quantization tradeoff"
        caption="Illustrative. Lower bit-width slashes the memory footprint (and thus decode latency) almost linearly, while task quality stays close to the fp16 baseline down to ~4 bits, then degrades sharply. The serving sweet spot lives where the two curves are both still favorable."
      >
        <LineChart
          series={[
            {
              label: "memory footprint (GB, 7B model)",
              color: "#22d3ee",
              points: [
                [16, 14],
                [8, 7],
                [4, 3.7],
                [3, 2.9],
                [2, 2.2],
              ],
            },
            {
              label: "task quality (% of fp16)",
              color: "#34d399",
              points: [
                [16, 100],
                [8, 99.6],
                [4, 98.2],
                [3, 92],
                [2, 74],
              ],
              dashed: true,
            },
          ]}
          xLabel="bits per weight"
          yLabel="value"
          xScale="linear"
          height={300}
          annotations={[{ x: 4, label: "serving sweet spot", color: "#f59e0b" }]}
        />
      </Figure>

      <h2>FlashAttention: exact attention without the n×n matrix</h2>
      <p>
        Standard attention has a hidden memory tax that the formula hides. To compute{" "}
        <M>{"\\text{softmax}(QK^{\\top}/\\sqrt{d})V"}</M>, the naïve implementation <em>materializes</em>{" "}
        the full <M>{"n\\times n"}</M> score matrix in HBM, reads it back to softmax it, and reads it
        again to multiply by <M>{"V"}</M>. For long sequences that <M>{"n^2"}</M> matrix is enormous,
        and every byte of it makes a slow round-trip to HBM. Attention becomes bound by{" "}
        <em>memory traffic</em>, not arithmetic.
      </p>
      <p>
        <strong>FlashAttention</strong> is an <em>IO-aware</em> reformulation that computes the{" "}
        <em>exact</em> same result while never writing the <M>{"n\\times n"}</M> matrix to HBM at all.
        It <strong>tiles</strong> <M>{"Q"}</M>, <M>{"K"}</M>, <M>{"V"}</M> into blocks small enough to
        fit in the GPU’s fast on-chip <strong>SRAM</strong>, and streams over the key/value blocks one
        at a time, maintaining a running softmax (the online-softmax trick: track a running max and
        normalizer so partial results combine correctly). The scores for each tile live and die in
        SRAM; only the final <M>{"n\\times d"}</M> output is written back.
      </p>
      <p>
        The payoff is twofold. Attention’s memory footprint drops from <M>{"O(n^2)"}</M> to{" "}
        <M>{"O(n)"}</M> — you can fit much longer contexts — and because HBM traffic falls
        dramatically, the kernel runs several times faster in wall-clock even though it does the same
        (or slightly more) arithmetic. It is not an approximation: bit-for-bit, it returns standard
        attention.
      </p>

      <Callout type="industry" title="FlashAttention is the default now">
        <p>
          FlashAttention (and FlashAttention-2/3, which improve work partitioning and use newer
          tensor cores) is the de-facto attention kernel in PyTorch (<code>
          scaled_dot_product_attention</code>), vLLM, and essentially every serious training and
          serving stack. If you call attention through a modern framework, you are almost certainly
          already getting it — for both training and inference. It is one of the highest-leverage
          systems contributions to the field.
        </p>
      </Callout>

      <h2>Speculative decoding: more than one token per step</h2>
      <p>
        Decode is memory-bound, which means a step that produces one token and a step that{" "}
        <em>verifies several</em> cost almost the same in wall-clock — the weights get read once
        either way. <strong>Speculative decoding</strong> exploits exactly this. A small, fast{" "}
        <strong>draft model</strong> proposes the next <M>{"k"}</M> tokens cheaply. Then the big{" "}
        <strong>target model</strong> runs a <em>single</em> parallel forward pass over all{" "}
        <M>{"k"}</M> proposed tokens at once (just like prefill) and checks them.
      </p>
      <p>
        Because the target processes the whole guessed chunk in one batched pass, you get its
        true next-token distribution at every guessed position simultaneously. A clever{" "}
        acceptance rule (rejection sampling) keeps the longest correct prefix and corrects the first
        mistake — and it is provably <strong>distribution-preserving</strong>: the accepted tokens are
        exactly samples from the big model, so output quality is identical, only faster. When the
        draft is good, you accept several tokens per target pass:
      </p>
      <MB>{String.raw`\text{tokens per target pass} \;=\; 1 + (\text{accepted drafts}), \qquad 1 \le \cdot \le k+1`}</MB>
      <p>
        On predictable text (boilerplate, code, formatting) the draft is usually right and you might
        accept 3–4 tokens per pass — a 2–3× speedup. On surprising text it’s wrong, you fall back to
        one token, and you’ve paid only the cheap draft. The expected speedup is governed by the
        acceptance rate, which is why a draft model that <em>agrees</em> with the target matters more
        than one that is merely small. (Variants like Medusa add extra prediction heads to the target
        itself, removing the need for a separate draft model.)
      </p>

      <Callout type="note" title="Why it’s free quality">
        <p>
          The draft only <em>proposes</em>; the target <em>decides</em>. Every emitted token is
          verified (or replaced) by the big model under a sampling rule that provably matches the
          target’s distribution. You are not trading accuracy for speed — you are using idle GPU
          arithmetic (during memory-bound decode) to check guesses for free.
        </p>
      </Callout>

      <h2>Distillation: train a smaller model to imitate the big one</h2>
      <p>
        Quantization shrinks a model’s <em>bits</em>; <strong>distillation</strong> shrinks its{" "}
        <em>parameters</em> by training a small <strong>student</strong> to mimic a large{" "}
        <strong>teacher</strong>. Instead of (or in addition to) the hard one-hot labels, the student
        learns from the teacher’s full <em>soft</em> probability distribution over the vocabulary —
        the “dark knowledge” in those probabilities (that “feline” is a plausible alternative to
        “cat”) is far richer supervision than a single correct token. The usual objective minimizes
        the KL divergence between student and (temperature-softened) teacher distributions:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{distill}} \;=\; \tau^2 \, \mathrm{KL}\!\big(\,p^{\text{teacher}}_{\tau} \;\big\|\; p^{\text{student}}_{\tau}\,\big)`}</MB>
      <p>
        where <M>{"\\tau"}</M> is a temperature that softens both distributions to expose the
        teacher’s relative preferences. A well-distilled student can recover most of a much larger
        teacher’s quality at a fraction of the cost — and unlike quantization, the result is a
        genuinely smaller, faster network. Many of the small “fast” model tiers you see in production
        are distilled from their larger siblings. The catch: distillation requires a <em>training</em>{" "}
        run and a capable teacher, so it’s a heavier investment than the post-hoc tricks above.
      </p>

      <Callout type="tip" title="These optimizations stack">
        <p>
          None of these are mutually exclusive — production stacks compose them. A typical recipe: a
          GQA model, weight-quantized to int4 (AWQ), served with FlashAttention and a KV cache, with
          speculative decoding bolted on for latency-sensitive traffic — and perhaps a distilled
          variant for the cheap tier. Each attacks a different part of the cost; together they’re
          often an order of magnitude over a naïve baseline.
        </p>
      </Callout>

      <h2>A note on batching → Chapter 23</h2>
      <p>
        Everything so far optimized a <em>single</em> generation. But the biggest lever of all is
        amortization across users. Because decode is memory-bandwidth-bound — you pay to read the
        weights regardless — running many users’ tokens through the <em>same</em> weight read is
        nearly free extra throughput. <strong>Batching</strong> raises arithmetic intensity and is how
        a single GPU serves hundreds of concurrent conversations. How to batch well when every request
        has a different length and arrives at a different time — static vs. continuous batching, paged
        KV caches, and the full serving stack — is the subject of{" "}
        <a href="/chapter/deployment-serving" className="prose-link">Chapter 23</a>.
      </p>
    </>
  );
}
