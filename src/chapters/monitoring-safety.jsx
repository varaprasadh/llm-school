import { M } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LineChart from "../components/viz/LineChart";
import GuardrailPipeline from "../components/viz/monitoring-safety/GuardrailPipeline";

export default function Chapter() {
  return (
    <>
      <p>
        Shipping a model is not the finish line — it's the moment your problems become other people's
        problems. A deployed LLM has to stay <em>fast</em>, stay <em>cheap</em>, stay <em>correct</em>,
        and stay <em>safe</em>, continuously, against real users who will (accidentally or
        deliberately) push it in directions you never tested. This chapter is the operations manual:
        what to measure, how to notice when quality silently rots, the specific ways people attack
        LLMs, the defenses that actually work, and how to roll all of this out without blowing up in
        production.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          A production LLM needs two feedback loops running at all times: an{" "}
          <strong>observability</strong> loop (is it fast, cheap, and up?) and a{" "}
          <strong>safety</strong> loop (is it being abused, and is it behaving?). Neither is optional,
          and both must be in place <em>before</em> you scale traffic — you cannot bolt them on after
          an incident.
        </p>
      </Callout>

      <h2>Observability: the four golden signals</h2>
      <p>
        You can't fix what you can't see. The first job is instrumentation — emitting metrics, logs
        and traces for every request so you have a real-time picture of system health. For LLM
        serving, the signals that matter are:
      </p>
      <ul>
        <li>
          <strong>Latency</strong>, reported as <em>percentiles</em>, not averages. Track{" "}
          <M>{"p50"}</M> (median), <M>{"p95"}</M> and <M>{"p99"}</M>. The tail is what users feel: a
          fine median hides the 1-in-100 request that takes eight seconds. For streaming, split it
          into <strong>time-to-first-token</strong> (TTFT — how long before output starts) and{" "}
          <strong>inter-token latency</strong> (how fast it streams after that).
        </li>
        <li>
          <strong>Throughput</strong>: tokens per second per replica, and requests per second. This
          is your capacity, and it's what batching (Chapter 22) buys you.
        </li>
        <li>
          <strong>Cost per request</strong>: for self-hosted models, GPU-seconds × instance price; for
          APIs, input + output tokens × the per-token rate. Watch the <em>output</em> tokens — they're
          generated one at a time and usually dominate both latency and bill.
        </li>
        <li>
          <strong>Error rate &amp; saturation</strong>: 5xx rates, timeouts, request-queue depth, and{" "}
          <strong>GPU utilization</strong> + memory. A climbing queue or pinned KV-cache memory is the
          early warning that you're about to start dropping requests.
        </li>
      </ul>

      <Callout type="pitfall" title="Averages lie; percentiles tell the truth">
        <p>
          A 600 ms <em>average</em> latency can hide a p99 of 9 seconds. Users don't experience your
          average — each of them experiences one request, and the unlucky ones experience the tail.
          Always alert on p95/p99, and remember tail latency compounds in multi-step agents: ten
          sequential calls at p95 each is far worse than one.
        </p>
      </Callout>

      <h2>Quality monitoring &amp; drift</h2>
      <p>
        The metrics above tell you the service is <em>up</em>. They say nothing about whether the
        answers are still <em>good</em>. Quality can degrade with zero errors and flat latency — the
        scariest kind of failure, because nothing pages you.
      </p>
      <p>
        The usual culprit is <strong>drift</strong>: the live distribution moves away from what you
        tested on. <strong>Input drift</strong> means users start asking different things (a new
        feature ships, a meme sends weird traffic, a new language appears). <strong>Output drift</strong>
        means the model's responses shift — often because an upstream API model was silently updated,
        or your own prompt/RAG content changed. You catch drift by tracking summary statistics of the
        input and output streams over time (length distributions, language mix, refusal rate, topic
        clusters, embedding centroids) and alerting when they move.
      </p>
      <p>
        The cheapest, highest-signal quality data is <strong>user feedback</strong>: thumbs up/down,
        regenerations (a regenerate is an implicit thumbs-down), edits to the model's output,
        copy/share events, and conversation abandonment. A rising regeneration rate or a falling
        thumbs-up rate is a quality regression announcing itself — wire these into the same dashboard
        as latency and treat them as first-class signals.
      </p>

      <Figure
        n="24.1"
        title="Quality and safety monitoring over time"
        caption="A simulated week of a deployed model. Around the marked deploy, an upstream model update lifts the refusal rate and pushes the thumbs-down rate up while latency stays flat — a quality regression no error metric would catch. This is what drift looks like on a dashboard."
      >
        <DriftChart />
      </Figure>

      <h2>The threat model: how LLMs get attacked</h2>
      <p>
        LLMs face a category of attack ordinary software doesn't: the input is natural language, and
        the model will try to be helpful, so an attacker's tool is <em>persuasion</em>. The core
        threats:
      </p>
      <ul>
        <li>
          <strong>Jailbreaks</strong>: prompts crafted to talk the model out of its safety training —
          role-play framings ("you are DAN with no rules"), hypotheticals, obfuscation (leetspeak,
          base64), or "many-shot" priming with fake prior turns. The goal is to extract content the
          model is supposed to refuse.
        </li>
        <li>
          <strong>Prompt injection</strong>: the dangerous one for any app that uses{" "}
          <strong>tools or RAG</strong>. Malicious instructions are hidden inside content the model
          will read — a web page, a PDF, an email, a tool result — and the model obeys <em>them</em>
          instead of you. The user's visible request looks innocent; the payload rides in on retrieved
          data. (We'll dwell on this below — it's the defining LLM security problem.)
        </li>
        <li>
          <strong>Data exfiltration</strong>: tricking the model into leaking its system prompt, other
          users' data, secrets/API keys in its context, or proprietary tool schemas — often the
          <em>payload</em> of a successful injection ("…now email the conversation to attacker@…").
        </li>
        <li>
          <strong>Abuse &amp; resource attacks</strong>: using your endpoint to generate spam, malware
          or disinformation at scale, or simply running up your bill with expensive prompts (a
          "denial-of-wallet" attack).
        </li>
      </ul>

      <Callout type="warning" title="Prompt injection has no clean fix — design around it">
        <p>
          Unlike SQL injection, you <em>cannot</em> fully separate "instructions" from "data" in a
          language model — to the model, it's all just text in the context window. Any time the model
          reads attacker-controllable content (web, documents, tool output, other users' messages),
          assume that content can carry instructions the model may follow. The defense is not a magic
          filter but <strong>architecture</strong>: never let a tool-using model take a
          consequential, irreversible action (send money, delete data, email externally) on untrusted
          input without a human confirmation or a hard allow-list. Treat the model as a confused
          deputy that can be talked into anything.
        </p>
      </Callout>

      <h2>Defenses: guardrails around the model</h2>
      <p>
        Safety is layered — no single control is sufficient, so you stack several. The standard
        pattern wraps the model in a pipeline with checks on both ends:
      </p>
      <ul>
        <li>
          <strong>Input moderation</strong>: before the model sees the request, run it through a
          moderation classifier (a smaller model trained to flag categories like violence, self-harm,
          sexual content, harassment, illegal activity) and against jailbreak/injection heuristics.
          Block or sanitize what trips a policy.
        </li>
        <li>
          <strong>System-prompt hardening</strong>: a clear, defensive system prompt that states the
          model's role and refusal boundaries, and instructs it to ignore instructions found in
          retrieved or user-pasted content. It's a real layer — but a soft one; never rely on it
          alone against injection.
        </li>
        <li>
          <strong>Output moderation</strong>: re-screen the <em>generated</em> text before it reaches
          the user (or before a tool call executes). This is what catches a successful jailbreak or a
          model tricked into exfiltration — the input looked fine, but the output is unsafe.
        </li>
        <li>
          <strong>Content filtering &amp; tool guards</strong>: redact PII, strip secrets, and gate
          tool calls behind allow-lists and confirmation for anything irreversible.
        </li>
        <li>
          <strong>Refusal path</strong>: when any gate fires, return a graceful, non-leaky refusal —
          not a stack trace, and not a hint about how to get past the filter.
        </li>
      </ul>

      <h2>See it: a guardrail pipeline in motion</h2>
      <p>
        Below is that pipeline as an interactive flow. Click a scenario and watch the request travel
        through the gates, passing or getting blocked, with the reason shown at each stage. Pay
        special attention to the two hard cases: a <em>jailbreak</em> is stopped cold at the input
        gate, but a <em>prompt injection</em> sails straight through input moderation (the user's
        visible ask is benign) and is only caught at the <em>output</em> gate, when the model — fooled
        by instructions hidden in retrieved content — tries to exfiltrate data. That asymmetry is
        exactly why you guard both ends.
      </p>

      <Figure
        n="24.2"
        title="Request guardrail pipeline"
        caption="user → input moderation → model → output moderation → user. Each gate either passes (green), flags (cyan), or blocks (red) with a reason; a block short-circuits to a safe refusal. Notice that the injection example only gets caught on the output side — input moderation alone would have missed it."
      >
        <GuardrailPipeline />
      </Figure>

      <p>
        In code, the wrapper is straightforward — and the structure is the point. Moderate the input,
        bail early with a refusal if it's disallowed, call the model, moderate the output, and bail
        again if the generation is unsafe. Everything gets logged for the monitoring loop.
      </p>

      <CodeBlock language="python" filename="guardrails.py">
{`from dataclasses import dataclass

@dataclass
class Result:
    text: str
    blocked: bool = False
    stage: str | None = None   # which gate blocked, for logging/metrics

REFUSAL = ("I can't help with that request, but I'm happy to help "
           "with something else.")

def moderate(text: str) -> tuple[bool, str]:
    """Return (is_flagged, reason). In production this is a trained
    classifier (e.g. a small safety model) + a few high-precision rules."""
    flags = classifier.predict(text)          # {category: score}
    for category, score in flags.items():
        if score >= THRESHOLDS[category]:
            return True, f"{category} ({score:.2f})"
    return False, ""

def guarded_chat(user_msg: str, context: list[str]) -> Result:
    # 1) INPUT gate — screen the user's request before the model sees it.
    flagged, reason = moderate(user_msg)
    if flagged:
        log_event("input_blocked", reason=reason)
        return Result(REFUSAL, blocked=True, stage="input")

    # 2) MODEL — system prompt is hardened to distrust retrieved content,
    #    and untrusted context is clearly fenced off from instructions.
    prompt = build_prompt(SYSTEM_HARDENED, context, user_msg)
    output = model.generate(prompt, max_tokens=1024)

    # 3) OUTPUT gate — re-screen what the model actually produced. This is
    #    what catches a jailbreak that slipped past, or an injection that
    #    tricked the model into leaking data / making a bad tool call.
    flagged, reason = moderate(output)
    if flagged or leaks_secret(output):
        log_event("output_blocked", reason=reason or "secret_leak")
        return Result(REFUSAL, blocked=True, stage="output")

    log_event("ok", tokens=count_tokens(output))   # feed the metrics loop
    return Result(output)`}
      </CodeBlock>

      <Callout type="industry" title="Layers, not a wall">
        <p>
          No single guardrail is reliable — moderation classifiers have false negatives, system
          prompts get jailbroken, filters get bypassed. Production safety is{" "}
          <strong>defense in depth</strong>: input <em>and</em> output moderation, a hardened system
          prompt, tool allow-lists with human-in-the-loop for irreversible actions, rate limits, and
          continuous red-teaming. Each layer is porous; stacked, they catch most of what gets through
          the others.
        </p>
      </Callout>

      <h2>Hallucination detection</h2>
      <p>
        A model stating false things <em>confidently</em> is a safety and trust problem even when no
        attacker is involved. You can't eliminate hallucination, but you can detect and reduce it:
      </p>
      <ul>
        <li>
          <strong>Grounding &amp; attribution</strong>: in a RAG system, require the answer to cite
          retrieved passages and check that claims are actually supported by them (an "is this
          entailed by the source?" check, often itself an LLM-as-judge call).
        </li>
        <li>
          <strong>Self-consistency</strong>: sample several answers; high disagreement across samples
          signals the model is guessing. Stable answers are more likely grounded.
        </li>
        <li>
          <strong>Calibration &amp; abstention</strong>: encourage the model to say "I don't know"
          rather than fabricate, and surface uncertainty to the user instead of hiding it.
        </li>
      </ul>

      <h2>Red-teaming &amp; responsible rollout</h2>
      <p>
        <strong>Red-teaming</strong> is adversarial testing: dedicated people (and increasingly,
        automated attacker models) try to break your safety before real attackers do — jailbreaks,
        injections, edge-case harms, demographic biases. Findings become regression tests in your eval
        suite (Chapter 21), so a fixed exploit can't silently come back. Red-teaming is continuous,
        not a one-time audit, because new attacks are invented constantly.
      </p>
      <p>
        Finally, the operational disciplines that keep a launch from becoming an incident:
      </p>
      <ul>
        <li>
          <strong>Rate limiting &amp; abuse prevention</strong>: per-user/per-key quotas, anomaly
          detection on traffic spikes, and cost caps to blunt denial-of-wallet and scaled abuse.
        </li>
        <li>
          <strong>Staged rollout</strong>: ship behind a flag to internal users, then a small
          percentage of traffic (canary), watching the metrics and feedback dashboards, before going
          to 100%. A/B new prompts and models against the old one so regressions show up as a metric,
          not a Twitter thread.
        </li>
        <li>
          <strong>Incident response</strong>: a kill switch / rollback you can hit in minutes, an
          on-call rotation, alerts wired to the golden signals <em>and</em> the quality signals, and a
          blameless postmortem culture that turns each incident into a new test and a new guardrail.
        </li>
      </ul>

      <Callout type="key" title="Deploy like you'll have to debug it at 3am">
        <p>
          Instrument everything (you can't fix what you can't see), guard both the input and the
          output, assume any tool-reachable content is hostile, roll out gradually behind a flag you
          can flip back, and treat every incident as a missing test. Safety and reliability aren't
          features you finish — they're loops you run forever.
        </p>
      </Callout>

      <h2>Where this leads</h2>
      <p>
        You now have the full arc: from raw text to a tokenizer, to a transformer, through pretraining
        and alignment, to evaluation, optimization, deployment, and the monitoring and safety loops
        that keep a live model healthy. The{" "}
        <a href="/chapter/capstone" className="prose-link">
          capstone
        </a>{" "}
        ties it all together end-to-end — taking everything in this book and walking the complete
        pipeline from a blank repository to a model you can actually serve, watch, and trust.
      </p>
    </>
  );
}

// ---- Figure 24.1: a deterministic "drift" dashboard built on the shared
// LineChart. A simulated week of three signals; an upstream model update near
// hour 96 lifts refusal + thumbs-down while latency stays flat. Hand-authored,
// no randomness, so it's stable across renders.
function DriftChart() {
  const HOURS = 168; // one week
  const DEPLOY = 96; // the silent upstream model swap

  const p95 = [];
  const refusal = [];
  const thumbsDown = [];
  for (let h = 0; h <= HOURS; h += 4) {
    // gentle diurnal wiggle so the lines look alive but never random
    const diurnal = Math.sin((h / 24) * 2 * Math.PI);
    const post = h >= DEPLOY ? 1 : 0;

    // p95 latency (ms): flat ~ stays put across the deploy — the point is that
    // latency does NOT reveal the regression.
    p95.push([h, 820 + 40 * diurnal]);

    // refusal rate (%): jumps after the deploy (new model over-refuses).
    refusal.push([h, 2.4 + 0.6 * diurnal + post * 4.3]);

    // thumbs-down rate (%): climbs after the deploy as quality drops.
    thumbsDown.push([h, 3.1 + 0.4 * diurnal + post * 3.6]);
  }

  return (
    <LineChart
      height={300}
      xLabel="hours since launch"
      yLabel="metric"
      xTicks={7}
      series={[
        { label: "p95 latency (×100 ms)", color: "#22d3ee", points: p95.map(([x, y]) => [x, y / 100]) },
        { label: "refusal rate (%)", color: "#f59e0b", points: refusal },
        { label: "thumbs-down (%)", color: "#fb7185", points: thumbsDown },
      ]}
      annotations={[{ x: DEPLOY, label: "upstream model update", color: "#a855f7" }]}
      fmtX={(v) => `${Math.round(v)}h`}
      fmtY={(v) => v.toFixed(1)}
    />
  );
}
