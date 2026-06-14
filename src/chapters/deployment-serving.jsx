import { M } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import BatchingViz from "../components/viz/deployment-serving/BatchingViz";

export default function Chapter() {
  return (
    <>
      <p>
        <a href="/chapter/inference-optimization" className="prose-link">Chapter 22</a> made one
        generation fast. This chapter makes <em>thousands at once</em> fast — turning a model that
        runs in a notebook into a service that answers a stream of concurrent users at high
        utilization and low cost. Serving is where machine learning meets distributed systems: the
        win comes less from clever math than from never letting an expensive GPU sit idle.
      </p>
      <p>
        The thread running through everything below is the same insight from the previous chapter:
        decode is <strong>memory-bandwidth-bound</strong>, so you pay to read the model’s weights from
        memory whether one user or fifty are decoding. Pack more users into that same weight read and
        throughput climbs almost for free. The entire serving stack is built to make that packing as
        tight as physically possible.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          A serving system’s job is to keep the GPU’s tensor cores busy. Because reading the weights
          dominates decode, the lever that matters most is <strong>batch size</strong> — how many
          users’ tokens share each weight read. <strong>Continuous batching</strong> keeps that batch
          full at every single step by swapping finished requests out and waiting ones in, which is
          why it can deliver several times the throughput of the naïve approach.
        </p>
      </Callout>

      <h2>The serving stack</h2>
      <p>
        Between a user’s HTTP request and a token coming back sits a small pipeline. Understanding its
        layers tells you where each optimization lives:
      </p>
      <ul>
        <li>
          <strong>API layer</strong> — an HTTP server (often FastAPI) that speaks an
          OpenAI-compatible protocol, authenticates, applies rate limits, and <em>streams</em> tokens
          back as they’re produced.
        </li>
        <li>
          <strong>Scheduler / batcher</strong> — the brain. It admits requests, groups their decode
          steps into batches, decides whose turn it is each step, and manages the KV cache. This is
          where continuous batching and paged attention live.
        </li>
        <li>
          <strong>Inference engine</strong> — the optimized model runtime (vLLM, TensorRT-LLM, TGI,
          SGLang) executing the forward passes with FlashAttention, quantized weights, and CUDA
          graphs.
        </li>
        <li>
          <strong>Replicas + load balancer</strong> — many copies of the engine across GPUs/nodes,
          fronted by a balancer and an autoscaler that adds or removes replicas with demand.
        </li>
      </ul>

      <h2>Static vs. continuous batching</h2>
      <p>
        Batching groups multiple requests so each model forward pass produces tokens for all of them
        at once. The question is <em>when</em> you form and dissolve the batch.
      </p>
      <p>
        <strong>Static (request-level) batching</strong> locks a group of requests together for their
        whole lifetime. The batch runs in lockstep until the <em>slowest</em> sequence finishes; any
        request that completes early leaves its slot <strong>idle</strong> — occupying a seat,
        contributing nothing — until the whole batch drains. Because completion lengths vary wildly
        (a 5-token reply next to a 500-token essay), a static batch spends much of its life
        half-empty. Worse, a new request arriving mid-batch must wait in line for the entire current
        batch to finish before it can even start.
      </p>
      <p>
        <strong>Continuous (in-flight, iteration-level) batching</strong> reschedules at the
        granularity of a <em>single decode step</em>. The instant a sequence emits its end token and
        frees a slot, the scheduler admits a waiting request into that slot on the very next step.
        Finished sequences leave immediately; new ones join immediately. The batch is reassembled
        every iteration to stay as full as possible, so the GPU keeps doing useful work instead of
        idling on completed seats — typically a <strong>2–4× throughput</strong> improvement on real,
        length-skewed traffic, and dramatically lower queueing latency.
      </p>

      <Figure
        n="23.1"
        title="Static vs. continuous batching"
        caption="GPU slots (rows) over decode steps (columns). In static batching, short requests finish early but their slots sit idle (dashed) until the slowest request drains the whole batch. Switch to continuous and freed slots are refilled from the queue on the next step — utilization climbs sharply. Step through to watch the gaps appear and disappear."
      >
        <BatchingViz />
      </Figure>

      <Callout type="key" title="Why continuous batching is the centerpiece">
        <p>
          Static batching wastes the most expensive resource you have — GPU-seconds — on idle,
          finished slots, and makes new requests wait for the slowest sequence in front of them.
          Continuous batching eliminates both: every step, the batch is repacked so freed capacity is
          handed straight to a waiting request. It’s the single change that most raises real-world
          throughput and is the default in every modern serving engine.
        </p>
      </Callout>

      <h2>PagedAttention and vLLM</h2>
      <p>
        Continuous batching creates a memory-management nightmare. Each active sequence owns a KV
        cache that <em>grows</em> token by token, and you don’t know its final length in advance.
        Reserve the maximum context length per sequence and you waste enormous memory on requests that
        finish early; reserve too little and you can’t grow. Storing each sequence’s cache as one
        contiguous block also <strong>fragments</strong> memory — freed gaps of the wrong size can’t be
        reused — so you fit far fewer concurrent sequences than the raw capacity allows.
      </p>
      <p>
        <strong>PagedAttention</strong>, the idea at the heart of <strong>vLLM</strong>, borrows the
        operating-system trick of <em>virtual memory</em>. The KV cache is split into fixed-size{" "}
        <strong>blocks</strong> (pages) of, say, 16 tokens. A sequence’s cache is a <em>list</em> of
        such blocks — physically scattered across GPU memory but logically contiguous via a block
        table, exactly like OS page tables mapping virtual pages to physical frames. Blocks are
        allocated on demand as a sequence grows and freed the instant it ends. The result:
      </p>
      <ul>
        <li>
          <strong>Near-zero fragmentation</strong> — blocks are uniform, so any freed block fits any
          new need. vLLM reports KV memory waste dropping from ~60–80% (reserved-contiguous) to a few
          percent, which directly translates to a much larger effective batch.
        </li>
        <li>
          <strong>Sharing via copy-on-write</strong> — sequences with a common prefix (the same system
          prompt, or parallel samples of one prompt) <em>share</em> the physical blocks for that
          prefix until they diverge, then copy only the block that differs. Beam search and{" "}
          <M>{"n"}</M>-way sampling get dramatically cheaper.
        </li>
      </ul>

      <Callout type="industry" title="vLLM in practice">
        <p>
          vLLM pairs PagedAttention with continuous batching and FlashAttention and has become a
          default open-source serving engine: it reports up to ~24× the throughput of naïve Hugging
          Face <code>generate</code> on serving workloads. It speaks the OpenAI API out of the box,
          supports quantized weights (AWQ/GPTQ), tensor parallelism across GPUs, and prefix caching.
          Production alternatives — NVIDIA TensorRT-LLM, Hugging Face TGI, SGLang — make similar
          architectural choices.
        </p>
      </Callout>

      <h2>Throughput vs. latency: the fundamental tradeoff</h2>
      <p>
        You cannot maximize everything at once. The central tension in serving is{" "}
        <strong>throughput</strong> (tokens/second across <em>all</em> users — what sets your cost per
        token) versus <strong>latency</strong> (how fast <em>one</em> user gets served). They pull in
        opposite directions, and <strong>batch size</strong> is the knob between them.
      </p>
      <p>Latency itself splits into two numbers users feel differently:</p>
      <ul>
        <li>
          <strong>TTFT — time to first token.</strong> How long until the response <em>starts</em>.
          Dominated by the prefill of the prompt (longer prompts → slower TTFT) plus any time spent
          waiting in the scheduler’s queue. This is what makes a chat feel responsive.
        </li>
        <li>
          <strong>TPOT — time per output token</strong> (a.k.a. inter-token latency). The steady-state
          gap between streamed tokens during decode; its reciprocal is the per-user generation speed
          in tokens/sec. This is what makes a long answer feel fast or sluggish.
        </li>
      </ul>
      <p>
        A rough budget for one request:{" "}
        <M>{"\\text{total} \\approx \\text{TTFT} + (\\text{output tokens}-1)\\times\\text{TPOT}"}</M>.
        Now the tradeoff: a <em>bigger</em> batch amortizes each weight read over more users, raising
        aggregate throughput — but every user in that batch shares the step, so each one’s TPOT rises,
        and a fuller queue raises TTFT. A <em>smaller</em> batch gives each user snappier tokens at the
        cost of GPU efficiency and dollars. Serving systems therefore cap batch size and queue depth to
        hold TTFT/TPOT under a Service-Level Objective (e.g. “TTFT &lt; 500&nbsp;ms, TPOT &lt;
        50&nbsp;ms”) while batching as much as that budget allows. (Some stacks also separate the
        compute-bound prefill from the memory-bound decode onto different replicas — “disaggregated
        serving” — so a big prefill can’t stall everyone’s decode.)
      </p>

      <Callout type="note" title="Pick the metric that matches the product">
        <p>
          Optimize <strong>TTFT</strong> for interactive chat and autocomplete; optimize{" "}
          <strong>throughput</strong> (and tolerate higher latency) for batch/offline jobs like bulk
          summarization or evals. The same model and hardware serve both — you just choose a different
          point on the batch-size curve. There is no single “fast”; there is fast <em>for whom</em>.
        </p>
      </Callout>

      <h2>The API layer: streaming and compatibility</h2>
      <p>
        Generation is incremental, so the API should be too. Rather than block for seconds and return
        the whole answer, production endpoints <strong>stream</strong> tokens as they’re decoded —
        which also slashes <em>perceived</em> latency, since the user starts reading at TTFT instead of
        waiting for the final token. The dominant mechanism is <strong>Server-Sent Events (SSE)</strong>:
        a long-lived HTTP response that emits <code>data:</code> chunks, one per token (or small group),
        terminated by a <code>data: [DONE]</code> sentinel. Most providers also expose an{" "}
        <strong>OpenAI-compatible</strong> schema (<code>/v1/chat/completions</code> with{" "}
        <code>stream=true</code>) so existing client SDKs work unchanged.
      </p>
      <p>
        Here is a minimal FastAPI streaming endpoint. In production you’d delegate the actual decoding
        to a batching engine; this shows the SSE contract directly:
      </p>

      <CodeBlock language="python" filename="serve.py" highlight={[14, 20, 23]}>
{`from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import json

app = FastAPI()
tok = AutoTokenizer.from_pretrained("my-model")
model = AutoModelForCausalLM.from_pretrained("my-model", device_map="cuda")

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256

def token_stream(prompt: str, max_new_tokens: int):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    # generate() runs in a background thread; tokens arrive via the streamer.
    Thread(target=model.generate,
           kwargs=dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)).start()
    for piece in streamer:                      # yields decoded text as it is produced
        yield f"data: {json.dumps({'token': piece})}\\n\\n"   # one SSE event per chunk
    yield "data: [DONE]\\n\\n"

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    return StreamingResponse(
        token_stream(req.prompt, req.max_new_tokens),
        media_type="text/event-stream",        # the SSE content type
    )`}
      </CodeBlock>

      <Callout type="industry" title="Don’t hand-roll the engine in production">
        <p>
          The snippet above is for understanding the SSE contract — it serves one request at a time
          with no batching. In production you put a real engine behind the same HTTP shape. With vLLM
          that is essentially a one-liner that gives you continuous batching, paged attention, and an
          OpenAI-compatible server for free:
        </p>
      </Callout>

      <CodeBlock language="bash" filename="serve.sh">
{`# Launch an OpenAI-compatible, continuously-batched, paged-attention server.
vllm serve my-org/my-model \\
  --quantization awq \\          # int4 weight-only quantization (Ch.22)
  --tensor-parallel-size 2 \\    # shard the model across 2 GPUs
  --max-model-len 8192 \\        # context window
  --gpu-memory-utilization 0.92 # leave headroom; the rest is KV-cache blocks

# Clients now hit it exactly like the OpenAI API:
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"my-org/my-model","messages":[{"role":"user","content":"Hello!"}],"stream":true}'`}
      </CodeBlock>

      <h2>Scaling out: replicas, autoscaling, load balancing</h2>
      <p>
        One GPU (or one tensor-parallel group) has a ceiling — bounded by memory for the weights plus
        KV cache, and by compute for prefill. Past that you scale <strong>horizontally</strong>:
      </p>
      <ul>
        <li>
          <strong>Multiple replicas</strong> — independent copies of the engine, each on its own
          GPU(s). A model too big for one card is first split <em>within</em> a replica via{" "}
          <strong>tensor parallelism</strong> (each GPU holds a shard of every layer) or{" "}
          <strong>pipeline parallelism</strong> (each GPU holds a contiguous set of layers); replicas
          then multiply that unit for throughput.
        </li>
        <li>
          <strong>Load balancing</strong> — route each request to a replica. Naïve round-robin is
          weak here because requests vary so much in cost; production balancers prefer the
          least-loaded replica (by active sequences or queue depth) and use{" "}
          <strong>prefix-aware</strong> routing — sending requests that share a system prompt to the
          same replica so its prefix cache (and shared KV blocks) is reused.
        </li>
        <li>
          <strong>Autoscaling</strong> — add replicas as load rises and remove them when it falls,
          scaling on a serving-relevant signal (queue depth, TTFT, GPU utilization) rather than CPU.
          The hard part is that a cold replica takes tens of seconds to load tens of gigabytes of
          weights, so you scale ahead of demand and keep warm headroom; bursty traffic makes this a
          genuine forecasting problem.
        </li>
      </ul>

      <Callout type="tip" title="Cost optimization checklist">
        <p>
          The dollar-per-token wins compound. <strong>Quantize</strong> the weights (int8/int4) to fit
          more on each GPU and read them faster.{" "}
          <strong>Continuous batching + paged attention</strong> to push utilization toward 100%.{" "}
          <strong>Prefix / prompt caching</strong> so shared system prompts are prefilled once and
          reused across requests. <strong>Spot / preemptible GPUs</strong> for fault-tolerant batch
          workloads at a fraction of on-demand price. <strong>Right-size the model</strong> — route
          easy requests to a small or distilled model and reserve the flagship for hard ones. And{" "}
          <strong>cap context length</strong> to what you actually need, since KV cache (and thus the
          number of users per GPU) scales with it.
        </p>
      </Callout>

      <h2>Putting it together → the capstone</h2>
      <p>
        You now have the full path from a single fast token to a fleet serving the world: prefill and
        decode, the KV cache and its memory cost, quantization, FlashAttention, and speculative
        decoding from{" "}
        <a href="/chapter/inference-optimization" className="prose-link">Chapter 22</a>; and here, the
        serving stack — continuous batching, paged attention, the throughput/latency tradeoff,
        streaming APIs, and horizontal scale. A request arrives, the balancer picks the least-loaded
        replica, the scheduler slots it into a continuous batch over a paged KV cache, the engine
        decodes it with quantized weights and FlashAttention, and tokens stream back over SSE — at a
        cost of fractions of a cent.
      </p>
      <p>
        What remains is keeping that system <em>honest and safe</em> once real users arrive —
        observability, drift, abuse, and guardrails — which is{" "}
        <a href="/chapter/monitoring-safety" className="prose-link">Chapter 24</a>, before the{" "}
        <a href="/chapter/capstone" className="prose-link">capstone</a> assembles the entire pipeline
        end to end.
      </p>
    </>
  );
}
