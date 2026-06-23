import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LifecyclePipeline from "../components/viz/LifecyclePipeline";
import PipelineRecap from "../components/viz/capstone/PipelineRecap";
import { useNavigate } from "react-router-dom";

export default function Chapter() {
  const navigate = useNavigate();
  return (
    <>
      <p>
        Twenty-four chapters ago, in{" "}
        <button onClick={() => navigate("/chapter/what-is-an-llm")} className="prose-link">
          Chapter 1
        </button>
        , we made a promise: an LLM does exactly one thing — it guesses the next token — and{" "}
        <em>everything else is an elaboration of that single idea</em>. You then watched that idea
        unfold across a whole pipeline: scraping the web, learning a vocabulary, stacking attention,
        running gradient descent at scale, teaching the model to follow instructions, aligning it to
        human preferences, squeezing it for speed, and finally serving tokens to a real person. This
        last chapter is the payoff. We assemble every stage into one coherent mental model, attach
        real numbers to it, and hand you a concrete plan to go build one yourself.
      </p>

      <Callout type="key" title="The whole book in one sentence">
        <p>
          An LLM is a next-token predictor that you <strong>grow</strong> (data → architecture →
          pretraining), <strong>shape</strong> (fine-tuning → alignment), and <strong>ship</strong>{" "}
          (optimize → deploy → monitor). Each arrow is a chapter; the chain only works end to end.
        </p>
      </Callout>

      <h2>The whole journey, in one view</h2>
      <p>
        You have seen this diagram before — it opened the book. Back then every stage was a mystery
        box. Now you have read every line inside every box. Hover the stages and let the closure
        land: there is no longer anything here you cannot explain to someone else.
      </p>

      <Figure
        n="25.1"
        title="The end-to-end LLM lifecycle — revisited"
        caption="The same pipeline from Chapter 1, Figure 1.2. The difference is you. Every stage that was a black box in Part I is now an open book; click any step to revisit the chapter where you opened it."
      >
        <LifecyclePipeline onPick={(slug) => navigate(`/chapter/${slug}`)} />
      </Figure>

      <p>
        The pipeline is not just a sequence of tasks — it is a sequence of{" "}
        <strong>artifacts</strong>, each produced by one stage and consumed by the next. Raw HTML
        becomes a clean corpus, which becomes integer IDs, which (with an initialized network)
        becomes a base model, which becomes an assistant, which becomes a fast quantized checkpoint,
        which becomes a live API. Lose the thread of <em>what object each stage hands the next one</em>{" "}
        and the whole thing dissolves into jargon. Keep that thread and the field is simple.
      </p>

      <h2>Stage-by-stage recap</h2>
      <p>
        Here is the entire pipeline as a single interactive recap. Click through the ten production
        stages. For each one, the card shows the <strong>input → output artifact</strong>, the{" "}
        <strong>one decision that defines the stage</strong>, the <strong>technique</strong> you
        reach for, and the chapter where you learned it. The panel on the right accumulates: as you
        advance, watch the chain of artifacts you have assembled grow until, at the final stage, you
        are holding the whole thing — from a scraped web page to a token streaming back to a user.
      </p>

      <Figure
        n="25.2"
        title="The pipeline, recapped — click each stage"
        caption="Ten stages, ten artifacts. The right panel is your running inventory: every stage hands the next a concrete object. This is the book, end to end, in one figure."
      >
        <PipelineRecap />
      </Figure>

      <p>The same recap as a reference table — keep this as your one-page map of the field:</p>

      <table>
        <thead>
          <tr>
            <th>Stage</th>
            <th>Key artifact produced</th>
            <th>The key decision</th>
            <th>Chapter</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Data</strong></td>
            <td>A clean, deduplicated token corpus</td>
            <td>What to keep vs. discard (filter, dedup, decontaminate)</td>
            <td><a href="/chapter/data-curation" className="prose-link">Ch. 4</a></td>
          </tr>
          <tr>
            <td><strong>Tokenize</strong></td>
            <td>A vocabulary + integer ID sequences</td>
            <td>Vocab size & merges (or reuse a tokenizer)</td>
            <td><a href="/chapter/tokenization" className="prose-link">Ch. 5</a></td>
          </tr>
          <tr>
            <td><strong>Architecture</strong></td>
            <td>An initialized GPT (random weights)</td>
            <td>Depth, width, heads, context, positional scheme</td>
            <td><a href="/chapter/transformer-overview" className="prose-link">Ch. 6–11</a></td>
          </tr>
          <tr>
            <td><strong>Pretrain</strong></td>
            <td>A fluent base model</td>
            <td>Compute-optimal token/param split; LR schedule</td>
            <td><a href="/chapter/training-objective" className="prose-link">Ch. 12–17</a></td>
          </tr>
          <tr>
            <td><strong>Evaluate</strong></td>
            <td>A capability & safety scorecard</td>
            <td>Which benchmarks to trust; contamination</td>
            <td><a href="/chapter/evaluation" className="prose-link">Ch. 21</a></td>
          </tr>
          <tr>
            <td><strong>SFT</strong></td>
            <td>An instruction-following model</td>
            <td>Chat template; mask loss to responses only</td>
            <td><a href="/chapter/supervised-finetuning" className="prose-link">Ch. 18, 20</a></td>
          </tr>
          <tr>
            <td><strong>Align</strong></td>
            <td>A helpful, honest, harmless assistant</td>
            <td>Preference objective (PPO vs. DPO)</td>
            <td><a href="/chapter/rlhf" className="prose-link">Ch. 19</a></td>
          </tr>
          <tr>
            <td><strong>Optimize</strong></td>
            <td>A small, fast, cheap checkpoint</td>
            <td>Quality-vs-latency: quantization bits, caching</td>
            <td><a href="/chapter/inference-optimization" className="prose-link">Ch. 22</a></td>
          </tr>
          <tr>
            <td><strong>Deploy</strong></td>
            <td>A production token-streaming API</td>
            <td>Batching & autoscaling to hit SLOs</td>
            <td><a href="/chapter/deployment-serving" className="prose-link">Ch. 23</a></td>
          </tr>
          <tr>
            <td><strong>Monitor</strong></td>
            <td>A guarded, observed service</td>
            <td>What to log; guardrails; when to retrain</td>
            <td><a href="/chapter/monitoring-safety" className="prose-link">Ch. 24</a></td>
          </tr>
        </tbody>
      </table>

      <Callout type="note" title="Two halves, one network">
        <p>
          The dividing line runs right down the middle. Everything up to and including pretraining
          produces a <em>base model</em> — a brilliant autocomplete engine with no manners.
          Everything after (SFT, alignment) is <em>post-training</em>: the same weights, nudged into
          a helpful assistant. It is the single most important distinction in the field, and you now
          own both sides of it.
        </p>
      </Callout>

      <h2>The numbers</h2>
      <p>
        Abstractions are nice; orders of magnitude make it real. Let us cost out a{" "}
        <strong>small-but-real</strong> model — roughly a <M>{"1\\text{B}"}</M>-parameter GPT trained
        compute-optimally. These figures are deliberately approximate (hardware, prices and recipes
        move fast), but they are the right <em>ballpark</em>, and the relationships between them are
        exact.
      </p>
      <p>
        The anchor is the compute identity from{" "}
        <button onClick={() => navigate("/chapter/scaling-laws")} className="prose-link">
          Chapter 17
        </button>
        . The floating-point operations to train a dense transformer are approximately
      </p>
      <MB>{String.raw`C \;\approx\; 6\,N\,D`}</MB>
      <p>
        where <M>{"N"}</M> is the number of parameters and <M>{"D"}</M> is the number of training
        tokens (the 6 is ~2 FLOPs per multiply-add in the forward pass plus ~4 in the backward pass).
        The Chinchilla rule of thumb says spend your budget so that{" "}
        <M>{"D \\approx 20\\,N"}</M> — about twenty tokens per parameter. So for{" "}
        <M>{"N = 1{\\times}10^{9}"}</M>:
      </p>
      <MB>{String.raw`D \approx 20N = 2{\times}10^{10}\ \text{tokens}, \qquad C \approx 6ND \approx 1.2{\times}10^{20}\ \text{FLOPs}.`}</MB>
      <p>
        Now turn FLOPs into wall-clock and dollars. A modern training GPU delivers on the order of{" "}
        <M>{"3{\\times}10^{14}"}</M> useful FLOP/s in <code>bf16</code> at a realistic{" "}
        <em>model-FLOP utilization</em> of ~40%. Dividing:
      </p>

      <table>
        <thead>
          <tr>
            <th>Quantity</th>
            <th>Symbol</th>
            <th>Small-but-real value</th>
            <th>How it’s derived</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Parameters</td>
            <td><M>{"N"}</M></td>
            <td>≈ 1 billion</td>
            <td>Your architecture choice</td>
          </tr>
          <tr>
            <td>Training tokens</td>
            <td><M>{"D"}</M></td>
            <td>≈ 20 billion</td>
            <td><M>{"D \\approx 20N"}</M> (Chinchilla)</td>
          </tr>
          <tr>
            <td>Training compute</td>
            <td><M>{"C"}</M></td>
            <td>≈ 1.2 × 10²⁰ FLOPs</td>
            <td><M>{"C \\approx 6ND"}</M></td>
          </tr>
          <tr>
            <td>GPU-hours</td>
            <td>—</td>
            <td>≈ 1,000 GPU-hours</td>
            <td><M>{"C / (3{\\times}10^{14}\\cdot 0.4 \\cdot 3600)"}</M></td>
          </tr>
          <tr>
            <td>Cloud cost</td>
            <td>—</td>
            <td>≈ $2k–$5k</td>
            <td>≈ 1k GPU-hr × $2–4/GPU-hr</td>
          </tr>
          <tr>
            <td>Wall-clock</td>
            <td>—</td>
            <td>≈ 1–2 days on 8×GPU</td>
            <td>1k GPU-hr ÷ 8 GPUs</td>
          </tr>
        </tbody>
      </table>

      <p>
        Sit with that for a second: a model that would have been world-class in 2019 now costs a few
        thousand dollars and a weekend. That collapse is the whole reason this book can end with{" "}
        <em>“now go do it.”</em>
      </p>

      <Callout type="industry" title="Where the money actually goes">
        <p>
          Pretraining is ~99% of the training compute, but in a real org the <em>human</em> cost is
          inverted: data curation and the SFT/alignment data (high-quality human-written or
          human-rated examples) are where teams spend their time and budget. GPUs are cheap relative
          to good data. Frontier runs, for contrast, scale every number above by 4–6 orders of
          magnitude — <M>{"\\sim 10^{25}\\text{–}10^{26}"}</M> FLOPs and tens of millions of dollars
          for a single training run.
        </p>
      </Callout>

      <Callout type="pitfall" title="These are training FLOPs, not inference FLOPs">
        <p>
          The <M>{"6ND"}</M> figure is a one-time cost. <em>Inference</em> is forever: serving the
          model costs roughly <M>{"2N"}</M> FLOPs <em>per generated token</em>, multiplied by every
          token of every request from every user. A popular model can burn more compute on inference
          in a month than its entire training run — which is exactly why{" "}
          <a href="/chapter/inference-optimization" className="prose-link">Chapter 22</a> exists.
        </p>
      </Callout>

      <h2>Build your own</h2>
      <p>
        You do not start by writing a distributed training framework. You start by reproducing
        something that already works, then scale one axis at a time. Here is the concrete on-ramp
        — the path a sane person actually takes:
      </p>
      <ol>
        <li>
          <strong>Overfit a tiny model on a tiny dataset, today.</strong> Grab Andrej Karpathy’s{" "}
          <strong>nanoGPT</strong> (or the <code>model.py</code> you read in{" "}
          <button onClick={() => navigate("/chapter/building-the-model")} className="prose-link">
            Chapter 14
          </button>
          ) and train a character-level model on <strong>TinyShakespeare</strong> — ~1&nbsp;MB of
          text. It trains in <em>minutes</em> on a single GPU (even a laptop), and watching the loss
          fall while the samples turn from noise into pseudo-Shakespeare is the single best
          confidence-builder there is.
        </li>
        <li>
          <strong>Swap in real data and a real tokenizer.</strong> Move from characters to{" "}
          <strong>FineWeb-Edu</strong> (a clean, high-quality web subset) and a pretrained BPE
          tokenizer (e.g. GPT-2’s <code>tiktoken</code> vocab). Do <em>not</em> train your own
          tokenizer yet — reusing one removes a whole class of bugs and is what most teams do anyway.
        </li>
        <li>
          <strong>Scale the model, not your ambition.</strong> Bump <code>n_layer</code>,{" "}
          <code>n_embd</code> and <code>block_size</code> toward the <M>{"\\sim 100\\text{M}"}</M>{" "}
          range. Keep the Chinchilla ratio in mind: more parameters need proportionally more tokens
          (<M>{"D \\approx 20N"}</M>), or you are wasting compute.
        </li>
        <li>
          <strong>Rent GPUs; don’t buy them.</strong> Spin up a multi-GPU box on a cloud (Lambda,
          RunPod, Vast.ai, a hyperscaler) for the hours you need it. A few hundred dollars gets you a
          genuinely capable base model. Use <code>torchrun</code> for data-parallel training across
          the GPUs you rented.
        </li>
        <li>
          <strong>Post-train it.</strong> Fine-tune on an open instruction dataset with{" "}
          <strong>LoRA</strong> (fits on one GPU), then optionally align with <strong>DPO</strong> on
          a preference set. Now you have a chatbot, not just an autocomplete.
        </li>
        <li>
          <strong>Serve and chat.</strong> Load the checkpoint into <strong>vLLM</strong>, hit it
          with the OpenAI-compatible API, and talk to the thing you built from scratch.
        </li>
      </ol>

      <Callout type="tip" title="Getting started: the 30-minute version">
        <p>
          Do not read the whole roadmap and freeze. Tonight: <code>git clone</code> nanoGPT, run its
          TinyShakespeare prepare-and-train script, and watch the samples improve. That single loop —{" "}
          <em>prepare data → train → sample</em> — is the entire field in miniature. Everything else
          is making each of those three steps bigger and better.
        </p>
      </Callout>

      <p>
        Below is a realistic end-to-end script sketch that strings the whole pipeline together,
        reusing the <code>model.py</code> (the <code>GPT</code> + <code>GPTConfig</code> classes) you
        built in Chapter 14. It is a skeleton — the kind of <code>run.sh</code> you would actually
        keep in a repo — not a turnkey program, but every step maps directly to a chapter you have
        read.
      </p>

      <CodeBlock language="bash" filename="run.sh" highlight={[10, 19, 30, 41]}>
{`#!/usr/bin/env bash
# End-to-end: a tiny-but-real LLM, from raw text to a chat reply.
# Each stage maps to a chapter of this book.
set -euo pipefail

# ── 0. Setup ───────────────────────────────────────────────────────────────
pip install torch numpy tiktoken datasets vllm trl peft
N_GPU=8

# ── 1–2. DATA + TOKENIZE  (Ch.4–5) ─────────────────────────────────────────
# Download a clean web subset and pre-tokenize it to a flat uint16 .bin file
# with an existing BPE tokenizer (no need to train your own).
python prepare_data.py \\
    --dataset HuggingFaceFW/fineweb-edu --subset sample-10BT \\
    --tokenizer gpt2 \\
    --out data/fineweb/            # -> train.bin, val.bin  (token-ID streams)

# ── 3–4. ARCHITECTURE + PRETRAIN  (Ch.6–17) ────────────────────────────────
# train.py imports GPT, GPTConfig from the model.py you wrote in Chapter 14,
# then runs the AdamW + cosine-LR loop over data/fineweb until val loss plateaus.
torchrun --standalone --nproc_per_node=$N_GPU train.py \\
    --data_dir data/fineweb \\
    --n_layer 24 --n_head 16 --n_embd 1024 --block_size 1024 \\   # ~1B params
    --batch_tokens 500000 --max_tokens 20_000_000_000 \\          # D ~= 20N
    --lr 3e-4 --warmup 2000 --grad_clip 1.0 \\
    --out_dir ckpt/base/           # -> ckpt/base/model.pt  (the base model)

# ── 5. EVALUATE  (Ch.21) ───────────────────────────────────────────────────
python eval.py --ckpt ckpt/base/model.pt --tasks perplexity,hellaswag

# ── 6. SFT  (Ch.18, 20)  — teach it to follow instructions, with LoRA ───────
python sft.py \\
    --base ckpt/base/model.pt \\
    --dataset HuggingFaceH4/no_robots \\   # instruction -> response pairs
    --lora_r 16 --epochs 3 \\
    --out_dir ckpt/sft/

# ── 7. ALIGN  (Ch.19)  — DPO on preference comparisons (optional) ───────────
python dpo.py --base ckpt/sft/ --dataset HuggingFaceH4/ultrafeedback_binarized \\
    --out_dir ckpt/aligned/

# ── 8–9. OPTIMIZE + DEPLOY  (Ch.22–23) ─────────────────────────────────────
# Quantize and serve behind an OpenAI-compatible API with continuous batching.
python -m vllm.entrypoints.openai.api_server \\
    --model ckpt/aligned/ --quantization awq --port 8000 &

# ── chat  — talk to the model you built from scratch ───────────────────────
curl http://localhost:8000/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model": "ckpt/aligned/",
         "messages": [{"role": "user", "content": "Explain attention in one line."}]}'`}
      </CodeBlock>

      <p>
        Read those forty lines top to bottom: it is the table of contents of this book, made
        executable. <code>prepare_data.py</code> is Part II; <code>train.py</code> imports the model
        you wrote in Part IV; <code>sft.py</code> and <code>dpo.py</code> are Part V; the{" "}
        <code>vllm</code> server is Part VI. Nothing in it should surprise you anymore.
      </p>

      <h2>What we didn’t cover — the frontiers</h2>
      <p>
        This book taught the durable core: the parts of LLMs that have been stable for years and
        will still be true in five. But the field has a moving edge. Here are the directions worth
        chasing next, each a brief pointer rather than a chapter:
      </p>
      <ul>
        <li>
          <strong>Mixture-of-Experts (MoE).</strong> Replace each dense feed-forward layer with many
          “expert” MLPs and a router that activates only a few per token. You get the capacity of a
          huge model at the inference cost of a small one — the architecture behind several frontier
          models.
        </li>
        <li>
          <strong>Multimodality.</strong> Feed images, audio and video into the same transformer by
          encoding them into the token stream (e.g. vision encoders → patch tokens). The
          next-token objective barely changes; the inputs get richer.
        </li>
        <li>
          <strong>Long context.</strong> Pushing context windows from thousands to <em>millions</em>{" "}
          of tokens, fighting the <M>{"O(n^2)"}</M> attention cost with RoPE scaling, ring/streaming
          attention, and state-space models (Mamba) that grow linearly.
        </li>
        <li>
          <strong>Tool use & agents.</strong> Let the model call functions, search, run code, and act
          in loops. This turns a text generator into a system that <em>does</em> things — and opens a
          whole discipline of orchestration, memory and safety.
        </li>
        <li>
          <strong>Reasoning & RL.</strong> Train models to “think” in long chains-of-thought before
          answering, rewarded by RL on verifiable outcomes (math, code). A large reasoning gain has
          come from spending compute at <em>inference</em> time, not just training time.
        </li>
        <li>
          <strong>Retrieval (RAG).</strong> Instead of cramming all knowledge into weights, fetch
          relevant documents at query time and condition on them. Cheaper to update, easier to cite,
          and a natural fix for staleness and hallucination.
        </li>
      </ul>

      <Callout type="industry" title="The core is the leverage">
        <p>
          Every frontier above is built <em>on top of</em> what you already understand. MoE is a
          swapped-out MLP. Multimodality is different tokens. Agents are a generation loop with tools
          in it. Reasoning is RLHF pointed at verifiable rewards. You are not behind — you have the
          foundation that makes all of these readable.
        </p>
      </Callout>

      <h2>Further resources</h2>
      <p>
        A short, high-signal reading list. No links — just names and authors you can search,
        chosen because each one rewards the time:
      </p>
      <ul>
        <li>
          <strong>nanoGPT</strong> (Andrej Karpathy) — the cleanest from-scratch GPT training repo;
          your launchpad. Pair it with his <em>“Let’s build GPT”</em> and <em>“Zero to Hero”</em>{" "}
          video lectures.
        </li>
        <li>
          <strong>“Attention Is All You Need”</strong> (Vaswani et al., 2017) — the Transformer
          paper. Read it now; it will finally make complete sense.
        </li>
        <li>
          <strong>“Language Models are Few-Shot Learners”</strong> (Brown et al., 2020) — the GPT-3
          paper that established in-context learning and scale.
        </li>
        <li>
          <strong>“Training Compute-Optimal Large Language Models”</strong> (Hoffmann et al., 2022) —
          the Chinchilla paper behind the <M>{"D \\approx 20N"}</M> rule you used above.
        </li>
        <li>
          <strong>“LoRA: Low-Rank Adaptation of Large Language Models”</strong> (Hu et al., 2021) —
          the low-rank fine-tuning trick that puts billion-parameter models on one GPU.
        </li>
        <li>
          <strong>“Direct Preference Optimization”</strong> (Rafailov et al., 2023) — alignment
          without a separate reward model or RL loop. Plus <strong>“InstructGPT”</strong>{" "}
          (Ouyang et al., 2022) for the original RLHF recipe.
        </li>
        <li>
          <strong>The Hugging Face LLM Course</strong> and <strong>Transformers</strong> /{" "}
          <strong>TRL</strong> libraries — the practical, hands-on path from notebook to deployed
          model.
        </li>
      </ul>

      <Callout type="note" title="Read papers like code">
        <p>
          You can now read these primary sources the way you read a function: skim the signature
          (abstract + figures), trace the hot path (method), then check the tests (results). The
          jargon that walls papers off from newcomers — <em>logits, KV cache, RoPE, cross-entropy,
          KL penalty</em> — is now just your vocabulary.
        </p>
      </Callout>

      <h2>Where you go from here</h2>
      <p>
        You started this book unable to read a single line of a transformer. You can now read the
        whole stack — and, more importantly, you understand <em>why</em> each piece is there. The
        attention formula is not a spell anymore; it is a soft dictionary lookup. The training loop
        is not magic; it is the chain rule, repeated. Alignment is not hand-waving; it is a loss
        function pointed at human preferences. You closed every box in Figure 25.1.
      </p>
      <p>
        That understanding is rarer and more durable than it feels right now. Architectures will
        churn, benchmarks will inflate, and this year’s frontier model will be next year’s footnote
        — but the load-bearing ideas you have internalized (predict the next token; attend to what
        matters; descend the gradient; learn from data, then from preferences) are the bedrock the
        whole field is built on. They will not go out of date.
      </p>

      <Callout type="key" title="Go build the thing">
        <p>
          The gap between reading about LLMs and building one is smaller than it has ever been — a
          weekend, a few thousand dollars of GPU time, and the knowledge you now have. So close this
          tab, open a terminal, clone nanoGPT, and watch a model you trained start to write. The best
          way to honor everything you just learned is to make a machine talk.
        </p>
      </Callout>

      <p className="text-slate-400">
        Thank you for reading <em>The LLM School</em>. Now go predict some tokens. 🚀
      </p>
    </>
  );
}
