import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LineChart from "../components/viz/LineChart";
import { mulberry32 } from "../components/viz/scales";
import GradAccum from "../components/viz/training-loop/GradAccum";

// A deterministic, realistic-looking train/val loss curve.
// Train: fast early drop, warmup wiggle, then a long slow decay toward a floor.
// Val: tracks train but sits slightly above it (the generalization gap) and is
// noisier because it's measured on held-out data at intervals.
function makeLossCurves() {
  const rng = mulberry32(7);
  const train = [];
  const val = [];
  const totalSteps = 6000;
  const warmup = 400;
  const floor = 1.45; // irreducible-ish loss for this toy run
  const start = 6.2;
  for (let s = 0; s <= totalSteps; s += 100) {
    // smooth exponential-ish decay toward the floor
    const decay = Math.exp(-s / 1500);
    let base = floor + (start - floor) * decay;
    // warmup wiggle: a small bump while LR ramps up, settling after `warmup`
    if (s < warmup) base += 0.55 * (1 - s / warmup);
    else if (s < warmup * 2) base += 0.18 * (1 - (s - warmup) / warmup);
    // a tiny LR-decay kink near the end where the schedule cools off
    if (s > 4500) base -= 0.06 * ((s - 4500) / 1500);
    const noise = (rng() - 0.5) * 0.05;
    train.push([s, +(base + noise).toFixed(3)]);
    // validation: measured every 500 steps, slightly higher + noisier
    if (s % 500 === 0) {
      const gap = 0.08 + 0.04 * (s / totalSteps); // gap widens slowly
      const vnoise = (rng() - 0.5) * 0.09;
      val.push([s, +(base + gap + vnoise).toFixed(3)]);
    }
  }
  return { train, val, warmup, totalSteps };
}

export default function Chapter() {
  const { train, val, warmup, totalSteps } = makeLossCurves();

  return (
    <>
      <p>
        In <a href="/chapter/building-the-model" className="prose-link">Chapter 14</a> you assembled a
        complete GPT — embeddings, attention, MLPs, the language-model head. It is a function that
        maps a batch of token indices to a batch of next-token probability distributions, and it is
        full of randomly initialized weights that, right now, produce gibberish. This chapter is the
        engine room: the <strong>training loop</strong> that turns those random weights into a model
        that has learned language. The core is only five lines — forward, loss, backward, step,
        zero — but a production loop wraps that core in batching, gradient accumulation, mixed
        precision, checkpointing, and logging. We’ll build all of it, and learn to read the loss
        curve that tells you whether it’s working.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          Training is a loop that repeatedly: (1) runs a batch <strong>forward</strong> to get a
          loss, (2) runs <strong>backward</strong> to get gradients of that loss w.r.t. every weight,
          (3) takes one <strong>optimizer step</strong> nudging weights down-gradient, and (4){" "}
          <strong>zeroes the gradients</strong> so the next batch starts clean. Everything else —
          accumulation, AMP, checkpoints — is plumbing around those four moves.
        </p>
      </Callout>

      <h2>The five-line core</h2>
      <p>
        Strip away every optimization and a training step is this:
      </p>
      <CodeBlock language="python">
{`logits = model(x)                       # 1. forward
loss = F.cross_entropy(logits.view(-1, V), y.view(-1))   # the objective
loss.backward()                         # 2. backward: fill .grad on every parameter
optimizer.step()                        # 3. step: weights -= lr * (adjusted grad)
optimizer.zero_grad(set_to_none=True)   # 4. reset grads for the next step`}
      </CodeBlock>
      <p>
        Step 1 computes predictions. The cross-entropy loss — derived in{" "}
        <a href="/chapter/language-modeling" className="prose-link">Chapter 3</a> — measures how
        surprised the model is by the true next tokens <M>{"y"}</M>. Step 2,{" "}
        <code>loss.backward()</code>, runs reverse-mode autodiff (the backpropagation from{" "}
        <a href="/chapter/neural-network-primer" className="prose-link">Chapter 2</a>) and deposits a
        gradient tensor into the <code>.grad</code> field of every parameter. Step 3 hands those
        gradients to the optimizer — AdamW, from{" "}
        <a href="/chapter/optimization" className="prose-link">Chapter 13</a> — which updates the
        weights. Step 4 clears the gradients.
      </p>

      <Callout type="pitfall" title="Forgetting zero_grad() is the #1 silent bug">
        <p>
          PyTorch <em>accumulates</em> gradients: each <code>backward()</code> call <em>adds</em> to{" "}
          <code>.grad</code> rather than overwriting it. If you forget{" "}
          <code>optimizer.zero_grad()</code>, every step uses the sum of all gradients seen so far —
          your effective learning rate explodes and training diverges, with no error message.
          (We’ll see that this accumulation is a <em>feature</em> we deliberately exploit for large
          batches — but you must control it.) Prefer{" "}
          <code>set_to_none=True</code>: it frees the gradient buffers and is slightly faster than
          zero-filling.
        </p>
      </Callout>

      <h2>Data: the loader and batching</h2>
      <p>
        The model trains on a giant stream of token IDs (from the tokenizer in{" "}
        <a href="/chapter/tokenization" className="prose-link">Chapter 5</a>) packed into one long
        array on disk. A training example is just a contiguous window of length{" "}
        <M>{"T"}</M> (the <strong>block size</strong> / context length): the inputs are tokens{" "}
        <M>{"[i, i+T)"}</M> and the targets are the same window shifted by one,{" "}
        <M>{"[i+1, i+T+1)"}</M>. A <strong>batch</strong> stacks <M>{"B"}</M> such windows so the
        GPU works on many sequences at once.
      </p>
      <CodeBlock language="python" filename="data.py">
{`import numpy as np, torch

# tokens.bin is one flat uint16 array of token IDs for the whole corpus
data = np.memmap("tokens.bin", dtype=np.uint16, mode="r")  # lazy, never loaded fully

def get_batch(split, B, T, device):
    d = data if split == "train" else val_data
    # B random start positions; each yields a T-length window
    ix = torch.randint(len(d) - T, (B,))
    x = torch.stack([torch.from_numpy(d[i      : i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1  : i + 1 + T].astype(np.int64)) for i in ix])
    # pin + non_blocking lets the H2D copy overlap with compute
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)`}
      </CodeBlock>
      <p>
        For pretraining, sampling random windows from a <code>memmap</code> is both simple and
        effectively shuffles the data — you never load the whole corpus into RAM. For fine-tuning on
        structured examples you’d instead use a <code>Dataset</code> +{" "}
        <code>DataLoader</code> with <code>num_workers</code> to prefetch on background threads, so
        the GPU is never starved waiting for I/O.
      </p>

      <Callout type="note" title="Keep the GPU fed">
        <p>
          A common, invisible performance killer is an input pipeline that can’t keep up: the
          (expensive) GPU sits idle waiting for the (cheap) CPU to assemble the next batch. Symptoms
          are low GPU utilization in <code>nvidia-smi</code>. Fixes: more{" "}
          <code>DataLoader</code> workers, <code>pin_memory=True</code>,{" "}
          <code>non_blocking=True</code> transfers, and pre-tokenizing the dataset so no tokenization
          happens in the hot loop.
        </p>
      </Callout>

      <h2>Gradient accumulation: a big batch on a small GPU</h2>
      <p>
        Large language models train best with <em>large</em> batches — hundreds of thousands of
        tokens per step — for a smoother, less noisy gradient estimate. But a big batch may not fit
        in GPU memory: activations scale with <M>{"B \\times T"}</M>. The trick is to exploit the
        very gradient-accumulation behavior that bit us above. Split the desired batch into{" "}
        <M>{"N"}</M> smaller <strong>micro-batches</strong>, run forward+backward on each (gradients
        sum into <code>.grad</code>), and only call <code>optimizer.step()</code> after all{" "}
        <M>{"N"}</M>. The optimizer sees one update computed from the whole effective batch, but you
        never held more than one micro-batch in memory at a time.
      </p>
      <p>The effective (global) batch size is simply the product:</p>
      <MB>{String.raw`B_{\text{eff}} \;=\; B_{\text{micro}} \,\times\, N_{\text{accum}} \,\times\, N_{\text{GPU}}`}</MB>
      <p>
        One subtlety: because cross-entropy <em>averages</em> over the tokens in a micro-batch,
        summing <M>{"N"}</M> micro-batch gradients would over-count by a factor of <M>{"N"}</M>. So
        you scale each micro-batch loss by <M>{"1/N"}</M> before <code>backward()</code>, making the
        accumulated gradient the correct average over the full effective batch.
      </p>

      <Figure
        n="15.1"
        title="Gradient accumulation, stepped"
        caption="Pick a micro-batch size, an accumulation count, and a GPU count. Step through one optimizer cycle: micro-batches stream in, their gradients sum into a single buffer, and the optimizer fires exactly once at the end. The effective batch size is the product of all three knobs — but peak memory is set only by the micro-batch."
      >
        <GradAccum />
      </Figure>

      <h2>Mixed precision: bf16 / fp16 + the GradScaler</h2>
      <p>
        By default PyTorch uses 32-bit floats (fp32). But matmuls run far faster — and use half the
        memory — in 16-bit, and modern GPUs have dedicated tensor cores for it.{" "}
        <strong>Automatic mixed precision (AMP)</strong> runs the heavy ops (matmuls, convolutions)
        in 16-bit while keeping numerically sensitive ops (softmax, layernorm, the loss reduction) in
        fp32, all decided for you by <code>torch.autocast</code>.
      </p>
      <p>There are two 16-bit formats, and the difference matters:</p>
      <ul>
        <li>
          <strong>fp16</strong> — 10 mantissa bits, 5 exponent bits. More precision but a{" "}
          <em>tiny</em> dynamic range; small gradients <em>underflow to zero</em>. It therefore needs
          a <strong>GradScaler</strong>: multiply the loss by a large factor before{" "}
          <code>backward()</code> to push gradients into representable range, then unscale before the
          step (and skip steps where gradients overflowed to <code>inf</code>/<code>NaN</code>).
        </li>
        <li>
          <strong>bf16</strong> — 7 mantissa bits but the <em>same 8 exponent bits as fp32</em>. Less
          precision, but the full dynamic range means gradients don’t underflow — so{" "}
          <strong>bf16 needs no GradScaler</strong>. On Ampere/Hopper GPUs (A100, H100) bf16 is the
          default choice for LLM training for exactly this reason.
        </li>
      </ul>

      <Callout type="industry" title="bf16 is the modern default">
        <p>
          Essentially all large-scale LLM pretraining today uses bf16 (or fp8 on the newest
          hardware), precisely because it sidesteps the loss-scaling fragility of fp16. Reach for
          fp16 + GradScaler only on older GPUs (e.g. V100, T4) that lack bf16 tensor cores. The code
          below supports both: it enables the scaler only when the dtype is fp16.
        </p>
      </Callout>

      <h2>Gradient checkpointing: trade compute for memory</h2>
      <p>
        Activations — the intermediate tensors saved during the forward pass so backward can use
        them — often dominate memory for long sequences. <strong>Gradient checkpointing</strong>{" "}
        (a.k.a. activation checkpointing) saves only a few activations and <em>recomputes</em> the
        rest during backward. You pay roughly one extra forward pass (≈30% more compute) to cut
        activation memory dramatically — often enough to fit a longer context or a larger
        micro-batch. In PyTorch it’s <code>torch.utils.checkpoint.checkpoint(block, x)</code> wrapped
        around each transformer block. Use it when memory-bound, not when compute-bound.
      </p>

      <h2>Putting it together: a real training loop</h2>
      <p>
        Here is a loop you could actually run. It combines accumulation, autocast, the conditional
        GradScaler, gradient clipping, an LR schedule, periodic evaluation, and checkpointing. Read
        it top to bottom — every line earns its place.
      </p>
      <CodeBlock language="python" filename="train.py" highlight={[28, 29, 33, 36, 39]}>
{`import torch, torch.nn.functional as F, math

# ---- hyperparameters ----
B, T          = 12, 1024          # micro-batch size, block (context) size
accum_steps   = 40                # → effective batch = 12 * 40 = 480 sequences
max_steps     = 6000
warmup        = 400
max_lr, min_lr = 6e-4, 6e-5
grad_clip     = 1.0
eval_every    = 500
dtype         = torch.bfloat16    # or torch.float16 on older GPUs
device        = "cuda"

model = GPT(cfg).to(device)
model = torch.compile(model)      # fuse kernels; big speedup on recent PyTorch
opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95),
                        weight_decay=0.1, fused=True)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

def lr_at(step):                                  # linear warmup → cosine decay
    if step < warmup:
        return max_lr * (step + 1) / warmup
    r = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * r))

for step in range(max_steps):
    # ---- 1. set this step's learning rate from the schedule ----
    for g in opt.param_groups: g["lr"] = lr_at(step)

    # ---- 2. accumulate gradients over accum_steps micro-batches ----
    opt.zero_grad(set_to_none=True)
    for micro in range(accum_steps):
        x, y = get_batch("train", B, T, device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / accum_steps                 # scale so grads = true average
        scaler.scale(loss).backward()             # sums into .grad; no-op scale for bf16

    # ---- 3. clip, step, update the scaler ----
    scaler.unscale_(opt)                          # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt)                              # skips the step if grads were inf/NaN
    scaler.update()

    # ---- 4. periodic eval + checkpoint ----
    if step % eval_every == 0:
        val_loss = estimate_loss(model, "val", B, T, device)
        print(f"step {step:5d} | train {loss.item()*accum_steps:.3f} | val {val_loss:.3f}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step}, "ckpt.pt")`}
      </CodeBlock>
      <p>
        A few details worth internalizing: <code>zero_grad</code> moves <em>outside</em> the inner
        loop (so micro-batch gradients accumulate); the loss is divided by{" "}
        <code>accum_steps</code> before backward; <code>scaler.unscale_</code> must come before
        clipping so you clip the <em>true</em> gradient norm, not the scaled one; and{" "}
        <code>scaler.step</code> automatically skips an update if gradients overflowed. With{" "}
        <code>dtype=bfloat16</code> the scaler is a no-op, so this same code is correct in both
        regimes.
      </p>

      <Callout type="warning" title="Loss spikes and how to survive them">
        <p>
          Occasional sudden jumps in the loss are normal at scale — a bad batch, a rare token, or
          numerical instability. <strong>Gradient clipping</strong> (<code>clip_grad_norm_</code>)
          is your seatbelt: it rescales the gradient so its global norm never exceeds a threshold
          (typically 1.0), preventing one pathological batch from blowing up the weights. If a spike
          does derail training (loss → <code>NaN</code> and stays there), the standard recovery is to{" "}
          <em>resume from the last good checkpoint</em>, optionally skipping the offending data
          batch — which is exactly why we checkpoint frequently.
        </p>
      </Callout>

      <h2>Checkpointing &amp; resuming</h2>
      <p>
        A real run lasts hours, days, or weeks; machines crash. A <strong>checkpoint</strong> must
        capture enough state to resume <em>bit-for-bit</em>, which is more than just the weights. You
        need: the model <code>state_dict</code>, the optimizer <code>state_dict</code> (AdamW’s
        first- and second-moment buffers — losing these resets the optimizer and causes a visible
        loss bump), the current <code>step</code>, the GradScaler state, and the RNG states so data
        sampling continues deterministically.
      </p>
      <CodeBlock language="python">
{`# --- save (periodically) ---
torch.save({
    "model":  model.state_dict(),
    "opt":    opt.state_dict(),     # ← Adam moments live here; don't drop them
    "scaler": scaler.state_dict(),
    "step":   step,
    "rng":    torch.get_rng_state(),
}, "ckpt.pt")

# --- resume ---
ck = torch.load("ckpt.pt", map_location=device)
model.load_state_dict(ck["model"])
opt.load_state_dict(ck["opt"])
scaler.load_state_dict(ck["scaler"])
torch.set_rng_state(ck["rng"])
start_step = ck["step"] + 1         # continue the loop from here`}
      </CodeBlock>

      <h2>Reading the loss curve</h2>
      <p>
        The loss curve is your single most important instrument. Below is a realistic run. The{" "}
        <span style={{ color: "#22d3ee" }}>train</span> loss falls fast at first, shows a small{" "}
        <strong>warmup wiggle</strong> while the learning rate ramps up, then settles into a long,
        slow decay toward a floor. The <span style={{ color: "#fb7185" }}>validation</span> loss
        (measured on held-out data every 500 steps) tracks just <em>above</em> train — that small,
        stable gap is the healthy <strong>generalization gap</strong>.
      </p>

      <Figure
        n="15.2"
        title="A healthy train vs. validation loss curve"
        caption="Hover to read off values. The warmup wiggle near the start is the LR ramp settling; the late dip is the cosine schedule cooling the LR. A small, stable val gap is good. If val starts rising while train keeps falling, you are overfitting."
      >
        <LineChart
          height={320}
          xLabel="training step"
          yLabel="loss (nats / token)"
          series={[
            { label: "train", color: "#22d3ee", points: train },
            { label: "val", color: "#fb7185", points: val, dashed: true },
          ]}
          annotations={[
            { x: warmup, label: "warmup ends", color: "#f59e0b" },
            { x: 4500, label: "LR decay", color: "#a855f7" },
          ]}
          fmtX={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0))}
          fmtY={(v) => v.toFixed(2)}
        />
      </Figure>

      <p>What to look for when you read a curve:</p>
      <ul>
        <li>
          <strong>Loss not dropping at all</strong> → learning rate too low, a bug in the data
          (inputs/targets misaligned), or the loss not connected to the graph. Run the overfit test
          below.
        </li>
        <li>
          <strong>Loss → NaN / spikes to the moon</strong> → learning rate too high, missing
          gradient clipping, or fp16 overflow. Lower the LR, add clipping, switch to bf16.
        </li>
        <li>
          <strong>Train keeps falling, val turns and rises</strong> → overfitting. With enough data
          this is rare in pretraining; it shows up in fine-tuning. Add data, regularization, or stop
          earlier.
        </li>
        <li>
          <strong>A sudden step-change up after a restart</strong> → you resumed without the
          optimizer state (or RNG), so Adam’s moments reset.
        </li>
      </ul>

      <h2>The overfit-one-batch sanity check</h2>
      <p>
        Before launching an expensive run, prove the machinery works with the single most valuable
        test in deep learning: <strong>overfit one batch</strong>. Take exactly one batch and train
        on it over and over with no regularization. A correct model+loss+optimizer{" "}
        <em>must</em> be able to memorize it — the loss should plummet toward zero. If it can’t, you
        have a bug (a detached graph, misaligned targets, a frozen parameter, a wrong reduction), and
        you’ve found it in seconds on one batch instead of after hours on the full dataset.
      </p>
      <CodeBlock language="python" filename="overfit_test.py">
{`# Sanity check: a healthy model MUST be able to memorize one batch.
x, y = get_batch("train", B=4, T=64, device=device)   # ONE fixed batch
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(200):
    logits = model(x)                                 # same x, y every iteration
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if i % 20 == 0:
        print(f"iter {i:3d}  loss {loss.item():.4f}")

# EXPECT: loss falls from ~log(vocab_size) toward ~0.
#   start ≈ ln(50257) ≈ 10.8   →   after 200 iters ≈ 0.01
# If it plateaus high, your model/loss/data pipeline has a bug. Fix it HERE.`}
      </CodeBlock>

      <Callout type="tip" title="A debugging ladder">
        <p>
          When something’s wrong, climb this ladder: (1) overfit one batch — does loss → 0? (2) check
          the initial loss equals <M>{"\\ln(V)"}</M> for vocab size <M>{"V"}</M> (a well-initialized
          model is maximally uncertain, so cross-entropy ≈ <M>{"\\ln V"}</M> at step 0). (3) verify
          inputs and targets are shifted by exactly one. (4) confirm gradients are non-zero
          (<code>p.grad.abs().mean()</code>) for every parameter. Each rung catches a whole class of
          bugs.
        </p>
      </Callout>

      <h2>The hyperparameters that actually matter</h2>
      <p>
        You can tune dozens of knobs, but three dominate everything:
      </p>
      <ul>
        <li>
          <strong>Learning rate</strong> — the single most important hyperparameter (see{" "}
          <a href="/chapter/optimization" className="prose-link">Chapter 13</a>). Too high diverges;
          too low wastes compute. The warmup+cosine schedule in the loop above is the standard recipe.
        </li>
        <li>
          <strong>Effective batch size</strong> (<M>{"B_{\\text{eff}}"}</M>) — bigger batches give
          smoother gradients and better hardware utilization, but with diminishing returns and a
          coupling to the LR (larger batches generally want a larger LR). Gradient accumulation lets
          you pick <M>{"B_{\\text{eff}}"}</M> independently of your GPU memory.
        </li>
        <li>
          <strong>Block size</strong> <M>{"T"}</M> (context length) — how many tokens of context the
          model sees. Longer context is more capable but costs <M>{"O(T^2)"}</M> in attention (see{" "}
          <a href="/chapter/self-attention" className="prose-link">Chapter 8</a>) and more activation
          memory.
        </li>
      </ul>

      <h2>Bridge: from one GPU to a thousand</h2>
      <p>
        You can now train a model end to end on a single GPU, watch its loss curve descend, and
        resume after a crash. But the models that matter are too big for one GPU — their weights,
        gradients, and optimizer state won’t fit, and a single device would take years to see enough
        tokens. The next chapter,{" "}
        <a href="/chapter/distributed-training" className="prose-link">Scaling Up: Distributed
        Training</a>, takes this exact loop and spreads it across many GPUs: splitting the batch
        (data parallelism), splitting the matmuls (tensor parallelism), splitting the layers
        (pipeline parallelism), and sharding the optimizer state (FSDP/ZeRO) — all stitched together
        by collective communication.
      </p>
    </>
  );
}
