import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import ParallelismViz from "../components/viz/distributed-training/ParallelismViz";

export default function Chapter() {
  return (
    <>
      <p>
        The training loop from{" "}
        <a href="/chapter/training-loop" className="prose-link">Chapter 15</a> assumes everything
        fits on one GPU. For real LLMs it doesn’t — not even close. A 70-billion-parameter model
        needs over a terabyte of GPU memory just for its training state, and a single device would
        take decades to process trillions of tokens. The solution is to split the work across many
        GPUs. There are four distinct things you can split — the <em>batch</em>, the{" "}
        <em>tensors</em>, the <em>layers</em>, and the <em>optimizer state</em> — and the art of
        large-scale training is combining them. This chapter builds the full picture, starting from
        the one number that forces the whole discussion: the memory budget.
      </p>

      <Callout type="key" title="Why we shard">
        <p>
          Three things grow with model size and must live in GPU memory: the{" "}
          <strong>model state</strong> (params + gradients + optimizer state), the{" "}
          <strong>activations</strong> (intermediate tensors for backprop), and the data flowing
          through. When any of these exceeds one GPU, you must <em>distribute</em> it. The strategies
          below differ only in <em>what</em> they split and <em>what they must communicate</em> to
          stitch the result back together.
        </p>
      </Callout>

      <h2>The memory budget: the “16 bytes/param” rule</h2>
      <p>
        Before choosing a parallelism strategy, do the arithmetic. For mixed-precision AdamW
        training (the standard recipe from{" "}
        <a href="/chapter/training-loop" className="prose-link">Chapter 15</a>), each parameter
        requires <em>five</em> copies of itself in different precisions:
      </p>
      <ul>
        <li>
          <strong>bf16 weights</strong> — the working copy used in the forward/backward pass:{" "}
          <MB>{String.raw`2\ \text{bytes}`}</MB>
        </li>
        <li>
          <strong>bf16 gradients</strong> — one per weight, produced by backward: <M>{"2"}</M> bytes
        </li>
        <li>
          <strong>fp32 master weights</strong> — a high-precision copy the optimizer actually
          updates (small bf16 updates would otherwise be lost to rounding): <M>{"4"}</M> bytes
        </li>
        <li>
          <strong>fp32 Adam first moment</strong> <M>{"m"}</M> (the momentum): <M>{"4"}</M> bytes
        </li>
        <li>
          <strong>fp32 Adam second moment</strong> <M>{"v"}</M> (the variance): <M>{"4"}</M> bytes
        </li>
      </ul>
      <p>That sums to a famous constant:</p>
      <MB>{String.raw`\underbrace{2}_{\text{bf16 } w} + \underbrace{2}_{\text{bf16 } \nabla} + \underbrace{4}_{\text{fp32 } w} + \underbrace{4}_{\text{Adam } m} + \underbrace{4}_{\text{Adam } v} \;=\; 16 \ \frac{\text{bytes}}{\text{param}}`}</MB>
      <p>
        Adam alone is responsible for <M>{"2\\times"}</M> the parameter count in optimizer state
        (<M>{"m"}</M> and <M>{"v"}</M>). So a model with <M>{"P"}</M> parameters needs roughly{" "}
        <M>{"16P"}</M> bytes of <em>model state</em> — before a single activation. For a 7B model
        that’s <M>{"16 \\times 7\\times 10^9 \\approx 112"}</M> GiB, already over a single 80 GiB
        H100. For 70B it’s ~1.1 TiB. <strong>This single calculation is why distributed training
        exists.</strong>
      </p>

      <Callout type="math" title="Don't forget activations">
        <p>
          The <M>{"16P"}</M> rule covers only model state. <strong>Activations</strong> — the tensors
          saved during forward for the backward pass — scale with{" "}
          <M>{"B \\times T \\times L \\times d_{\\text{model}}"}</M> (batch, sequence, layers, width)
          and can rival or exceed model state for long contexts. Gradient checkpointing (from{" "}
          <a href="/chapter/training-loop" className="prose-link">Chapter 15</a>) trades compute to
          shrink this term. Unlike model state, activation memory scales with batch size, so it’s the
          part you most directly control.
        </p>
      </Callout>

      <p>
        The interactive budget below lets you dial the parameter count and watch the per-GPU VRAM
        stack up, segment by segment, against the 80 GiB line of a single H100. Toggle ZeRO/FSDP
        sharding to see model state divide across GPUs (activations don’t shard — they live where the
        compute happens).
      </p>

      <Figure
        n="16.1"
        title="The per-GPU memory budget"
        caption="Drag the parameter count. Every bf16 model needs ~16 bytes/param of model state under mixed-precision AdamW: 2 (weights) + 2 (grads) + 4 (fp32 master) + 4 (Adam m) + 4 (Adam v). Sharding (ZeRO-3 / FSDP) divides model state across GPUs; activations stay put. Past the dashed 80 GiB line, the model no longer fits on one GPU."
      >
        <ParallelismViz variant="memory" />
      </Figure>

      <h2>Data parallelism (DDP): replicate, split, all-reduce</h2>
      <p>
        The simplest strategy, and the one you reach for first. Every GPU holds a{" "}
        <strong>full copy</strong> of the model. You split the global batch into one shard per GPU;
        each GPU runs forward and backward on <em>its</em> data independently, producing its own
        gradients. Then comes the only communication step: an <strong>all-reduce</strong> averages
        the gradients across all GPUs, so every replica applies the identical, batch-averaged update
        and the copies stay in lockstep.
      </p>
      <MB>{String.raw`g \;=\; \frac{1}{N}\sum_{i=1}^{N} g_i \qquad \text{(every GPU ends with this same averaged } g \text{)}`}</MB>
      <p>
        DDP is the workhorse: it’s easy, it scales throughput nearly linearly, and PyTorch overlaps
        the gradient all-reduce with the backward pass so communication hides behind compute. Its one
        limitation is the premise — <em>the whole model must fit on each GPU</em>. When it doesn’t,
        you need the strategies below.
      </p>

      <CodeBlock language="python" filename="ddp_train.py" highlight={[8, 14, 18, 20]}>
{`import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# torchrun sets RANK / LOCAL_RANK / WORLD_SIZE in the environment.
def setup():
    dist.init_process_group(backend="nccl")          # NCCL = NVIDIA GPU collectives
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

setup()
local_rank = int(os.environ["LOCAL_RANK"])
model = GPT(cfg).to(local_rank)
model = DDP(model, device_ids=[local_rank])          # wraps backward with all-reduce

# DistributedSampler gives each rank a *disjoint* shard of the data each epoch.
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=B, sampler=sampler)

for epoch in range(epochs):
    sampler.set_epoch(epoch)                          # reshuffle differently each epoch
    for x, y in loader:
        x, y = x.to(local_rank), y.to(local_rank)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        loss.backward()                               # gradients all-reduced here
        opt.step(); opt.zero_grad(set_to_none=True)

dist.destroy_process_group()`}
      </CodeBlock>
      <p>
        You launch this with <code>torchrun --nproc_per_node=8 ddp_train.py</code>, which spawns one
        process per GPU. Two details people miss: call{" "}
        <code>sampler.set_epoch(epoch)</code> so each epoch reshuffles (otherwise every epoch sees
        the same per-rank order), and remember that the effective batch size is now{" "}
        <code>B × world_size</code>.
      </p>

      <h2>ZeRO / FSDP: shard the model state itself</h2>
      <p>
        DDP wastes memory: it stores <M>{"N"}</M> identical copies of the params, gradients, and
        optimizer state — one per GPU. <strong>ZeRO</strong> (Zero Redundancy Optimizer), implemented
        in PyTorch as <strong>FSDP</strong> (Fully Sharded Data Parallel), removes that redundancy by{" "}
        <em>sharding</em> the model state across GPUs. It comes in three progressively aggressive
        stages:
      </p>
      <ul>
        <li>
          <strong>ZeRO-1</strong> — shard the <em>optimizer state</em> (<M>{"m, v"}</M>, fp32 master).
          That’s the biggest chunk (12 of the 16 bytes), so this alone is a big win.
        </li>
        <li>
          <strong>ZeRO-2</strong> — also shard the <em>gradients</em>.
        </li>
        <li>
          <strong>ZeRO-3 / FSDP</strong> — also shard the <em>parameters</em> themselves. Each GPU
          stores only a <M>{"1/N"}</M> slice; the full weights for a layer are{" "}
          <strong>all-gathered</strong> just-in-time right before that layer runs, used, then freed.
        </li>
      </ul>
      <p>
        FSDP behaves like DDP from the outside (each GPU still processes a different data shard) but
        trades a little extra communication — an all-gather of params on the way in, a
        reduce-scatter of gradients on the way out — for an <M>{"N"}</M>-fold reduction in per-GPU
        model-state memory. It is how most teams train models that are far too big for one device.
        Wrapping a model is essentially a one-liner:
      </p>
      <CodeBlock language="python">
{`from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Shards params + grads + optimizer state across the process group, and
# all-gathers each layer's weights just-in-time during forward/backward.
model = FSDP(model)   # (in practice: pass an auto_wrap_policy to shard per-block)`}
      </CodeBlock>

      <Callout type="industry" title="ZeRO-3 vs. tensor parallelism">
        <p>
          Both reduce per-GPU memory, but differently. FSDP/ZeRO-3 keeps each layer’s math{" "}
          <em>whole</em> and gathers the weights momentarily — great communication/compute overlap,
          minimal code change, scales across many nodes. Tensor parallelism (next) keeps the weights
          sharded and splits the <em>math</em> itself — lower latency per layer but heavier, more
          frequent communication that wants very fast interconnects (NVLink). In practice large runs
          use both: TP <em>within</em> a node, FSDP or DP <em>across</em> nodes.
        </p>
      </Callout>

      <h2>Tensor parallelism: split the matmuls</h2>
      <p>
        Sometimes a single layer’s weights — or its activations — are too big to gather even
        momentarily. <strong>Tensor parallelism</strong> (TP) splits the individual matrix
        multiplications across GPUs. Take a linear layer <M>{"Y = XW"}</M>. Partition the weight{" "}
        <M>{"W"}</M> by columns, <M>{"W = [W_1 \\; W_2]"}</M>, give each GPU one piece, and each
        computes a slice of the output: <M>{"Y_1 = X W_1"}</M>, <M>{"Y_2 = X W_2"}</M>. The full{" "}
        <M>{"Y = [Y_1 \\; Y_2]"}</M> is the concatenation — no communication needed yet. The very
        next layer is partitioned by <em>rows</em> instead, so its outputs must be summed with an{" "}
        <strong>all-reduce</strong>. In a transformer block this happens twice per layer (once in
        attention, once in the MLP), and the heads themselves split naturally across GPUs.
      </p>
      <p>
        TP is powerful but communication-hungry: those all-reduces fire <em>inside every layer</em>,
        on the critical path, so TP is almost always confined to GPUs within a single node connected
        by high-bandwidth NVLink. You rarely run tensor parallelism across nodes.
      </p>

      <h2>Pipeline parallelism: split the layers</h2>
      <p>
        Instead of splitting <em>within</em> layers, <strong>pipeline parallelism</strong> (PP)
        splits <em>across</em> them: assign a contiguous block of layers — a <strong>stage</strong> —
        to each GPU. GPU 0 runs layers 0–7, GPU 1 runs 8–15, and so on. An activation flows forward
        from stage to stage (a point-to-point send/recv), and gradients flow back the same way.
      </p>
      <p>
        The catch: naively, while stage 0 works on a batch, stages 1–3 sit idle waiting for it, then
        vice versa — most GPUs are starved most of the time. The fix is to split the batch into{" "}
        <strong>micro-batches</strong> and pipeline them, so that while stage 1 processes
        micro-batch 0, stage 0 already starts micro-batch 1. Even so, the pipeline must fill up and
        drain, leaving idle time at the ends — the <strong>bubble</strong>. With <M>{"S"}</M> stages
        and <M>{"m"}</M> micro-batches the bubble fraction is roughly:
      </p>
      <MB>{String.raw`\text{bubble fraction} \;\approx\; \frac{S - 1}{m + S - 1}`}</MB>
      <p>
        More micro-batches shrink the bubble (at the cost of smaller, less efficient matmuls), which
        is the central tuning knob of pipeline parallelism. You can watch this directly in the
        visualization below.
      </p>

      <Figure
        n="16.2"
        title="Four ways to split a model across 4 GPUs"
        caption="Switch strategies to see what each GPU holds (the layer × width-shard grid), how the batch is divided, and the communication each mode requires. Data: full model, different data, all-reduce gradients. Tensor: every GPU holds a width-slice of every layer, all-reduce inside each layer. Pipeline: each GPU owns contiguous layers; micro-batches flow through with a bubble. FSDP: each GPU holds a param shard, gathered just-in-time."
      >
        <ParallelismViz />
      </Figure>

      <h2>3D parallelism: combining all three</h2>
      <p>
        The largest models use all of these at once — <strong>3D parallelism</strong> — because each
        axis solves a different problem:
      </p>
      <ul>
        <li>
          <strong>Tensor parallel</strong> within a node (8 GPUs on fast NVLink) — splits each
          layer, the most communication-heavy axis, kept local.
        </li>
        <li>
          <strong>Pipeline parallel</strong> across a handful of nodes — splits the layers into
          stages, communication is just activation hand-offs at stage boundaries.
        </li>
        <li>
          <strong>Data parallel</strong> (or FSDP) across the remaining replicas — splits the batch,
          all-reducing gradients once per step.
        </li>
      </ul>
      <p>
        The GPU count multiplies: <M>{"N_{\\text{total}} = N_{\\text{TP}} \\times N_{\\text{PP}} \\times N_{\\text{DP}}"}</M>.
        Training a frontier model on thousands of GPUs means, for example, TP=8 × PP=8 × DP=16 = 1024
        GPUs, each axis chosen so its communication fits the interconnect available at that scale.
      </p>

      <Callout type="note" title="The collective operations, in one place">
        <p>
          Every strategy above is built from a small set of <strong>collective operations</strong> —
          group communication primitives provided by NCCL:
        </p>
        <ul>
          <li>
            <strong>all-reduce</strong> — every GPU contributes a tensor; all of them receive the sum
            (used for DDP gradients, TP activations). Equivalent to a reduce-scatter followed by an
            all-gather.
          </li>
          <li>
            <strong>all-gather</strong> — every GPU contributes a shard; all of them receive the full
            concatenation (FSDP gathers params before a layer).
          </li>
          <li>
            <strong>reduce-scatter</strong> — every GPU contributes a full tensor; each receives the
            sum of only <em>its</em> shard (FSDP gradients on the way back).
          </li>
        </ul>
      </Callout>

      <h2>How to choose</h2>
      <p>A practical decision order for picking a strategy:</p>
      <ol>
        <li>
          <strong>Model fits on one GPU?</strong> Use plain DDP — split the batch, all-reduce
          gradients. Simplest, fastest path.
        </li>
        <li>
          <strong>Model state too big, but a layer fits?</strong> Use FSDP / ZeRO-3 — shard params,
          grads, and optimizer state. Minimal code change, scales widely.
        </li>
        <li>
          <strong>A single layer is too big, or you need lower latency?</strong> Add tensor
          parallelism within each node.
        </li>
        <li>
          <strong>Thousands of GPUs?</strong> Combine all three into 3D parallelism, matching each
          axis to the interconnect.
        </li>
      </ol>

      <Callout type="tip" title="Memory first, then throughput">
        <p>
          The questions are ordered deliberately. First make the model <em>fit</em> (the{" "}
          <M>{"16P"}</M> budget plus activations) — that decides whether you need sharding or tensor
          parallelism at all. Only then optimize <em>throughput</em> (overlap, micro-batch counts,
          bubble size). A configuration that doesn’t fit has zero throughput; a configuration that
          fits can always be tuned faster.
        </p>
      </Callout>

      <h2>Bridge: how big should we go?</h2>
      <p>
        You can now spread a single model across thousands of GPUs, fitting parameters, gradients,
        and optimizer state that no one device could hold. That raises the strategic question: given
        a fixed budget of GPUs and time — a fixed amount of <em>compute</em> — how large should the
        model be, and how many tokens should it see? Spending compute on a bigger model versus more
        data is a real trade-off with a surprisingly clean answer. The next chapter,{" "}
        <a href="/chapter/scaling-laws" className="prose-link">Scaling Laws &amp; Compute-Optimal
        Training</a>, gives the power laws and the Chinchilla recipe that turn a compute budget into
        the optimal model size and dataset size.
      </p>
    </>
  );
}
