import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LoRAViz from "../components/viz/peft-lora/LoRAViz";

export default function Chapter() {
  return (
    <>
      <p>
        By now you can take a pretrained model and fine-tune it (
        <a href="/chapter/supervised-finetuning" className="prose-link">
          Chapter 18
        </a>
        ). But there is a brutal practical problem: fine-tuning a 7B-parameter model the
        naive way needs roughly <strong>60–80 GB of GPU memory</strong> — more than a single
        consumer or even most single data-center GPUs have. This chapter is about the trick
        that broke that barrier and put fine-tuning into everyone&rsquo;s hands. The whole
        idea fits in one sentence: <em>the change you make to a model during fine-tuning is
        simpler than the model itself</em>, so you can store that change in a tiny fraction
        of the space.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          The weight <em>update</em> learned during fine-tuning, <M>{"\\Delta W"}</M>, has
          low <strong>intrinsic rank</strong>. So instead of training a full dense{" "}
          <M>{"\\Delta W"}</M>, freeze the pretrained weights <M>{"W_0"}</M> and learn{" "}
          <M>{"\\Delta W"}</M> as the product of two skinny matrices,{" "}
          <M>{"\\Delta W = B A"}</M>. You train a few million parameters instead of a few
          billion — and reach almost the same quality.
        </p>
      </Callout>

      <h2>Full fine-tuning is expensive</h2>
      <p>
        Fine-tuning means continuing gradient descent on a pretrained model with new data.
        Conceptually it&rsquo;s just more training. The cost comes from the optimizer.
        Recall from{" "}
        <a href="/chapter/distributed-training" className="prose-link">
          Chapter 16
        </a>{" "}
        that with the standard <strong>Adam</strong> optimizer in mixed precision, every
        single trainable parameter drags along a fixed retinue of memory:
      </p>
      <ul>
        <li>
          <strong>Weights</strong> — a master copy in fp32 (4 bytes) plus often a bf16 copy
          (2 bytes) used for the forward/backward pass.
        </li>
        <li>
          <strong>Gradients</strong> — one number per parameter (4 bytes in fp32).
        </li>
        <li>
          <strong>Optimizer state</strong> — Adam keeps a first moment{" "}
          <M>{"m"}</M> and a second moment <M>{"v"}</M>, 4 bytes each.
        </li>
      </ul>
      <p>
        Add it up and you are carrying on the order of <strong>16 bytes per parameter</strong>{" "}
        just for the model and optimizer (4 + 4 + 4 + 4), before you spend a single byte on
        activations. For a model with <M>{"P"}</M> parameters:
      </p>
      <MB>{String.raw`\text{memory} \;\approx\; \underbrace{16\,P}_{\text{weights + grads + Adam } m,v} \;+\; \text{activations}`}</MB>
      <p>
        That is the killer. A 7B model needs <M>{"16 \\times 7\\text{B} \\approx 112"}</M> GB
        of state for full fine-tuning; a 65B model needs over a <em>terabyte</em>. The model
        weights themselves are only a quarter of that — the optimizer states for{" "}
        <em>every</em> parameter are what crush you.
      </p>

      <Figure
        n="20.1"
        title="Where the memory goes"
        caption="Full fine-tuning carries Adam optimizer state (and gradients) for every one of the billions of parameters. LoRA freezes the base weights — no gradient, no m, no v for them — and pays the optimizer tax only on the tiny adapter."
      >
        <MemoryBars />
      </Figure>

      <Callout type="note" title="Why not just freeze most layers?">
        <p>
          An older trick is to freeze the bottom layers and fine-tune only the top few. It
          saves some optimizer memory but is a blunt instrument: you lose the ability to
          adapt the frozen layers at all, and choosing <em>which</em> layers to unfreeze is
          guesswork. LoRA is better because it lets <em>every</em> layer adapt, just along a
          low-rank slice — full coverage at a tiny cost.
        </p>
      </Callout>

      <h2>The low-rank insight</h2>
      <p>
        Here is the observation that makes everything work. When you fine-tune a model, the
        final weights are <M>{"W = W_0 + \\Delta W"}</M>, where <M>{"W_0"}</M> are the
        pretrained weights and <M>{"\\Delta W"}</M> is the total change. The pretrained{" "}
        <M>{"W_0"}</M> is a high-rank matrix — it stores enormous amounts of general
        knowledge. But the <em>update</em> <M>{"\\Delta W"}</M> need not be. Adapting a model
        to a specific task (summarize legal documents, speak in a brand voice, answer medical
        questions) is a much smaller change than learning language from scratch.
      </p>
      <p>
        The LoRA authors (Hu et al., 2021) hypothesized — and measured — that{" "}
        <M>{"\\Delta W"}</M> has a low <strong>intrinsic rank</strong>. Recall that the rank
        of a matrix is the number of linearly independent directions it spans. A{" "}
        <M>{"d \\times k"}</M> matrix can have rank up to <M>{"\\min(d, k)"}</M>, but a
        low-rank one is far more compressible. Any rank-<M>{"r"}</M> matrix factors exactly
        into a product of two thin matrices:
      </p>
      <MB>{String.raw`\underbrace{\Delta W}_{d \times k,\ \text{rank } r} \;=\; \underbrace{B}_{d \times r}\,\underbrace{A}_{r \times k}, \qquad r \ll \min(d, k)`}</MB>
      <p>
        If <M>{"\\Delta W"}</M> truly lives in a low-rank subspace, we never have to
        materialize the full <M>{"d \\times k"}</M> matrix during training. We can{" "}
        <em>parameterize</em> it directly by its two small factors and learn those instead.
        That is LoRA.
      </p>

      <Callout type="math" title="Intrinsic dimension, briefly">
        <p>
          A precursor result (Aghajanyan et al., 2020) showed you can fine-tune large models
          by optimizing in a randomly-projected subspace of only a few hundred to a few
          thousand dimensions and still recover most of full fine-tuning&rsquo;s performance.
          The fine-tuning objective has a low <em>intrinsic dimension</em>. LoRA turns that
          empirical fact into an architecture: it constrains the update to rank{" "}
          <M>{"r"}</M> per weight matrix, which is exactly a low-dimensional reparameterization.
        </p>
      </Callout>

      <h2>LoRA: Low-Rank Adaptation</h2>
      <p>
        Take any weight matrix in the model you want to adapt — typically the attention
        projections <M>{"W_Q, W_K, W_V, W_O"}</M>, and often the MLP weights too. Call it{" "}
        <M>{"W_0 \\in \\mathbb{R}^{d \\times k}"}</M>. LoRA <strong>freezes</strong>{" "}
        <M>{"W_0"}</M> (no gradients, no optimizer state) and represents the update with two
        new trainable matrices:
      </p>
      <MB>{String.raw`h \;=\; W_0\,x \;+\; \Delta W\,x \;=\; W_0\,x \;+\; \frac{\alpha}{r}\,B A\,x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}`}</MB>
      <p>where, piece by piece:</p>
      <ul>
        <li>
          <M>{"A \\in \\mathbb{R}^{r \\times k}"}</M> projects the input <em>down</em> to a
          tiny <M>{"r"}</M>-dimensional bottleneck;
        </li>
        <li>
          <M>{"B \\in \\mathbb{R}^{d \\times r}"}</M> projects <em>back up</em> to the output
          dimension. Together they span a rank-<M>{"r"}</M> update.
        </li>
        <li>
          <M>{"r"}</M> is the rank, a hyperparameter you choose, with{" "}
          <M>{"r \\ll \\min(d, k)"}</M> — typically 4, 8, 16, 32, or 64.
        </li>
        <li>
          <M>{"\\alpha"}</M> is a scaling constant. The update is scaled by{" "}
          <M>{"\\alpha / r"}</M> so that changing <M>{"r"}</M> doesn&rsquo;t force you to
          re-tune the learning rate; <M>{"\\alpha"}</M> is commonly set to{" "}
          <M>{"r"}</M> or <M>{"2r"}</M> and then mostly left alone.
        </li>
      </ul>
      <p>
        Two initialization choices make this work cleanly. <M>{"A"}</M> is initialized to
        small random Gaussian values, and <M>{"B"}</M> is initialized to{" "}
        <strong>exactly zero</strong>. So at the very start of training{" "}
        <M>{"\\Delta W = B A = 0"}</M>: the adapter is a <em>no-op</em>, and the model behaves
        identically to the pretrained one. Training then gently moves <M>{"B"}</M> away from
        zero. You begin from the pretrained model, not from a randomly perturbed one — which
        makes fine-tuning stable from step one.
      </p>

      <Callout type="pitfall" title="Don't initialize both A and B randomly">
        <p>
          If both <M>{"A"}</M> and <M>{"B"}</M> started random, then <M>{"BA \\neq 0"}</M> at
          step 0 and you would inject a large random update into a carefully pretrained
          network — corrupting it before training even begins. Exactly one of the two factors
          must start at zero (LoRA zeroes <M>{"B"}</M>) so the initial <M>{"\\Delta W"}</M> is
          zero while gradients can still flow through the nonzero factor.
        </p>
      </Callout>

      <p>
        The interactive figure below is the heart of the chapter. The frozen{" "}
        <M>{"W_0"}</M> on the left never changes. The trainable update on the right is{" "}
        <M>{"(\\alpha/r)\\,B A"}</M> — a tall strip <M>{"B"}</M> times a wide strip{" "}
        <M>{"A"}</M>. Drag the rank slider and watch the two skinny matrices grow while the
        trainable-parameter count stays a sliver of the full matrix.
      </p>

      <Figure
        n="20.2"
        title="Two skinny matrices replace one huge one"
        caption="ΔW is a full d×d matrix, but it is constructed from B (d×r) and A (r×d), so only 2·d·r numbers are trainable. At r = 8 against d = 4096, that is 99.6% fewer trainable parameters than a dense update."
      >
        <LoRAViz />
      </Figure>

      <h2>The parameter savings</h2>
      <p>
        Let&rsquo;s make the arithmetic concrete. A dense update to a{" "}
        <M>{"d \\times k"}</M> matrix has <M>{"d \\cdot k"}</M> trainable parameters. The
        LoRA factors have:
      </p>
      <MB>{String.raw`\underbrace{d \cdot k}_{\text{dense } \Delta W} \;\longrightarrow\; \underbrace{d \cdot r \;+\; r \cdot k \;=\; r\,(d + k)}_{\text{LoRA: } B \text{ and } A}`}</MB>
      <p>
        Take a square attention projection with <M>{"d = k = 4096"}</M> and choose{" "}
        <M>{"r = 8"}</M>:
      </p>
      <MB>{String.raw`\frac{r(d+k)}{d\,k} \;=\; \frac{8 \cdot (4096 + 4096)}{4096 \cdot 4096} \;=\; \frac{65{,}536}{16{,}777{,}216} \;\approx\; 0.39\%`}</MB>
      <p>
        That is a <strong>99.6% reduction</strong> in trainable parameters for that matrix —
        from ~16.8M down to ~65.5K. Scaled across a whole model, LoRA typically makes well
        under <strong>1% of parameters trainable</strong>. For a 7B model you might train
        only a few million LoRA parameters. And because only those few million parameters
        have gradients and Adam states, the optimizer-memory term collapses from{" "}
        <M>{"12\\,P"}</M> over <em>all</em> parameters to <M>{"12\\,P_{\\text{LoRA}}"}</M>{" "}
        over only the adapter — the frozen base needs just its (often quantized) weights and
        no optimizer state at all.
      </p>

      <Callout type="industry" title="Typical recipe">
        <p>
          A common, strong default: apply LoRA to all attention and MLP linear layers,{" "}
          <M>{"r = 16"}</M>, <M>{"\\alpha = 32"}</M>, dropout 0.05, learning rate{" "}
          <M>{"1\\text{–}3 \\times 10^{-4}"}</M> (higher than full fine-tuning, since you are
          training far fewer, freshly-initialized parameters). The resulting adapter for a 7B
          model is often just <strong>10–200 MB on disk</strong> — small enough to email.
        </p>
      </Callout>

      <h2>Inference: merge or keep separate</h2>
      <p>
        Because the update is just an additive term, LoRA imposes <strong>zero extra
        latency</strong> at inference if you want it to. You have two options:
      </p>
      <ul>
        <li>
          <strong>Merge.</strong> Compute <M>{"W = W_0 + \\frac{\\alpha}{r} B A"}</M> once and
          fold it back into the original weight matrix. The merged model has the exact same
          shape and FLOPs as the original — there is no adapter to run, no overhead. This is
          what you ship when you have a single fine-tuned model to deploy.
        </li>
        <li>
          <strong>Keep separate.</strong> Leave <M>{"W_0"}</M> untouched and apply the small{" "}
          <M>{"B A"}</M> path at runtime. You pay a tiny extra matmul, but you gain something
          powerful: the adapter is a swappable plug-in.
        </li>
      </ul>

      <Callout type="industry" title="One base model, many adapters">
        <p>
          Keeping adapters separate is a deployment superpower. A serving stack can hold a
          single copy of the base model in GPU memory and hot-swap{" "}
          <em>hundreds</em> of small LoRA adapters — one per customer, per task, per language
          — selecting the right one per request. Systems like S-LoRA and Punica batch
          requests across <em>different</em> adapters at once. That is how a provider offers
          thousands of customized models without thousands of full model copies. Merging, by
          contrast, gives a frozen one-off — great for a dedicated endpoint, useless for
          multi-tenant serving.
        </p>
      </Callout>

      <h2>Implementing LoRA from scratch</h2>
      <p>
        A LoRA layer is genuinely small. Here is a drop-in replacement for a{" "}
        <code>nn.Linear</code> that wraps a frozen base layer and adds the trainable low-rank
        path. The base layer&rsquo;s parameters have <code>requires_grad = False</code>, so
        the optimizer never allocates state for them.
      </p>

      <CodeBlock language="python" filename="lora.py" highlight={[19, 20, 21, 28]}>
{`import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """A frozen Linear with an additive trainable low-rank update BA."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base                      # the pretrained W0 (and bias)
        for p in self.base.parameters():
            p.requires_grad = False           # freeze: no grad, no optim state

        d_out, d_in = base.weight.shape       # W0 is (d_out, d_in)
        self.r = r
        self.scaling = alpha / r              # the alpha/r factor
        self.dropout = nn.Dropout(dropout)

        # Low-rank factors. A ~ small Gaussian, B = 0  ->  BA = 0 at init.
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.normal_(self.A, std=1.0 / r)  # B stays zero (a no-op at start)

    def forward(self, x):
        # frozen path + scaled low-rank update
        base_out = self.base(x)               # x @ W0^T + b   (no grad to W0)
        lora_out = self.dropout(x) @ self.A.T @ self.B.T   # (x A^T) B^T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self):
        """Fold BA into W0 for zero-overhead inference, then it's a plain Linear."""
        delta = (self.B @ self.A) * self.scaling          # (d_out, d_in)
        self.base.weight.add_(delta)
        return self.base                       # the merged, standalone layer`}
      </CodeBlock>

      <p>
        In real projects you don&rsquo;t hand-wire every layer — you use Hugging Face&rsquo;s{" "}
        <code>peft</code> library, which wraps any 🤗 Transformers model, injects LoRA into
        the modules you name, and tracks only the adapter for saving:
      </p>

      <CodeBlock language="python" filename="train_peft.py">
{`from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 40,370,176 || all params: 6,778,769,408 || trainable%: 0.5956

# ...now train with a normal Trainer / loop. Only A,B receive gradients.
model.save_pretrained("./llama2-7b-lora")   # saves ONLY the small adapter`}
      </CodeBlock>

      <h2>QLoRA: quantize the frozen base</h2>
      <p>
        LoRA already removes the optimizer tax. But the frozen base weights still sit in
        memory — 7B params at 2 bytes (bf16) is ~14 GB, and a 65B model is ~130 GB, still too
        big for one GPU. <strong>QLoRA</strong> (Dettmers et al., 2023) closes the gap with a
        beautiful observation: since the base is frozen and only read during the forward and
        backward pass, you can store it in very low precision without ever needing to update
        it. QLoRA quantizes the frozen base to <strong>4-bit</strong> and keeps the small
        LoRA adapters in higher precision (bf16), where the actual learning happens. It has
        three ingredients:
      </p>
      <ul>
        <li>
          <strong>4-bit NormalFloat (NF4).</strong> A 4-bit data type whose 16 quantization
          levels are spaced to be <em>information-theoretically optimal for
          normally-distributed weights</em> (neural-net weights are roughly Gaussian). NF4
          preserves accuracy far better than naive 4-bit integer quantization at the same bit
          width.
        </li>
        <li>
          <strong>Double quantization.</strong> Quantization itself needs scaling constants
          (one per block of weights). QLoRA quantizes <em>those constants too</em>, saving an
          extra ~0.4 bits per parameter — small per weight, but meaningful across billions.
        </li>
        <li>
          <strong>Paged optimizers.</strong> Using NVIDIA unified memory, optimizer state can
          be paged to CPU RAM and back automatically, absorbing the memory spikes (e.g. from
          a long sequence) that would otherwise trigger an out-of-memory crash.
        </li>
      </ul>
      <p>
        The frozen weights live in NF4; during the forward/backward pass each block is
        dequantized <em>on the fly</em> to bf16 for the matmul, used, and discarded — so peak
        memory stays at the 4-bit footprint. Gradients flow only into the bf16 LoRA adapters.
        The headline result: QLoRA fine-tunes a <strong>65B-parameter model on a single 48 GB
        GPU</strong>, and a 7B model fits comfortably in under 12 GB — while matching the task
        performance of 16-bit full fine-tuning on their benchmarks.
      </p>

      <Callout type="industry" title="QLoRA put fine-tuning on a laptop GPU">
        <p>
          Before QLoRA, fine-tuning a 30B+ model meant a multi-GPU node. After it, hobbyists
          fine-tuned 33B and 65B models on a single rented A6000 or A100. The Guanaco models
          in the QLoRA paper were trained this way and were, at release, among the strongest
          openly-available chat models. In <code>peft</code> + <code>bitsandbytes</code> it is
          a few lines: load the base with <code>load_in_4bit=True</code> and an{" "}
          <code>NF4</code> quant config, then attach LoRA exactly as before.
        </p>
      </Callout>

      <CodeBlock language="python" filename="qlora.py" highlight={[6, 7, 8, 9, 10]}>
{`import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # NormalFloat-4, not int4
    bnb_4bit_use_double_quant=True,          # quantize the quant constants
    bnb_4bit_compute_dtype=torch.bfloat16,   # dequantize to bf16 for matmuls
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf", quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)   # cast norms, enable grad ckpt

config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, config)
# Frozen base in 4-bit NF4; only the bf16 LoRA adapters are trained.`}
      </CodeBlock>

      <h2>Other PEFT methods</h2>
      <p>
        LoRA is the dominant member of a broader family of{" "}
        <strong>parameter-efficient fine-tuning</strong> (PEFT) methods — all sharing the goal
        of adapting a frozen model by training only a small set of new parameters. The others
        are worth knowing:
      </p>
      <ul>
        <li>
          <strong>Prompt tuning</strong> learns a handful of continuous &ldquo;soft
          prompt&rdquo; vectors that are prepended to the input embeddings. The entire model
          is frozen; only these virtual tokens are trained. Extremely cheap, but generally
          less expressive than LoRA.
        </li>
        <li>
          <strong>Prefix tuning</strong> generalizes this: it prepends trainable vectors to
          the keys and values at <em>every</em> attention layer, not just the input. More
          capacity than prompt tuning, still no change to the base weights.
        </li>
        <li>
          <strong>Adapters</strong> (Houlsby et al., 2019) — the original PEFT idea — insert
          small bottleneck MLP modules <em>between</em> the frozen transformer sublayers.
          Effective, but they add sequential layers, so unlike merged LoRA they impose some
          inference latency.
        </li>
        <li>
          <strong>(IA)³</strong> learns three tiny vectors per block that <em>rescale</em> the
          keys, values, and MLP activations element-wise. It trains even fewer parameters than
          LoRA and, like LoRA, can be merged for zero overhead.
        </li>
      </ul>
      <p>
        Variants of LoRA itself keep appearing: <strong>DoRA</strong> decomposes weights into
        magnitude and direction and applies LoRA to the direction; <strong>QA-LoRA</strong>{" "}
        and <strong>LoftQ</strong> make quantization and low-rank adaptation cooperate better.
        But plain LoRA and QLoRA remain the workhorses you will reach for first.
      </p>

      <Callout type="tip" title="When NOT to reach for PEFT">
        <p>
          PEFT shines for adapting to tasks, styles, and domains. It is <em>not</em> the right
          tool for teaching a model large amounts of genuinely new knowledge or capabilities —
          that low-rank bottleneck is a real constraint. If you have the compute and need a
          deep behavioral change, full fine-tuning can still pull ahead. Most practitioners
          start with LoRA/QLoRA and only escalate if quality demands it.
        </p>
      </Callout>

      <h2>Where this leaves us</h2>
      <p>
        You now have the complete toolkit to take a pretrained model and specialize it on a
        single GPU: freeze the base, learn a low-rank update, and — with QLoRA — quantize the
        frozen part to 4-bit so even 65B models fit. The output is a small, swappable adapter
        you can merge for fast inference or keep separate to serve many tasks from one base
        model. The natural next questions are: <em>did the fine-tune actually work?</em> and{" "}
        <em>how do we serve it fast and cheaply?</em> Those take us into{" "}
        <a href="/chapter/evaluation" className="prose-link">
          Chapter 21 (Evaluation)
        </a>{" "}
        and{" "}
        <a href="/chapter/inference-optimization" className="prose-link">
          Chapter 22 (Inference Optimization)
        </a>
        , where the quantization ideas you met in QLoRA come back at serving time.
      </p>
    </>
  );
}

/* ------------------------------------------------------------------ *
 * Figure 20.1 — a small, deterministic SVG comparing the memory
 * footprint of full fine-tuning vs LoRA for a 7B model. Pure markup,
 * no state: weights / gradients / Adam(m) / Adam(v) bytes per param.
 * ------------------------------------------------------------------ */
function MemoryBars() {
  // Bytes per parameter for each component (mixed-precision Adam baseline).
  const segs = [
    { label: "weights (fp32)", bytes: 4, color: "#5b7dff" },
    { label: "gradients", bytes: 4, color: "#22d3ee" },
    { label: "Adam m", bytes: 4, color: "#a855f7" },
    { label: "Adam v", bytes: 4, color: "#f59e0b" },
  ];
  const P = 7e9; // 7B parameters
  const fracTrainable = 0.006; // ~0.6% trainable under LoRA
  const W = 560;
  const labelW = 150;
  const barW = W - labelW - 90;
  const totalBytes = segs.reduce((s, x) => s + x.bytes, 0); // 16 B/param
  const gb = (bytes, params) => (bytes * params) / 1024 ** 3;

  // For LoRA, the frozen base keeps only its weights; grads + Adam apply to
  // the tiny trainable slice. (Base shown in bf16, 2 bytes, as commonly loaded.)
  const loraSegs = [
    { label: "frozen weights (bf16)", bytes: 2 * P, color: "#475569" },
    { label: "grads (adapter)", bytes: 4 * P * fracTrainable, color: "#22d3ee" },
    { label: "Adam m,v (adapter)", bytes: 8 * P * fracTrainable, color: "#a855f7" },
  ];

  const fullTotalGB = gb(totalBytes, P); // ~104 GB
  const loraTotalGB = loraSegs.reduce((s, x) => s + x.bytes, 0) / 1024 ** 3;
  const maxGB = fullTotalGB; // scale bars to the full-FT total

  const Row = ({ y, name, parts, totalGB, note }) => {
    let x = labelW;
    return (
      <g>
        <text x={0} y={y + 15} fontSize="12" fill="#cbd5e1" fontFamily="JetBrains Mono, monospace">
          {name}
        </text>
        {parts.map((p, i) => {
          const w = (p.gb / maxGB) * barW;
          const rect = (
            <g key={i}>
              <rect x={x} y={y} width={Math.max(0, w)} height={24} fill={p.color} rx={2}>
                <title>{`${p.label}: ${p.gb.toFixed(1)} GB`}</title>
              </rect>
            </g>
          );
          x += w;
          return rect;
        })}
        <text
          x={labelW + barW + 8}
          y={y + 16}
          fontSize="12"
          fill="#e2e8f0"
          fontFamily="JetBrains Mono, monospace"
        >
          {totalGB.toFixed(0)} GB
        </text>
        {note && (
          <text x={labelW} y={y + 38} fontSize="10" fill="#64748b">
            {note}
          </text>
        )}
      </g>
    );
  };

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto scrollbar-thin">
        <svg width={W} height={150} className="block min-w-[520px]">
          <Row
            y={14}
            name="Full FT"
            totalGB={fullTotalGB}
            note="optimizer state for all 7B params"
            parts={segs.map((s) => ({ ...s, gb: gb(s.bytes, P) }))}
          />
          <Row
            y={86}
            name="LoRA"
            totalGB={loraTotalGB}
            note="grads + Adam only on the ~0.6% trainable adapter"
            parts={loraSegs.map((s) => ({ ...s, gb: s.bytes / 1024 ** 3 }))}
          />
        </svg>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
        {[...segs, { label: "frozen (no optim state)", color: "#475569" }].map((s) => (
          <span key={s.label} className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <p className="text-xs text-slate-500">
        Model-state memory only (excludes activations), 7B params, mixed-precision Adam. Full
        fine-tuning ≈ {fullTotalGB.toFixed(0)} GB; LoRA ≈ {loraTotalGB.toFixed(0)} GB. QLoRA
        shrinks the frozen bar further by storing it in 4-bit (~3.5 GB).
      </p>
    </div>
  );
}
