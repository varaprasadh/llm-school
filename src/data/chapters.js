/**
 * The LLM School — chapter registry.
 *
 * This is the single source of truth for navigation, routing and ordering.
 * Each chapter has a component file at `src/chapters/<slug>.jsx` that
 * default-exports a React component. Routes are wired up automatically in
 * App.jsx via `import.meta.glob`, so adding a chapter here + creating the
 * matching file is all that is required.
 */

export const parts = [
  {
    id: "foundations",
    label: "Part I",
    title: "Foundations",
    blurb: "The mental models you need before touching a single matrix.",
    accent: "cyan",
    chapters: [
      {
        num: 1,
        slug: "what-is-an-llm",
        title: "What Is a Large Language Model?",
        summary:
          "The 30,000-foot view: what an LLM actually is, the autoregressive idea, and the full lifecycle from raw text to deployed assistant.",
        minutes: 14,
      },
      {
        num: 2,
        slug: "neural-network-primer",
        title: "A Neural Network Primer",
        summary:
          "Neurons, layers, activations, the forward pass and backpropagation — the calculus engine that makes learning possible.",
        minutes: 22,
      },
      {
        num: 3,
        slug: "language-modeling",
        title: "The Language Modeling Objective",
        summary:
          "What it means to 'model language' as probability, the chain rule of language, and next-token prediction.",
        minutes: 16,
      },
    ],
  },
  {
    id: "data",
    label: "Part II",
    title: "Data & Representation",
    blurb: "Turning the messy internet into clean integer sequences a model can learn from.",
    accent: "emerald",
    chapters: [
      {
        num: 4,
        slug: "data-curation",
        title: "Data: Collection, Cleaning & Curation",
        summary:
          "Where pretraining data comes from, the cleaning pipeline, deduplication, quality filtering, and why data is the real moat.",
        minutes: 20,
      },
      {
        num: 5,
        slug: "tokenization",
        title: "Tokenization & Byte-Pair Encoding",
        summary:
          "From characters to subwords. Build a BPE tokenizer step-by-step with an interactive merge visualizer.",
        minutes: 24,
      },
      {
        num: 6,
        slug: "embeddings",
        title: "Embeddings: Words as Vectors",
        summary:
          "How token IDs become dense vectors, what the embedding space means geometrically, and vector arithmetic on meaning.",
        minutes: 18,
      },
    ],
  },
  {
    id: "transformer",
    label: "Part III",
    title: "The Transformer",
    blurb: "The architecture that ate the world — dissected one block at a time.",
    accent: "violet",
    chapters: [
      {
        num: 7,
        slug: "transformer-overview",
        title: "The Transformer Architecture",
        summary:
          "The full decoder-only stack, end-to-end. Follow a sequence of tokens through every layer to the output logits.",
        minutes: 20,
      },
      {
        num: 8,
        slug: "self-attention",
        title: "Self-Attention from Scratch",
        summary:
          "Queries, keys and values; scaled dot-product attention; causal masking — with a live attention-matrix visualizer.",
        minutes: 26,
      },
      {
        num: 9,
        slug: "multi-head-attention",
        title: "Multi-Head Attention",
        summary:
          "Why one attention pattern is not enough. Splitting into heads, parallel subspaces, and recombination.",
        minutes: 16,
      },
      {
        num: 10,
        slug: "positional-encoding",
        title: "Positional Encoding & RoPE",
        summary:
          "Attention is permutation-invariant — so how does the model know word order? Sinusoidal, learned, and rotary embeddings.",
        minutes: 18,
      },
      {
        num: 11,
        slug: "transformer-block",
        title: "Feed-Forward, Norm & Residuals",
        summary:
          "The other half of the block: position-wise MLPs, residual highways, and LayerNorm/RMSNorm that keep deep stacks trainable.",
        minutes: 16,
      },
    ],
  },
  {
    id: "training",
    label: "Part IV",
    title: "Training",
    blurb: "Where the weights actually get learned — objective, optimizer, code and hardware.",
    accent: "amber",
    chapters: [
      {
        num: 12,
        slug: "training-objective",
        title: "Loss, Cross-Entropy & Perplexity",
        summary:
          "The objective function in full: cross-entropy, its gradient, and perplexity as the metric you will live and die by.",
        minutes: 18,
      },
      {
        num: 13,
        slug: "optimization",
        title: "Optimizers, Learning Rates & Schedules",
        summary:
          "SGD → Momentum → Adam → AdamW. Warmup, cosine decay, gradient clipping, and the knobs that decide whether training converges.",
        minutes: 20,
      },
      {
        num: 14,
        slug: "building-the-model",
        title: "Building the Model in PyTorch",
        summary:
          "A complete, runnable GPT implementation — every module from the embedding table to the language-model head, fully annotated.",
        minutes: 28,
      },
      {
        num: 15,
        slug: "training-loop",
        title: "The Training Loop",
        summary:
          "Forward, backward, step. Batching, gradient accumulation, mixed precision, checkpointing, and reading the loss curve.",
        minutes: 22,
      },
      {
        num: 16,
        slug: "distributed-training",
        title: "Scaling Up: Distributed Training",
        summary:
          "When the model no longer fits on one GPU: data, tensor, pipeline and fully-sharded parallelism explained visually.",
        minutes: 24,
      },
      {
        num: 17,
        slug: "scaling-laws",
        title: "Scaling Laws & Compute-Optimal Training",
        summary:
          "The power laws relating loss to compute, data and parameters — and the Chinchilla recipe for spending a compute budget.",
        minutes: 18,
      },
    ],
  },
  {
    id: "post-training",
    label: "Part V",
    title: "Post-Training & Alignment",
    blurb: "Turning a raw next-token predictor into a helpful, harmless assistant.",
    accent: "rose",
    chapters: [
      {
        num: 18,
        slug: "supervised-finetuning",
        title: "Supervised Fine-Tuning (SFT)",
        summary:
          "Instruction tuning: chat templates, loss masking on prompts, and teaching the base model to follow instructions.",
        minutes: 18,
      },
      {
        num: 19,
        slug: "rlhf",
        title: "RLHF, Reward Models & DPO",
        summary:
          "Aligning to human preferences: reward modeling, PPO, and the simpler Direct Preference Optimization that replaced it.",
        minutes: 24,
      },
      {
        num: 20,
        slug: "peft-lora",
        title: "Parameter-Efficient Fine-Tuning",
        summary:
          "Fine-tune billion-parameter models on a single GPU with LoRA and QLoRA — the low-rank trick that changed the game.",
        minutes: 18,
      },
    ],
  },
  {
    id: "deploy",
    label: "Part VI",
    title: "Evaluation & Deployment",
    blurb: "Proving it works, making it fast, and putting it in front of real users.",
    accent: "cyan",
    chapters: [
      {
        num: 21,
        slug: "evaluation",
        title: "Evaluation & Benchmarks",
        summary:
          "Perplexity, MMLU, HumanEval, LLM-as-judge and the art of not fooling yourself. Contamination, and what the leaderboards hide.",
        minutes: 20,
      },
      {
        num: 22,
        slug: "inference-optimization",
        title: "Inference Optimization",
        summary:
          "The KV cache, quantization, FlashAttention, speculative decoding, and distillation — squeezing latency and cost out of serving.",
        minutes: 24,
      },
      {
        num: 23,
        slug: "deployment-serving",
        title: "Deployment & Serving at Scale",
        summary:
          "Continuous batching, paged attention, vLLM, autoscaling, and exposing your model behind a production-grade API.",
        minutes: 22,
      },
      {
        num: 24,
        slug: "monitoring-safety",
        title: "Monitoring, Safety & Guardrails",
        summary:
          "Observability, drift, red-teaming, jailbreak defense, content filtering, and responsible deployment.",
        minutes: 18,
      },
      {
        num: 25,
        slug: "capstone",
        title: "Capstone: The Full Pipeline",
        summary:
          "Everything assembled into one diagram and one checklist — from a scraped web page to a token streaming back to a user.",
        minutes: 16,
      },
    ],
  },
];

// ---- Derived helpers -------------------------------------------------------

/** Flat, ordered list of every chapter, with its part attached. */
export const allChapters = parts.flatMap((part) =>
  part.chapters.map((ch) => ({ ...ch, partId: part.id, partTitle: part.title, accent: part.accent }))
);

/** slug -> chapter (with part metadata). */
export const chapterBySlug = Object.fromEntries(allChapters.map((c) => [c.slug, c]));

/** Previous / next navigation for a given slug. */
export function neighbors(slug) {
  const i = allChapters.findIndex((c) => c.slug === slug);
  return {
    prev: i > 0 ? allChapters[i - 1] : null,
    next: i >= 0 && i < allChapters.length - 1 ? allChapters[i + 1] : null,
  };
}

export const totalMinutes = allChapters.reduce((s, c) => s + c.minutes, 0);
export const chapterCount = allChapters.length;
