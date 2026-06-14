import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import NextTokenDemo from "../components/viz/NextTokenDemo";
import LifecyclePipeline from "../components/viz/LifecyclePipeline";
import { useNavigate } from "react-router-dom";

export default function Chapter() {
  const navigate = useNavigate();
  return (
    <>
      <p>
        A <strong>large language model</strong> (LLM) can write essays, debug code, translate
        languages and explain quantum mechanics — yet underneath the magic it does exactly one
        thing, billions of times per second: it guesses the next word. Everything else in this book
        is an elaboration of that single idea. Before we touch any mathematics, let’s build a rock-solid
        intuition for what these systems are and how the pieces fit together.
      </p>

      <h2>The one-sentence definition</h2>
      <p>
        An LLM is a <em>probability distribution over sequences of text</em>, implemented as a giant
        neural network, learned from a giant pile of text. Given some words, it tells you how likely
        every possible next word is. That’s it. The “large” refers to two numbers that are both
        enormous: the amount of text it learned from, and the number of internal dials (
        <em>parameters</em>) it can tune.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          A language model assigns a probability to text. If you can score how likely any sentence
          is, you can <em>generate</em> sentences by repeatedly choosing likely next words. Scoring
          and generating are two sides of the same coin.
        </p>
      </Callout>

      <h2>It’s just predicting the next token</h2>
      <p>
        Suppose you read the words <em>“The cat sat on the …”</em>. Your brain instantly expects
        “mat”, “floor”, or “sofa” — and definitely not “photosynthesis”. You hold a distribution
        over what comes next. An LLM does the same, but it makes that distribution explicit: a number
        for every word in its vocabulary, all summing to 1.
      </p>
      <p>
        Play with the toy model below. It holds a tiny hand-built vocabulary, but the mechanics are
        identical to GPT-class models. Watch how the <strong>temperature</strong> knob reshapes the
        probabilities, and hit <em>Sample</em> repeatedly to watch text generate itself one token at
        a time.
      </p>

      <Figure
        n="1.1"
        title="A toy autoregressive language model"
        caption="Generation is a loop: predict a distribution over the next token, pick one, append it, and feed the longer sequence back in. This is literally how ChatGPT writes — token by token, left to right."
      >
        <NextTokenDemo />
      </Figure>

      <h2>“Token”, not “word”</h2>
      <p>
        We keep saying “word”, but real models operate on <strong>tokens</strong>: chunks of text
        that are often whole words but sometimes word-pieces. The word <code>tokenization</code>{" "}
        might split into <code>token</code> + <code>ization</code>. Punctuation, spaces and even
        emoji are tokens too. We’ll build a tokenizer from scratch in{" "}
        <button onClick={() => navigate("/chapter/tokenization")} className="prose-link">
          Chapter 5
        </button>
        ; for now, just read “token” as “a small piece of text the model treats as one unit”.
      </p>

      <h2>From one guess to fluent paragraphs</h2>
      <p>
        A single prediction isn’t impressive. The power comes from <strong>autoregression</strong>:
        feeding the model’s own output back in as input. Formally, the probability of a whole
        sequence factorizes into a product of next-token predictions via the chain rule of
        probability:
      </p>
      <MB>{String.raw`P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})`}</MB>
      <p>
        Read that right-hand side aloud: “the probability of token <M>{"x_t"}</M> given everything
        before it.” The model only ever has to answer that one question — what comes next? — and the
        product stitches individual guesses into coherent text. We’ll make this objective precise in{" "}
        <button onClick={() => navigate("/chapter/language-modeling")} className="prose-link">
          Chapter 3
        </button>
        .
      </p>

      <Callout type="industry" title="Why this framing won">
        <p>
          Next-token prediction needs <em>no human labels</em>. The “correct answer” for each
          position is simply the word that actually came next in the text — which the internet
          already contains in unlimited supply. This is called <em>self-supervised</em> learning,
          and it’s why LLMs could be trained on trillions of words: the labels were free.
        </p>
      </Callout>

      <h2>Why “large”? The role of scale</h2>
      <p>
        Three quantities grew by orders of magnitude to turn next-token prediction from a party
        trick into something that feels intelligent:
      </p>
      <ul>
        <li>
          <strong>Parameters</strong> — the tunable weights. GPT-2 (2019) had 1.5&nbsp;billion;
          modern frontier models have hundreds of billions to trillions.
        </li>
        <li>
          <strong>Data</strong> — the training corpus. Measured in <em>tokens</em>, this went from
          billions to <em>tens of trillions</em>.
        </li>
        <li>
          <strong>Compute</strong> — arithmetic operations during training, measured in FLOPs. A
          frontier run can exceed <M>{"10^{25}"}</M> FLOPs and cost tens of millions of dollars.
        </li>
      </ul>
      <p>
        Remarkably, the relationship between these quantities and model quality is{" "}
        <em>smooth and predictable</em> — captured by <strong>scaling laws</strong> that we devote{" "}
        <button onClick={() => navigate("/chapter/scaling-laws")} className="prose-link">
          Chapter 17
        </button>{" "}
        to. That predictability is what justified the investment: you could forecast that a bigger
        model would be better <em>before</em> spending the money.
      </p>

      <h2>The full lifecycle</h2>
      <p>
        Building an LLM is a pipeline of distinct stages, each with its own goals, tools and
        failure modes. Here is the whole journey at a glance — every stage is a chapter (or several)
        in this book. Hover each step; click to jump ahead.
      </p>

      <Figure n="1.2" title="The end-to-end LLM lifecycle" caption="We will walk this pipeline left to right. Parts I–III give you the foundations and architecture; Part IV trains the base model; Part V aligns it into an assistant; Part VI evaluates and ships it.">
        <LifecyclePipeline onPick={(slug) => navigate(`/chapter/${slug}`)} />
      </Figure>

      <p>The stages, in one breath:</p>
      <ol>
        <li>
          <strong>Data.</strong> Scrape, filter and deduplicate trillions of tokens of text. Quality
          here caps quality everywhere downstream.
        </li>
        <li>
          <strong>Tokenize.</strong> Convert text into integer IDs the network can consume.
        </li>
        <li>
          <strong>Pretrain.</strong> Train a transformer to predict the next token over the whole
          corpus. This is 99% of the compute and produces a <em>base model</em> — fluent but
          unsteerable.
        </li>
        <li>
          <strong>Fine-tune (SFT).</strong> Show the base model thousands of high-quality
          instruction → response examples so it learns to <em>follow instructions</em>.
        </li>
        <li>
          <strong>Align.</strong> Use human preference data (RLHF or DPO) to make it helpful,
          honest and harmless.
        </li>
        <li>
          <strong>Evaluate.</strong> Measure capability, safety and regressions on benchmarks and
          with human judges.
        </li>
        <li>
          <strong>Deploy.</strong> Serve the model efficiently so users get fast, cheap tokens.
        </li>
      </ol>

      <Callout type="note" title="Base model vs. assistant">
        <p>
          A common point of confusion: the model that comes out of <em>pretraining</em> is not a
          chatbot. It’s an autocomplete engine that will happily continue your text in any
          direction. Turning it into the polite, instruction-following assistant you’re used to is
          the job of Part V. Same network, very different behavior.
        </p>
      </Callout>

      <h2>A two-minute history</h2>
      <p>
        It helps to know how we got here. The throughline is a steady move away from hand-engineered
        language rules toward learning everything from data at ever-greater scale.
      </p>

      <Figure n="1.3" title="Milestones on the road to modern LLMs">
        <ol className="relative ml-3 border-l border-white/10">
          {[
            ["2013", "word2vec", "Words become dense vectors; “king − man + woman ≈ queen.” Meaning becomes geometry."],
            ["2014–15", "Seq2seq + attention", "Encoder–decoder RNNs translate languages; attention lets the decoder “look back” at any input word."],
            ["2017", "“Attention Is All You Need”", "The Transformer drops recurrence entirely. Pure attention + MLPs, fully parallelizable. The architecture this whole book is about."],
            ["2018–19", "BERT & GPT-2", "Pretrain-then-finetune goes mainstream. GPT-2 shows scaled next-token prediction produces shockingly fluent text."],
            ["2020", "GPT-3 & scaling laws", "175B parameters. Few-shot “in-context learning” emerges. Bigger is reliably better."],
            ["2022", "InstructGPT / ChatGPT", "RLHF turns a base model into a helpful assistant. The interface moment that reached everyone."],
            ["2023→", "The open & frontier era", "LLaMA, Mistral, DeepSeek, Claude, GPT-4-class models; long context, multimodality, tool use, reasoning."],
          ].map(([year, title, body]) => (
            <li key={title} className="mb-5 ml-5">
              <span className="absolute -left-[7px] mt-1.5 h-3 w-3 rounded-full border-2 border-brand-400 bg-ink-900" />
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-xs text-brand-300">{year}</span>
                <span className="font-semibold text-slate-100">{title}</span>
              </div>
              <p className="mt-0.5 text-sm text-slate-400">{body}</p>
            </li>
          ))}
        </ol>
      </Figure>

      <h2>What you’ll actually build</h2>
      <p>
        This isn’t a survey. By the end you’ll have read — and understood every line of — a complete,
        runnable implementation of a GPT-style model in PyTorch. Here’s a taste of where we’re
        headed: the entire forward pass of a transformer language model is just this.
      </p>

      <CodeBlock language="python" filename="gpt.py (preview of Chapter 14)" highlight={[10, 11, 12]}>
{`import torch, torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # token  -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)   # position -> vector
        self.blocks  = nn.ModuleList(
            [Block(n_embd, n_head) for _ in range(n_layer)])  # the transformer stack
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # vector -> logits

    def forward(self, idx):                # idx: (batch, time) integer token IDs
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)                   # attention + MLP, repeated N times
        x = self.ln_f(x)
        return self.head(x)                # (batch, time, vocab_size) logits`}
      </CodeBlock>

      <p>
        If that looks like gibberish right now, good — that’s what the next thirteen chapters are
        for. Every symbol will become obvious. By Chapter 14 you’ll read this and think “of course.”
      </p>

      <h2>How to read this book</h2>
      <ul>
        <li>
          <strong>Don’t skip the visualizations.</strong> They’re not decoration — attention,
          tokenization and embeddings are far easier to <em>see</em> than to read about. Drag the
          sliders.
        </li>
        <li>
          <strong>The math is friendly.</strong> Every equation is explained in words. If you can
          follow the chain-rule line above, you’re fine.
        </li>
        <li>
          <strong>Read in order, at least once.</strong> Each part assumes the previous one. The
          payoff compounds.
        </li>
      </ul>

      <Callout type="tip" title="Ready?">
        <p>
          Next we open the engine: how a neural network actually turns numbers into learned behavior
          via the forward pass and backpropagation. That machinery powers every stage of the
          pipeline above.
        </p>
      </Callout>
    </>
  );
}
