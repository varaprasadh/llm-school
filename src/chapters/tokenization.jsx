import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import BPETokenizer from "../components/viz/BPETokenizer";
import LineChart from "../components/viz/LineChart";

export default function Chapter() {
  // Illustrative: average tokens-per-word shrinks as vocabulary grows.
  const tradeoff = [
    [256, 4.1], [1000, 2.6], [4000, 1.8], [8000, 1.5],
    [16000, 1.35], [32000, 1.25], [50000, 1.18], [100000, 1.12],
  ];

  return (
    <>
      <p>
        A neural network can’t read text — it eats numbers. <strong>Tokenization</strong> is the
        bridge: the deterministic process that turns a string like{" "}
        <code>"unbelievable"</code> into a short list of integers the model can embed. Get it wrong
        and you waste context, mangle code and bloat your vocabulary. Get it right and it disappears
        into the background. Let’s build one from scratch.
      </p>

      <h2>Why not just use words? Or characters?</h2>
      <p>The two obvious choices both fail, in opposite directions:</p>
      <ul>
        <li>
          <strong>Word-level.</strong> One ID per word. Problem: vocabulary is unbounded (new words,
          typos, <code>covfefe</code>), morphology is lost (<em>run/running/ran</em> look unrelated),
          and you need a dreaded <code>&lt;UNK&gt;</code> token for anything unseen.
        </li>
        <li>
          <strong>Character-level.</strong> One ID per character — tiny vocabulary, nothing is ever
          unknown. Problem: sequences become brutally long (every letter is a step), so the model
          spends its limited context and compute spelling instead of thinking.
        </li>
      </ul>
      <p>
        The winning compromise is <strong>subword</strong> tokenization: common words stay whole,
        rare words split into reusable pieces. <code>"unbelievable"</code> →{" "}
        <code>un</code> + <code>believ</code> + <code>able</code>. Frequent enough to be efficient,
        granular enough to never be stuck.
      </p>

      <Callout type="key" title="The goal">
        <p>
          Find a fixed vocabulary of ~30k–100k subword units such that <em>any</em> string can be
          encoded, common text uses few tokens, and morphologically related words share pieces. The
          algorithm that does this is <strong>Byte-Pair Encoding</strong> (BPE).
        </p>
      </Callout>

      <h2>The vocabulary trade-off</h2>
      <p>
        Vocabulary size is a dial. Bigger vocab → fewer tokens per sentence (cheaper, longer
        effective context) but a larger embedding table and softmax, and rarer tokens that are
        harder to learn. Smaller vocab → the opposite. Most modern models land between 32k and 128k.
      </p>

      <Figure
        n="5.1"
        title="Average tokens per word vs. vocabulary size"
        caption="As the vocabulary grows, more whole words and word-pieces fit, so each word costs fewer tokens — with diminishing returns. The embedding and output-projection cost, meanwhile, grows linearly with vocab size, so you stop well before 'one token per word.'"
      >
        <LineChart
          series={[{ label: "tokens / word", color: "#22d3ee", points: tradeoff }]}
          xScale="log"
          xLabel="vocabulary size"
          yLabel="avg tokens per word"
          fmtX={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : `${Math.round(v)}`)}
          height={300}
        />
      </Figure>

      <h2>Byte-Pair Encoding, intuitively</h2>
      <p>
        BPE was originally a 1994 <em>compression</em> algorithm. The idea is delightfully simple:
        start with the smallest possible units (characters), then repeatedly find the most frequent
        adjacent pair and merge it into a new unit. Do this <M>{"N"}</M> times and you’ve “grown” a
        vocabulary of <M>{"N"}</M> learned merges on top of the base characters.
      </p>
      <p>The training loop is just four steps, repeated:</p>
      <ol>
        <li>Split the corpus into words; represent each word as a sequence of characters.</li>
        <li>Count every adjacent symbol pair across the whole corpus.</li>
        <li>Merge the single most frequent pair into a new symbol, everywhere.</li>
        <li>Record the merge rule. Repeat until you hit your target vocabulary size.</li>
      </ol>

      <h2>Watch it learn, then watch it tokenize</h2>
      <p>
        The widget below is a real BPE implementation running on a tiny corpus. In{" "}
        <strong>Learn merges</strong>, step through training and watch frequent pairs like{" "}
        <code>e</code>+<code>r</code> and <code>n</code>+<code>e</code> fuse into reusable subwords.
        Then switch to <strong>Tokenize text</strong> and type anything — the learned merges encode
        it, falling back to characters for unseen strings (so nothing is ever “unknown”).
      </p>

      <Figure
        n="5.2"
        title="Byte-Pair Encoding from scratch"
        caption="The exact algorithm used (with byte-level tweaks) by GPT-2, GPT-4, LLaMA and most modern models. The merge rules learned during training are saved and shipped with the model — encoding is just replaying them in order."
      >
        <BPETokenizer />
      </Figure>

      <h2>Byte-level BPE: handling literally anything</h2>
      <p>
        What about emoji, Chinese characters, or a corrupted byte? GPT-2 introduced a clever fix:
        run BPE not over Unicode characters but over raw <strong>bytes</strong>. There are only 256
        possible bytes, so the base vocabulary is always exactly 256 symbols, and <em>any</em>{" "}
        possible string — any language, any emoji, any binary — is guaranteed encodable with zero
        unknown tokens. Merges then build up common byte sequences into words.
      </p>

      <Callout type="industry" title="A token is ~¾ of a word">
        <p>
          A useful rule of thumb for English: <strong>1 token ≈ 4 characters ≈ 0.75 words</strong>.
          So 1,000 tokens is roughly 750 words. This is exactly what you’re billed for by API
          providers, and what fills a model’s context window. Code, non-English languages and unusual
          formatting tokenize less efficiently — sometimes 1 token per character.
        </p>
      </Callout>

      <h2>Special tokens</h2>
      <p>
        Beyond text, we reserve a handful of <strong>special tokens</strong> with structural meaning.
        These are added to the vocabulary by hand and the model learns what they imply:
      </p>
      <ul>
        <li>
          <code>&lt;|endoftext|&gt;</code> / <code>&lt;/s&gt;</code> — document boundary; tells the
          model “a new, unrelated document starts here.”
        </li>
        <li>
          <code>&lt;|pad|&gt;</code> — filler so sequences in a batch are the same length (masked
          out of the loss).
        </li>
        <li>
          Chat roles like <code>&lt;|user|&gt;</code>, <code>&lt;|assistant|&gt;</code> — structure
          conversations during fine-tuning (
          <a href="/chapter/supervised-finetuning" className="prose-link">Chapter 18</a>).
        </li>
      </ul>

      <h2>In practice: use a fast, tested implementation</h2>
      <p>
        You’ll <em>understand</em> BPE by writing it, but in production you’ll use a battle-tested
        library — OpenAI’s <code>tiktoken</code> or Hugging Face <code>tokenizers</code> (Rust-fast).
        Here’s both encoding and training:
      </p>

      <CodeBlock language="python" filename="tokenize.py">
{`# --- Using a pretrained tokenizer (inference) ---
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")   # GPT-4's tokenizer
ids = enc.encode("Tokenization is the bridge to numbers.")
print(ids)            # [3404, 2065, 374, 279, 14497, 311, 5219, 13]
print(len(ids))       # 8 tokens
print(enc.decode(ids))  # round-trips back to the original string

# --- Training your own BPE on a corpus ---
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
)
tok.train(["corpus.txt"], trainer)
tok.save("my-tokenizer.json")`}
      </CodeBlock>

      <Callout type="pitfall" title="Tokenizer mistakes are forever">
        <p>
          The tokenizer is frozen <em>before</em> pretraining and can essentially never change —
          every weight is learned against these exact token IDs. A bad vocabulary (too small,
          English-only, bad with code or numbers) handicaps the model permanently. Famously, poor
          number tokenization is a big reason early LLMs were bad at arithmetic: <code>327</code>{" "}
          might be one token but <code>328</code> three. Choose and test it carefully.
        </p>
      </Callout>

      <h2>Where we are</h2>
      <p>
        Text is now a sequence of integer IDs in <M>{"[0, V)"}</M> where <M>{"V"}</M> is the
        vocabulary size. But integers carry no meaning — ID 4087 isn’t “bigger” than ID 12 in any
        useful sense. Next we give each token a learnable <strong>vector</strong>, turning discrete
        IDs into points in a continuous space where geometry encodes meaning.
      </p>
    </>
  );
}
