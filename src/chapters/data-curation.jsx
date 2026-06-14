import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import DataFunnel from "../components/viz/data-curation/DataFunnel";

export default function Chapter() {
  return (
    <>
      <p>
        Before a single weight is initialized, the most consequential decision in training an LLM
        has already been made: <strong>what data it learns from</strong>. Architecture gets the
        headlines, but two models with identical architectures and compute can differ wildly in
        quality based purely on their training corpus. A model is, in a very real sense, a lossy
        compression of its data — it can only be as good, as broad, and as truthful as what it was
        fed. This chapter is about turning the chaotic, spam-ridden internet into a clean stream of
        tokens worth learning from.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          <strong>Data quality and scale cap everything downstream.</strong> No clever optimizer,
          fancy architecture, or longer training run can recover information that wasn’t in the
          corpus — or undo the damage of training on garbage. The frontier labs guard their data
          recipes more jealously than their model code. <em>Data is the real moat.</em>
        </p>
      </Callout>

      <h2>How much data? Counting in tokens</h2>
      <p>
        Pretraining corpora are measured in <strong>tokens</strong> (the subword units from{" "}
        <a href="/chapter/tokenization" className="prose-link">Chapter 5</a>), and the numbers are
        staggering. GPT-2 (2019) trained on roughly <M>{"10"}</M> billion tokens. GPT-3 (2020) used
        about <M>{"300"}</M> billion. Modern open models like LLaMA 3 train on{" "}
        <strong>15 trillion</strong> tokens, and the frontier is pushing toward tens of trillions.
        That’s the entire high-quality public web, several times over.
      </p>
      <p>
        Why so much more over time? A landmark result — the <strong>Chinchilla</strong> scaling laws
        (2022) — showed that the earlier giants were badly <em>under-trained</em>: for a fixed
        compute budget, you should grow the model and the data <em>together</em>. The rule of thumb
        it gave is roughly <strong>20 tokens of training data per model parameter</strong> for
        compute-optimal training. So a 70B-parameter model wants on the order of{" "}
        <M>{"70 \\times 10^9 \\times 20 \\approx 1.4"}</M> trillion tokens just to be{" "}
        <em>compute-optimal</em> — and far more if you want to keep squeezing out quality at
        inference time.
      </p>

      <Callout type="history" title="The Chinchilla shift">
        <p>
          Before 2022, the instinct was “bigger model, same-ish data.” Chinchilla flipped the field:
          a 70B model trained on 1.4T tokens beat the 280B Gopher trained on 300B tokens, using the
          same compute. The lesson — <em>feed your model more</em> — kicked off the race to scrape,
          clean, and curate trillions of tokens. We derive these laws properly in{" "}
          <a href="/chapter/scaling-laws" className="prose-link">Chapter 17</a>.
        </p>
      </Callout>

      <p>
        There’s a catch lurking here that motivates the rest of the chapter: high-quality text is{" "}
        <em>finite</em>. We are approaching the point where models consume a meaningful fraction of
        all the good text humans have ever written. When you can’t get more data, the only lever left
        is making the data you have <em>better</em> — which is exactly what curation does.
      </p>

      <h2>Where it comes from: the data mix</h2>
      <p>
        No single source is enough. A corpus is a deliberately weighted <strong>mixture</strong> of
        domains, each contributing a different competence. The proportions below are illustrative of
        a typical recipe (every lab’s is secret and different), but the shape is consistent across
        the field:
      </p>

      <table>
        <thead>
          <tr>
            <th>Source</th>
            <th>Rough share</th>
            <th>What it teaches</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Filtered web (Common Crawl, C4, FineWeb)</td>
            <td>~60–67%</td>
            <td>Breadth: general knowledge, style, the long tail of topics.</td>
          </tr>
          <tr>
            <td>Code (GitHub, StackExchange)</td>
            <td>~10–17%</td>
            <td>Reasoning, structure, syntax — and, oddly, better text reasoning too.</td>
          </tr>
          <tr>
            <td>Books (public-domain &amp; licensed)</td>
            <td>~8–12%</td>
            <td>Long-range coherence, narrative, deep domain prose.</td>
          </tr>
          <tr>
            <td>Academic (arXiv, papers)</td>
            <td>~2–5%</td>
            <td>Technical/scientific knowledge, math, formal argument.</td>
          </tr>
          <tr>
            <td>Q&amp;A / forums (Reddit, Stack Overflow)</td>
            <td>~3–5%</td>
            <td>Conversational, instructional, problem-solving registers.</td>
          </tr>
          <tr>
            <td>Wikipedia &amp; reference</td>
            <td>~2–4%</td>
            <td>Clean, factual, encyclopedic baseline — punches above its weight.</td>
          </tr>
        </tbody>
      </table>

      <p>
        Two ideas make this a <em>mix</em> rather than a pile. First, <strong>up-weighting</strong>:
        you can repeat (oversample) a small, high-quality source so it contributes more than its raw
        size. Wikipedia is tiny next to the web, but it’s clean enough that labs often run a few
        passes over it. Second, <strong>balance</strong>: too much code and the model writes prose
        like a compiler; too much web and it absorbs the internet’s worst habits. The mixture weights
        are hyperparameters, tuned by training small models and watching which blend yields the best
        evals.
      </p>

      <Callout type="industry" title="Code helps everything">
        <p>
          A counterintuitive, repeatedly-observed result: adding a healthy fraction of source code to
          the pretraining mix improves performance on <em>non-code</em> reasoning tasks — math word
          problems, logical deduction, structured generation. The leading theory is that code is
          dense, unambiguous, and full of explicit step-by-step structure, which teaches the model to
          reason more rigorously. Almost every serious model now includes 10%+ code, even ones never
          meant to program.
        </p>
      </Callout>

      <h2>The cleaning pipeline</h2>
      <p>
        Raw web data is mostly junk: HTML chrome, navigation menus, spam, machine-translated sludge,
        adult content, and endless duplicates. The job of the data pipeline is to distill this
        firehose into a clean corpus. It’s a sequence of filtering stages, each one cheap enough to
        run over <em>trillions</em> of documents and each one throwing away a large fraction of what
        came in. The standard stages, in order:
      </p>
      <ol>
        <li>
          <strong>Extraction</strong> — pull readable text out of raw HTML, discarding tags,
          scripts, boilerplate, and navigation.
        </li>
        <li>
          <strong>Language identification</strong> — classify each document’s language; keep the
          ones you want (e.g. English, or a chosen multilingual set).
        </li>
        <li>
          <strong>Quality filtering</strong> — drop low-quality text using heuristics and trained
          classifiers (covered below).
        </li>
        <li>
          <strong>Deduplication</strong> — remove exact and near-duplicate documents.
        </li>
        <li>
          <strong>PII &amp; toxicity removal</strong> — scrub personal data (emails, phone numbers,
          SSNs) and filter the most toxic/unsafe content.
        </li>
        <li>
          <strong>Decontamination</strong> — remove any text that overlaps with benchmark test sets,
          so evaluation stays honest.
        </li>
      </ol>
      <p>
        The cumulative effect is dramatic. Watch the corpus collapse stage by stage — and click each
        bar to see a concrete example of what gets thrown away.
      </p>

      <Figure
        n="4.1"
        title="The data funnel"
        caption="A raw web crawl shrinks to a fraction of its original size as it passes through extraction, language ID, quality filtering, and deduplication. The exact fractions vary by pipeline, but losing 80–90% of the raw bytes is entirely normal — most of the web simply isn't worth training on. Flip on the MinHash sketch to see how near-duplicates are caught."
      >
        <DataFunnel />
      </Figure>

      <h2>Deduplication</h2>
      <p>
        The web is profoundly repetitive. The same article is mirrored across hundreds of sites; SEO
        farms reword each other; licenses, terms-of-service, and product descriptions appear
        verbatim millions of times. Left in, duplicates cause three distinct harms:
      </p>
      <ul>
        <li>
          <strong>Memorization &amp; privacy.</strong> Text seen many times gets memorized verbatim
          rather than generalized — including any private data it contains, which the model can later
          regurgitate.
        </li>
        <li>
          <strong>Wasted compute.</strong> Every duplicate copy is gradient steps spent re-learning
          something you already know, instead of seeing new information.
        </li>
        <li>
          <strong>Evaluation leakage.</strong> If a duplicated passage happens to be a benchmark
          question, your test scores become fiction (more on this under decontamination).
        </li>
      </ul>
      <p>
        Empirically, deduplicating the training set <em>improves</em> the model while <em>shrinking</em>{" "}
        the data — one of the rare free lunches in ML. There are two flavors.
      </p>

      <h3>Exact deduplication</h3>
      <p>
        The easy case: byte-for-byte identical documents (or substrings). Hash every document with a
        strong hash (SHA-256), and keep only the first occurrence of each hash. Fast, simple, and it
        catches verbatim mirrors. But it misses documents that differ by a single character — a
        timestamp, a tracking ID, one reworded sentence — which are still effectively duplicates.
      </p>

      <h3>Near-duplicate detection: MinHash + LSH</h3>
      <p>
        To catch <em>approximate</em> duplicates at web scale, we need a way to estimate document
        similarity without the impossible cost of comparing all <M>{"\\binom{N}{2}"}</M> pairs. The
        classic recipe is <strong>MinHash</strong> + <strong>Locality-Sensitive Hashing (LSH)</strong>.
        It rests on four ideas:
      </p>
      <ul>
        <li>
          <strong>Shingling.</strong> Represent each document as a <em>set</em> of overlapping
          k-grams (“shingles”) — e.g. all consecutive 5-word sequences. Two similar documents share
          most of their shingles.
        </li>
        <li>
          <strong>Jaccard similarity.</strong> Measure set overlap. For shingle sets{" "}
          <M>{"A"}</M> and <M>{"B"}</M>:
          <MB>{String.raw`J(A, B) = \frac{|A \cap B|}{|A \cup B|}`}</MB>
          which is <M>{"1"}</M> for identical sets and <M>{"0"}</M> for disjoint ones. We want to
          flag pairs with <M>{"J"}</M> above a threshold (say <M>{"0.8"}</M>).
        </li>
        <li>
          <strong>MinHashing.</strong> Computing exact Jaccard is expensive, so we approximate it.
          Apply <M>{"k"}</M> independent hash functions to a document’s shingles; for each function,
          keep only the <em>minimum</em> hash value. This length-<M>{"k"}</M> vector is the document’s{" "}
          <strong>MinHash signature</strong>. The beautiful fact: for any one hash function, the
          probability that two documents share the same minimum equals their Jaccard similarity:
          <MB>{String.raw`\Pr[\,\min h(A) = \min h(B)\,] = J(A, B)`}</MB>
          So the fraction of matching signature slots is an unbiased estimate of <M>{"J(A,B)"}</M> —
          comparing two short signatures instead of two huge sets.
        </li>
        <li>
          <strong>Banding (LSH).</strong> Even comparing all signature pairs is too many. Split each
          signature into <M>{"b"}</M> bands of <M>{"r"}</M> rows each. Hash each band into a bucket;
          two documents become <em>candidates</em> only if they collide in at least one band. Tuning{" "}
          <M>{"b"}</M> and <M>{"r"}</M> sets the similarity threshold sharply: the probability a pair
          becomes a candidate is <M>{"1 - (1 - J^{r})^{b}"}</M>, an S-curve that you place wherever
          you want the cutoff.
        </li>
      </ul>

      <Callout type="math" title="Why MinHash works in one line">
        <p>
          Under a random hash, the element with the smallest hash value among <M>{"A \\cup B"}</M> is
          equally likely to be any element of the union. That minimum-valued element lies in{" "}
          <M>{"A \\cap B"}</M> with probability exactly <M>{"|A \\cap B| / |A \\cup B| = J(A,B)"}</M>{" "}
          — and that is precisely the event <M>{"\\min h(A) = \\min h(B)"}</M>. Repeat over{" "}
          <M>{"k"}</M> hashes and average to drive the variance down.
        </p>
      </Callout>

      <h2>Quality filtering</h2>
      <p>
        “Quality” is fuzzy, so we attack it from two angles: cheap heuristics that reject obvious
        garbage, and a learned classifier that captures the rest.
      </p>

      <h3>Heuristic filters</h3>
      <p>
        These are dumb, fast rules applied to every document. None is perfect alone, but together
        they sweep away the worst of the web. Typical checks:
      </p>
      <ul>
        <li>
          <strong>Length.</strong> Reject documents that are too short (a few words — no signal) or
          absurdly long in a way that signals a dump or log file.
        </li>
        <li>
          <strong>Symbol &amp; digit ratios.</strong> Real prose has a low ratio of punctuation,
          symbols, and digits to letters. A high ratio flags code-as-noise, spam, or tables of
          numbers.
        </li>
        <li>
          <strong>Stopword presence.</strong> Natural language is full of function words (“the”,
          “and”, “of”). Their <em>absence</em> is a strong signal of keyword spam or a non-prose
          dump. Gopher’s rule: require at least two of a stopword list to be present.
        </li>
        <li>
          <strong>Repetition.</strong> Drop documents with too many repeated lines, paragraphs, or
          n-grams (a classic spam and template signature).
        </li>
        <li>
          <strong>Perplexity filtering.</strong> Score each document with a small reference language
          model (e.g. an n-gram model trained on Wikipedia). Very <em>high</em> perplexity means
          “unlike clean text” — gibberish, OCR errors, broken encoding — so drop the tail.
        </li>
      </ul>

      <h3>Classifier-based filtering</h3>
      <p>
        Heuristics miss subtler junk. The trick — popularized by GPT-3 and now standard — is to train
        a lightweight binary classifier to recognize “good” text. Take a set of known-high-quality
        documents (Wikipedia, well-curated books, link-rich Reddit posts) as <em>positives</em>, and
        a random sample of raw Common Crawl as <em>negatives</em>. A fast linear model over n-gram
        features (or a tiny neural net) learns to score how “curated-looking” any new document is. You
        then keep documents above a threshold — often with a touch of randomness so you don’t discard
        every borderline doc and collapse diversity.
      </p>

      <CodeBlock language="python" filename="quality_filter.py">
{`import re
from collections import Counter

STOPWORDS = {"the", "be", "to", "of", "and", "a", "in", "that",
             "have", "it", "for", "on", "with", "as", "you", "is"}

def passes_quality(text: str) -> bool:
    """Heuristic quality gate. Returns False for likely-junk documents.
    Thresholds here mirror common pipelines (Gopher / C4 / FineWeb)."""
    words = text.split()
    n = len(words)

    # 1) Length: too short carries no signal; extreme length is often a dump.
    if n < 50 or n > 100_000:
        return False

    # 2) Mean word length: real prose sits ~3-10 chars/word. Outside => junk.
    mean_word_len = sum(len(w) for w in words) / n
    if mean_word_len < 3 or mean_word_len > 10:
        return False

    # 3) Symbol ratio: lots of #, *, |, ... relative to letters => spam/markup.
    n_letters = sum(c.isalpha() for c in text)
    n_symbols = sum(c in "#*|{}<>~^=" for c in text)
    if n_letters == 0 or n_symbols / n_letters > 0.10:
        return False

    # 4) Stopwords: natural language must contain common function words.
    lower = {w.lower() for w in words}
    if len(STOPWORDS & lower) < 2:
        return False

    # 5) Repetition: dedup of lines; >30% duplicate lines => boilerplate spam.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        counts = Counter(lines)
        dup_fraction = 1 - len(counts) / len(lines)
        if dup_fraction > 0.30:
            return False

    return True


# ---- Near-duplicate detection: a minimal MinHash sketch ----
def shingles(text: str, k: int = 5) -> set[str]:
    """Set of overlapping k-word shingles."""
    w = text.split()
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}

def minhash(text: str, num_hashes: int = 64) -> list[int]:
    """MinHash signature: per hash function, the minimum shingle hash."""
    sh = shingles(text)
    sig = []
    for seed in range(num_hashes):
        # hash(shingle) salted by seed; keep the smallest -> one signature slot
        sig.append(min(hash((seed, s)) & 0xFFFFFFFF for s in sh) if sh else 0)
    return sig

def estimated_jaccard(a: str, b: str) -> float:
    """Fraction of matching signature slots ~= Jaccard similarity J(a, b)."""
    sa, sb = minhash(a), minhash(b)
    matches = sum(x == y for x, y in zip(sa, sb))
    return matches / len(sa)

# Two near-duplicates score high; bucket them via LSH and keep just one.
doc1 = "Breaking news: the central bank raised interest rates today by half a point."
doc2 = "Breaking news: the central bank raised interest rates today by 0.5 percent."
if estimated_jaccard(doc1, doc2) > 0.7:
    drop(doc2)   # near-duplicate -> remove`}
      </CodeBlock>

      <Callout type="pitfall" title="Filters can be biased — and a sieve, not a wall">
        <p>
          A quality classifier trained on “Wikipedia &amp; Reddit = good” will quietly penalize
          dialects, regional varieties, and registers under-represented in those positives, narrowing
          what the model can fluently produce. And every heuristic threshold is a blunt instrument
          that discards good text along with bad. Treat filtering as <em>probabilistic</em>: keep some
          randomness, monitor what you’re throwing away, and audit the survivors — not just the
          rejects.
        </p>
      </Callout>

      <h2>Decontamination</h2>
      <p>
        We judge models with <strong>benchmarks</strong> — fixed test sets like MMLU, GSM8K, or
        HumanEval. The unspoken contract is that the model has never seen the answers. But those
        benchmarks are <em>published on the internet</em>, which means copies leak into Common Crawl.
        If benchmark questions (and answers) sit in your training data, the model can simply{" "}
        <strong>memorize</strong> them, and your eval scores become a measure of memorization, not
        capability — a phenomenon called <strong>contamination</strong> or <strong>train/test
        leakage</strong>.
      </p>
      <p>
        <strong>Decontamination</strong> is the defensive scrub: before training, take every test
        example from the benchmarks you care about and remove any training document that overlaps it.
        Overlap is usually detected with n-gram matching (e.g. flag any training document sharing a
        13-gram with a test question) or the same MinHash machinery used for dedup. It’s never
        perfect — paraphrases slip through, and new benchmarks appear after training — which is why
        the community increasingly prizes <em>held-out</em> and freshly-created evals.
      </p>

      <Callout type="warning" title="Contamination inflates scores — silently">
        <p>
          A contaminated model looks brilliant on the leaderboard and disappoints in the wild. Worse,
          contamination is easy to do <em>by accident</em>: you swap in a new web snapshot, forget to
          re-run decontamination, and your reported numbers quietly inflate. Always decontaminate
          against your full eval suite, document exactly what you removed, and be suspicious of any
          benchmark result you can’t reproduce on fresh data.
        </p>
      </Callout>

      <h2>Tokenizing &amp; packing into sequences</h2>
      <p>
        Once the corpus is clean, it becomes the model’s input. Each document is run through the
        tokenizer, turning text into integer IDs. To mark where one document ends and the next
        begins, we append a special <code>&lt;|endoftext|&gt;</code> token — the boundary signal the
        model learns to treat as “a new, unrelated document starts here.”
      </p>
      <p>
        Then comes <strong>packing</strong>. Transformers train on fixed-length sequences (the{" "}
        <em>block size</em> or context length, e.g. <M>{"1024"}</M> or <M>{"8192"}</M> tokens). Rather
        than padding each short document — which wastes compute on padding tokens — we{" "}
        <em>concatenate</em> the entire tokenized corpus into one gigantic stream, then slice it into
        contiguous chunks of exactly <code>block_size</code> tokens. The <code>&lt;|endoftext|&gt;</code>{" "}
        separators are the only hint of original document boundaries; a single training chunk may span
        the tail of one document and the head of the next.
      </p>

      <CodeBlock language="python" filename="pack.py">
{`import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token            # the <|endoftext|> id (50256 for gpt2)
BLOCK_SIZE = 1024

# 1) Tokenize every document, separating them with the EOT token.
ids = []
for doc in clean_corpus:              # an iterable of cleaned strings
    ids.extend(enc.encode_ordinary(doc))
    ids.append(EOT)                   # mark the document boundary

# 2) Concatenate into one flat array of token ids (uint16 fits gpt2's vocab).
ids = np.array(ids, dtype=np.uint16)

# 3) Drop the ragged tail and reshape into fixed-length training blocks.
n_blocks = len(ids) // BLOCK_SIZE
blocks = ids[: n_blocks * BLOCK_SIZE].reshape(n_blocks, BLOCK_SIZE)

# Each row is one training example. The target is the input shifted by one:
#   inputs  = block[:-1]   targets = block[1:]   (next-token prediction)
np.save("train.npy", blocks)
print(f"{blocks.shape[0]:,} blocks of {BLOCK_SIZE} tokens "
      f"= {blocks.size / 1e9:.2f}B training tokens")`}
      </CodeBlock>

      <p>
        That <code>train.npy</code> — a dense array of token IDs — is the literal input to
        pretraining. Every gradient step samples a batch of these fixed-length blocks and asks the
        model to predict each token from the ones before it. The messy internet has become a tidy
        tensor.
      </p>

      <h2>Data is the moat</h2>
      <p>
        Step back and notice what we did: we took the raw, hostile sprawl of the web and, through
        extraction, language ID, quality filtering, deduplication, decontamination, and packing,
        forged a clean stream of tokens. That stream — its scale, its mixture, its cleanliness — sets
        the ceiling on everything that follows. Two labs with the same GPUs and the same transformer
        code will produce very different models if one of them has a better corpus. That’s why the
        recipe is the secret, and why <strong>data is the real moat</strong>.
      </p>
      <p>
        We’ve been talking about tokens as if they already exist. They don’t yet — turning text into
        those integer IDs is its own rich problem, full of trade-offs that ripple through the whole
        model. Next, we build a tokenizer from scratch and watch <strong>Byte-Pair Encoding</strong>{" "}
        learn its vocabulary, before giving each of those IDs a meaningful <strong>vector</strong> in{" "}
        <a href="/chapter/embeddings" className="prose-link">the embeddings chapter</a>.
      </p>
    </>
  );
}
