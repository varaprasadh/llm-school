import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import ChatTemplate from "../components/viz/supervised-finetuning/ChatTemplate";

export default function Chapter() {
  return (
    <>
      <p>
        At the end of pretraining you have a <strong>base model</strong>: a
        breathtakingly capable next-token predictor that has read a good slice of
        the internet. It can complete code, finish sentences, and mimic almost
        any style. There is just one problem — it does not <em>follow
        instructions</em>. Ask it “What is the capital of France?” and a base
        model is just as likely to continue with “What is the capital of Germany?
        What is the capital of Spain?” because, statistically, that is the kind of
        text that follows a list of trivia questions. It is a brilliant
        autocomplete that has never been told its job is to be helpful.
      </p>
      <p>
        <strong>Supervised fine-tuning (SFT)</strong> — also called{" "}
        <em>instruction tuning</em> — is the first and simplest step of
        post-training that fixes this. We keep doing exactly what pretraining did
        (predict the next token, minimize cross-entropy), but we change the{" "}
        <em>data</em>: instead of raw web text, we train on a curated set of
        <em> demonstrations</em> of the behavior we want — prompts paired with
        ideal responses. With one clever twist (loss masking, which we’ll spend
        real time on), a few thousand good examples are enough to transform the
        base model into an instruction-follower.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          SFT is ordinary next-token training on{" "}
          <em>(prompt, ideal-response)</em> pairs, with the loss computed{" "}
          <strong>only on the response tokens</strong>. You are not teaching the
          model new facts so much as teaching it a new <em>format</em>: “when you
          see a user request in this chat layout, produce a helpful answer and
          then stop.”
        </p>
      </Callout>

      <h2>Base models vs. instruct models</h2>
      <p>
        It’s worth being precise about the two artifacts, because people conflate
        them constantly. They share the same architecture and often the same
        weights as a starting point; they differ only in what they were trained
        on last.
      </p>
      <ul>
        <li>
          A <strong>base / foundation model</strong> (e.g. <code>Llama-3-8B</code>,
          the un-suffixed checkpoint) is the raw pretraining output. It models{" "}
          <M>{"p(\\text{next token} \\mid \\text{text so far})"}</M> over generic
          text. It has no notion of “user” or “assistant”, no special chat tokens,
          and no instinct to stop talking.
        </li>
        <li>
          An <strong>instruct / chat model</strong> (e.g.{" "}
          <code>Llama-3-8B-Instruct</code>) is that base model after SFT (and
          usually preference tuning on top — the subject of{" "}
          <a href="/chapter/rlhf" className="prose-link">
            Chapter 19
          </a>
          ). It understands a conversation structure, answers the question you
          actually asked, and emits an end-of-turn token when it’s done.
        </li>
      </ul>
      <p>
        The capabilities — the knowledge, the reasoning, the latent skills — were
        almost entirely learned during pretraining. SFT mostly{" "}
        <em>surfaces and steers</em> abilities the base model already had, aligning
        its output distribution with the helpful-assistant format. This is why a
        surprisingly small amount of SFT data can have such a large effect:
        you’re not building the engine, you’re installing the steering wheel.
      </p>

      <h2>Instruction datasets</h2>
      <p>
        The fuel for SFT is a dataset of demonstrations. The canonical schema, made
        popular by Stanford’s Alpaca, is the{" "}
        <strong>instruction / input / output triple</strong>:
      </p>
      <CodeBlock language="json" filename="sample.jsonl">
{`{
  "instruction": "Translate the sentence to French.",
  "input": "I would like a coffee, please.",
  "output": "Je voudrais un café, s'il vous plaît."
}
{
  "instruction": "Summarize the paragraph in one sentence.",
  "input": "The mitochondria is the powerhouse of the cell. It ...",
  "output": "Mitochondria generate most of the cell's chemical energy."
}`}
      </CodeBlock>
      <p>
        The <code>instruction</code> is the task, the optional <code>input</code>{" "}
        is the data the task operates on (empty for open-ended prompts like “Write
        a poem about autumn”), and the <code>output</code> is the gold response we
        want the model to learn to produce. Where do millions of these come from?
        Three broad sources, usually blended:
      </p>
      <ul>
        <li>
          <strong>Human-written.</strong> Annotators write prompts and ideal
          answers by hand. Highest quality and most expensive; this is what
          OpenAI’s original InstructGPT and Google’s FLAN collections leaned on.
        </li>
        <li>
          <strong>Synthetic / self-instruct.</strong> Bootstrap a large, diverse
          set by prompting a capable LLM to <em>generate</em> new
          instruction–response pairs from a small seed set. The{" "}
          <em>Self-Instruct</em> method and Alpaca (52K examples generated from{" "}
          <code>text-davinci-003</code> off 175 seed tasks) showed this works
          shockingly well for a few hundred dollars.
        </li>
        <li>
          <strong>Distilled.</strong> Collect high-quality outputs from a stronger
          “teacher” model and fine-tune a smaller “student” on them (e.g. Vicuna
          trained on ShareGPT conversations). Cheap and effective, though it
          inherits the teacher’s style and biases — and may run into the teacher’s
          terms of service.
        </li>
      </ul>

      <Callout type="industry" title="Quality &gt; quantity: the LIMA finding">
        <p>
          Meta’s <strong>LIMA</strong> (“Less Is More for Alignment”, 2023)
          fine-tuned a 65B base model on just <strong>1,000</strong> carefully
          curated, hand-written examples — no RLHF — and it was competitive with
          far more heavily trained assistants. The lesson, repeated across the
          industry since: a small set of <em>diverse, high-quality,
          consistently-formatted</em> demonstrations beats a giant pile of noisy
          ones. For SFT, you curate; you don’t just scrape. Most of the “alignment”
          a model needs is a thin layer over pretraining.
        </p>
      </Callout>

      <h2>Chat templates: structuring the conversation</h2>
      <p>
        A base model sees an undifferentiated stream of tokens. To hold a
        multi-turn conversation, we need to tell the model <em>who is speaking</em>
        — system, user, or assistant — and where each turn begins and ends. We do
        this with a <strong>chat template</strong>: a fixed convention that wraps
        each turn in <strong>special tokens</strong>. One widely used format is
        ChatML, which uses <code>{"<|im_start|>"}</code> and{" "}
        <code>{"<|im_end|>"}</code> (im = “instant message”):
      </p>
      <CodeBlock language="text" filename="chatml.txt">
{`<|im_start|>system
You are a concise assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>`}
      </CodeBlock>
      <p>
        The <strong>system</strong> turn sets the persona and rules; the{" "}
        <strong>user</strong> turn is the request; the <strong>assistant</strong>{" "}
        turn is the model’s reply. These special tokens are added to the
        tokenizer’s vocabulary as single, atomic ids — the model learns dedicated
        embeddings for them, so they act as unambiguous “role” delimiters rather
        than ordinary text. Critically, the model learns to emit{" "}
        <code>{"<|im_end|>"}</code> at the end of its reply; that token is what
        tells the inference loop to <em>stop generating</em>. Different model
        families use different markers (Llama-3 uses{" "}
        <code>{"<|start_header_id|>"}</code>/<code>{"<|eot_id|>"}</code>), but the
        idea is identical. You should never hand-format these strings yourself —
        always use the tokenizer’s built-in template.
      </p>

      <Callout type="pitfall" title="Use the model's own template, exactly">
        <p>
          A model only behaves well with the <em>exact</em> chat template it was
          trained on — right special tokens, right whitespace, right role names. A
          single wrong newline or a missing <code>{"<|im_end|>"}</code> can
          noticeably degrade quality, because you’ve pushed the input
          off-distribution. In practice you call{" "}
          <code>tokenizer.apply_chat_template(messages, ...)</code> and let the
          library reproduce the format byte-for-byte.
        </p>
      </Callout>

      <h2>Loss masking: train only on the response</h2>
      <p>
        Here is the single most important mechanical idea in this chapter. When we
        fine-tune, we feed the model the <em>entire</em> formatted conversation —
        system, user, and assistant tokens all in one sequence — and run a normal
        forward pass. But we do <strong>not</strong> want the model to be penalized
        for failing to predict the <em>user’s</em> question or the system prompt.
        Those tokens are the <em>input</em>; we don’t want to teach the model to
        generate user requests. We only want to teach it to generate the{" "}
        <strong>assistant’s response</strong>.
      </p>
      <p>
        So we <strong>mask</strong> the loss on every non-assistant token. Recall
        the standard language-modeling loss is the average negative log-likelihood
        over a sequence of length <M>{"T"}</M>:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\left(x_t \mid x_{<t}\right)`}</MB>
      <p>
        For SFT we keep the model conditioned on the whole prefix{" "}
        <M>{"x_{<t}"}</M> (so it <em>reads</em> the prompt), but we sum the loss{" "}
        <em>only over the assistant’s response positions</em>. Let{" "}
        <M>{"\\mathcal{R}"}</M> be the set of token positions belonging to the
        assistant’s reply. The masked objective is:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{R}|}\sum_{t \in \mathcal{R}} \log p_\theta\!\left(x_t \mid x_{<t}\right)`}</MB>
      <p>
        Equivalently, with a binary mask <M>{"m_t \\in \\{0,1\\}"}</M> that is 1 on
        response tokens and 0 everywhere else:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{SFT}} = -\frac{\sum_{t=1}^{T} m_t \,\log p_\theta\!\left(x_t \mid x_{<t}\right)}{\sum_{t=1}^{T} m_t}`}</MB>
      <p>
        Tokens with <M>{"m_t = 0"}</M> contribute zero to the loss and zero to the
        gradient. The conventional way to implement this is to set the{" "}
        <em>label</em> of every masked position to a sentinel value —{" "}
        <strong>−100</strong> in PyTorch / Hugging Face — which{" "}
        <code>cross_entropy</code> is hardwired to ignore via its{" "}
        <code>ignore_index</code> argument. The figure below makes the masking
        concrete: toggle the highlight to see which tokens are{" "}
        <span className="text-emerald-300">trained</span> (label = token id) versus{" "}
        <span className="text-slate-400">masked</span> (label = −100).
      </p>

      <Figure
        n="18.1"
        title="Chat template + loss masking"
        caption="The full conversation is one token sequence. The model reads all of it, but loss is computed only on the assistant's response (green). System, user, role markers and structural tokens are masked to −100 (grey), so no gradient flows from them. Hover any token to inspect its training target."
      >
        <ChatTemplate />
      </Figure>

      <Callout type="key" title="Why masking matters so much">
        <p>
          Without masking, the model spends most of its gradient learning to
          predict user questions and boilerplate it will <em>never need to
          generate</em> — diluting the signal and even teaching bad habits (like
          asking itself questions). Masking concentrates all the learning on the
          one behavior you care about: producing the assistant turn. It’s a tiny
          change to the labels with an outsized effect on the result.
        </p>
      </Callout>

      <h2>Building a training example in code</h2>
      <p>
        Let’s turn theory into the exact tensor a trainer consumes. We tokenize the
        prompt and the response separately, concatenate them into{" "}
        <code>input_ids</code>, and build a parallel <code>labels</code> array
        where every prompt position is <code>−100</code> and every response
        position carries the real token id. The model is trained to predict the
        next token, so labels are simply the inputs shifted by one — which Hugging
        Face’s <code>...ForCausalLM</code> models do internally, letting us pass{" "}
        <code>labels</code> aligned with <code>input_ids</code>.
      </p>

      <CodeBlock language="python" filename="build_example.py" highlight={[26, 27, 28]}>
{`import torch

IGNORE_INDEX = -100  # cross_entropy skips these positions

def build_sft_example(tokenizer, system, user, assistant):
    # Format the prompt (everything the model READS but is not trained on)...
    prompt_msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # add_generation_prompt=True appends the "<|im_start|>assistant\\n" header,
    # so the prompt ends exactly where the model should start writing.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, tokenize=True,
    )

    # ...and the assistant response (the only part we compute loss on),
    # including the closing end-of-turn token so the model learns to STOP.
    response_ids = tokenizer.encode(
        assistant + tokenizer.eos_token, add_special_tokens=False,
    )

    input_ids = prompt_ids + response_ids

    # Loss mask: -100 over the prompt, real token ids over the response.
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels":    torch.tensor(labels),   # same length as input_ids
    }

# The training step is then completely ordinary:
#   out = model(input_ids=batch["input_ids"], labels=batch["labels"])
#   out.loss.backward()        # loss is the masked cross-entropy above
# Hugging Face shifts labels internally and ignores the -100 positions.`}
      </CodeBlock>
      <p>
        That’s the entire data-side trick. Everything else — the optimizer, mixed
        precision, the training loop from{" "}
        <a href="/chapter/training-loop" className="prose-link">
          Chapter 15
        </a>{" "}
        — is reused unchanged. SFT is pretraining with better data and a smarter
        label mask.
      </p>

      <h2>The training recipe</h2>
      <p>
        SFT is deliberately gentle. You are <em>nudging</em> a fully-formed model,
        not building one from scratch, so the hyperparameters are very different
        from pretraining. A typical, sane starting recipe:
      </p>
      <ul>
        <li>
          <strong>Low learning rate</strong>, around{" "}
          <M>{"1\\times10^{-5}"}</M> to <M>{"2\\times10^{-5}"}</M> for full
          fine-tuning (often a bit higher for LoRA). Pretraining might use{" "}
          <M>{"3\\times10^{-4}"}</M>; going that hot here would scramble the
          model’s knowledge.
        </li>
        <li>
          <strong>1–3 epochs.</strong> Just a few passes over a small, clean
          dataset. More than that and you start to <em>overfit</em> the
          demonstrations and erode general ability.
        </li>
        <li>
          <strong>A short warmup + cosine or linear decay</strong> schedule, the
          same family of schedules as pretraining (see{" "}
          <a href="/chapter/optimization" className="prose-link">
            Chapter 13
          </a>
          ), just over far fewer steps.
        </li>
        <li>
          <strong>AdamW</strong> with modest weight decay, packed or padded
          sequences, and gradient clipping — nothing exotic.
        </li>
      </ul>
      <p>
        The headline finding, again, is that <strong>data quality dominates</strong>.
        Doubling your compute buys you little; doubling the diversity and cleanliness
        of a few thousand demonstrations buys you a lot. SFT is the one stage where
        a careful human curating a few thousand examples can outperform a careless
        team with a million.
      </p>

      <Callout type="tip" title="Watch the right signal">
        <p>
          The SFT training-loss curve falling does <em>not</em> mean your model is
          getting better at being an assistant — it mostly means it’s memorizing
          your demonstrations. Evaluate on <em>held-out instructions</em> and with
          qualitative spot-checks (and later, the preference-based and benchmark
          methods of{" "}
          <a href="/chapter/evaluation" className="prose-link">
            Chapter 21
          </a>
          ). If responses get more fluent on your dataset but more rigid or repetitive
          in the wild, you’ve trained too long.
        </p>
      </Callout>

      <h2>Catastrophic forgetting (and how to fight it)</h2>
      <p>
        Fine-tuning hard on a narrow distribution can make a model{" "}
        <strong>forget</strong> capabilities it had after pretraining — a
        phenomenon called <strong>catastrophic forgetting</strong>. Push too many
        epochs of, say, only-Python coding demonstrations and the model may get
        worse at general reasoning or at languages other than English. The weights
        that encoded those skills get overwritten by gradients pointed entirely at
        your narrow task.
      </p>
      <p>The standard mitigations are practical and stackable:</p>
      <ul>
        <li>
          <strong>Low LR + few epochs.</strong> The single biggest lever — small,
          brief updates simply don’t move the weights far enough to forget much.
        </li>
        <li>
          <strong>Data mixing / replay.</strong> Blend in diverse instructions
          (and sometimes a little pretraining-style or general-chat data) so no
          single skill dominates the gradient.
        </li>
        <li>
          <strong>Parameter-efficient fine-tuning.</strong> Freeze the base
          weights and train only a small set of new parameters — the LoRA approach
          of{" "}
          <a href="/chapter/peft-lora" className="prose-link">
            Chapter 20
          </a>
          . Because the original weights are untouched, forgetting is structurally
          limited.
        </li>
        <li>
          <strong>Regularization toward the base.</strong> A KL or L2 penalty that
          discourages drifting far from the original model — the same{" "}
          <em>stay-close-to-the-reference</em> instinct that, as we’ll see next
          chapter, is the heart of RLHF.
        </li>
      </ul>

      <Callout type="warning" title="SFT teaches format, not preferences">
        <p>
          SFT can only imitate the demonstrations it’s shown. It teaches the model{" "}
          <em>one good way</em> to answer, but it has no notion of “response A is{" "}
          <em>better</em> than response B.” It can’t easily learn to be{" "}
          <em>more</em> helpful, <em>less</em> verbose, or <em>safer</em> than its
          demonstrations, because every demonstration is treated as equally,
          maximally correct. Learning from comparisons is precisely the gap the
          next stage fills.
        </p>
      </Callout>

      <h2>Where SFT leaves us — and what’s next</h2>
      <p>
        After SFT you have a genuine instruction-following model. It reads a chat
        prompt, answers the question, and stops. For many applications that is
        enough. But its ceiling is the quality of its demonstrations, and it has no
        mechanism to learn from <em>preferences</em> — the “this answer is better
        than that one” signal that humans give so naturally and that is so much
        cheaper to collect than writing perfect answers from scratch.
      </p>
      <p>
        In{" "}
        <a href="/chapter/rlhf" className="prose-link">
          Chapter 19
        </a>{" "}
        we take this SFT model and align it to human preferences: we’ll train a{" "}
        <strong>reward model</strong> from pairwise comparisons, optimize the
        policy against it with <strong>PPO</strong> under a KL leash that keeps it
        anchored to this SFT checkpoint, and then see how{" "}
        <strong>Direct Preference Optimization (DPO)</strong> collapses that whole
        pipeline into a single, elegant classification loss. The SFT model you just
        built is the indispensable starting point for all of it.
      </p>
    </>
  );
}
