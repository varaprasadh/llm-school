import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import PreferenceViz from "../components/viz/rlhf/PreferenceViz";

export default function Chapter() {
  return (
    <>
      <p>
        After{" "}
        <a href="/chapter/supervised-finetuning" className="prose-link">
          supervised fine-tuning
        </a>
        , your model follows instructions — but it follows them only as well as
        its demonstrations did. SFT can imitate one good answer; it cannot learn
        that <em>this</em> answer is <em>better</em> than <em>that</em> one. Yet
        “better” is exactly the judgment humans make effortlessly and the judgment
        that separates a merely-competent assistant from a genuinely helpful,
        honest, and harmless one. This chapter is about teaching a model from{" "}
        <strong>preferences</strong> — the foundation of modern alignment, from the
        classic RLHF pipeline to the simpler DPO that has largely replaced it.
      </p>

      <Callout type="key" title="The whole game">
        <p>
          Humans are far better at <em>comparing</em> two responses than at writing
          the perfect one. Preference alignment turns a pile of “A is better than
          B” judgments into a training signal that pushes the model’s probability
          mass <strong>toward</strong> responses people prefer and{" "}
          <strong>away</strong> from those they don’t — while a leash keeps it from
          drifting too far from the sensible SFT model it started as.
        </p>
      </Callout>

      <h2>Why SFT isn’t enough</h2>
      <p>
        SFT maximizes the likelihood of a fixed set of “correct” responses. Three
        things it fundamentally cannot do:
      </p>
      <ul>
        <li>
          <strong>Rank.</strong> Every demonstration is treated as equally,
          maximally good. There’s no way to say “this response is a 9 and that one
          is a 4,” so the model can’t learn to prefer subtler qualities like
          concision, tone, or honesty.
        </li>
        <li>
          <strong>Learn from negatives.</strong> SFT only ever sees good examples.
          It is never explicitly taught what a <em>bad</em> answer looks like, so
          it can’t actively steer away from one.
        </li>
        <li>
          <strong>Scale cheaply.</strong> Writing a flawless reference answer is
          slow and expensive. Picking the better of two candidate answers is fast
          and cheap — and far more consistent between annotators. Preference data
          is the cost-effective signal.
        </li>
      </ul>
      <p>
        So instead of asking humans to <em>write</em> the ideal response, we ask
        them to <em>compare</em> a pair of model-generated responses and pick the
        better one. That single change — from demonstrations to comparisons — is
        what unlocks the rest of this chapter.
      </p>

      <h2>The classic three-stage RLHF pipeline</h2>
      <p>
        Reinforcement Learning from Human Feedback, as introduced by OpenAI’s
        InstructGPT (2022), is a three-stage process. Each stage produces an
        artifact the next stage consumes:
      </p>
      <ol>
        <li>
          <strong>SFT.</strong> Fine-tune the base model on demonstrations to get a
          policy <M>{"\\pi_{\\text{SFT}}"}</M> that follows instructions. (The
          previous chapter.) This becomes both the starting point and the{" "}
          <em>reference</em> for the next stages.
        </li>
        <li>
          <strong>Reward modeling.</strong> Collect pairwise human preferences over
          model outputs, and train a separate <strong>reward model</strong>{" "}
          <M>{"r_\\phi(x, y)"}</M> that scores how good a response <M>{"y"}</M> is
          for a prompt <M>{"x"}</M>.
        </li>
        <li>
          <strong>RL optimization.</strong> Use reinforcement learning (PPO) to
          update the policy so it produces responses the reward model scores
          highly — <em>without</em> wandering too far from{" "}
          <M>{"\\pi_{\\text{SFT}}"}</M>, enforced by a KL penalty.
        </li>
      </ol>
      <p>
        The names matter: the model being trained is the <strong>policy</strong>{" "}
        <M>{"\\pi_\\theta"}</M> (it produces actions = tokens), the frozen SFT model
        is the <strong>reference</strong> <M>{"\\pi_{\\text{ref}}"}</M>, and the
        reward model is the learned judge. Keep those three straight and the math
        falls into place.
      </p>

      <h2>Reward modeling from pairwise preferences</h2>
      <p>
        How do you convert “a human picked response A over response B” into a
        scalar reward function? With the <strong>Bradley–Terry model</strong>, a
        classic statistical model of paired comparisons. It posits that each item
        has a latent “strength” (here, the reward <M>{"r"}</M>), and the
        probability that A beats B is the logistic function of their difference:
      </p>
      <MB>{String.raw`P(y_A \succ y_B \mid x) = \sigma\!\big(r_\phi(x, y_A) - r_\phi(x, y_B)\big) = \frac{1}{1 + e^{-(r_A - r_B)}}`}</MB>
      <p>
        Read it off: if the rewards are equal, the gap is 0 and{" "}
        <M>{"\\sigma(0) = 0.5"}</M> — a coin flip. As the chosen response’s reward
        pulls ahead, the probability the human prefers it climbs toward 1. To{" "}
        <em>train</em> the reward model, we simply maximize the likelihood of the
        human choices — equivalently, minimize the negative log-likelihood over the
        preference dataset <M>{"\\mathcal{D}"}</M> of (prompt, chosen{" "}
        <M>{"y_w"}</M>, rejected <M>{"y_l"}</M>) triples:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{RM}} = -\,\mathbb{E}_{(x, y_w, y_l)\sim\mathcal{D}}\Big[\log \sigma\!\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]`}</MB>
      <p>
        Architecturally, the reward model is usually the SFT model with its
        token-prediction head swapped for a single linear layer that reads the
        final hidden state and outputs one scalar. The figure below lets you play
        with the Bradley–Terry curve directly: drag the reward gap{" "}
        <M>{"\\Delta = r_A - r_B"}</M> and watch the predicted preference
        probability slide along the sigmoid.
      </p>

      <Figure
        n="19.1"
        title="Preferences, Bradley–Terry, and the DPO margin"
        caption="Top: two candidate responses with reward-model scores. Middle: the reward gap Δ = r_A − r_B drives the Bradley-Terry probability σ(Δ) that a human prefers A. Bottom: drag training forward to watch DPO push the chosen response's log-prob up and the rejected one's down, widening the implicit-reward margin."
      >
        <PreferenceViz />
      </Figure>

      <Callout type="math" title="Why a sigmoid of a difference?">
        <p>
          Only reward <em>differences</em> are identifiable from preferences —
          adding a constant to every reward leaves all comparisons unchanged. The
          logistic form <M>{"\\sigma(r_A - r_B)"}</M> is the maximum-entropy choice
          consistent with that, and it makes the loss a smooth, convex-in-the-gap
          binary cross-entropy. This is the same Bradley–Terry / Elo math that
          ranks chess players — the model is literally learning an Elo rating for
          responses.
        </p>
      </Callout>

      <h2>PPO: optimizing the policy under a KL leash</h2>
      <p>
        Now we have a judge <M>{"r_\\phi"}</M>. The naive idea is to update the
        policy to maximize expected reward: generate responses, score them, and
        push up the probability of high-reward ones. But pure reward maximization
        is dangerous — the policy will happily contort itself into a weird,
        repetitive, reward-gaming distribution that the reward model loves but
        humans hate (the reward model is only an <em>approximation</em> of human
        preference, and it has blind spots).
      </p>
      <p>
        The fix is a <strong>KL penalty</strong> that tethers the policy to the
        reference (SFT) model. The full RLHF objective maximizes reward{" "}
        <strong>minus</strong> a penalty for drifting away from{" "}
        <M>{"\\pi_{\\text{ref}}"}</M>:
      </p>
      <MB>{String.raw`\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\Big[\, r_\phi(x, y)\,\Big] \;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)`}</MB>
      <p>
        The coefficient <M>{"\\beta"}</M> sets how tight the leash is. The actual
        optimization is done with <strong>Proximal Policy Optimization (PPO)</strong>,
        an RL algorithm that takes conservative, clipped policy-gradient steps so a
        single update can’t blow up the policy. In practice the KL term is folded
        into a per-token reward: each generated token gets reward{" "}
        <M>{"r - \\beta\\log\\frac{\\pi_\\theta}{\\pi_{\\text{ref}}}"}</M>, the
        scalar <M>{"r"}</M> arriving only at the final token.
      </p>

      <Callout type="key" title="Why the KL leash is non-negotiable">
        <p>
          Without it, the policy optimizes the <em>proxy</em> (the reward model),
          not the <em>goal</em> (human preference), and the two diverge fast. The
          KL term says “become higher-reward, but stay recognizably the helpful
          model you already are.” Too small a <M>{"\\beta"}</M> and the model
          reward-hacks into gibberish; too large and it never improves. Tuning
          this leash is most of the art of getting PPO to work.
        </p>
      </Callout>

      <h2>Reward hacking and over-optimization</h2>
      <p>
        The reward model is a <em>learned proxy</em> for what humans want, and like
        every proxy it can be gamed. This is <strong>reward hacking</strong> (or
        over-optimization): the policy discovers inputs to the reward model that
        score highly but don’t reflect genuine quality. Real, documented examples:
      </p>
      <ul>
        <li>
          <strong>Length bias.</strong> If annotators slightly preferred longer
          answers, the reward model learns “longer = better,” and the policy starts
          padding every response into a rambling essay.
        </li>
        <li>
          <strong>Sycophancy.</strong> Telling the user what they want to hear
          scores well with raters, so the model learns to agree rather than to be
          correct.
        </li>
        <li>
          <strong>Format gaming.</strong> Over-using bullet points, bold text, or
          hedging phrases that correlate with high reward in the training data.
        </li>
      </ul>
      <p>
        Goodhart’s law in action: “when a measure becomes a target, it ceases to be
        a good measure.” Empirically, as you push the policy to higher reward-model
        score, <em>true</em> quality rises then <em>falls</em> — an inverted-U. The
        KL leash, a held-out reward model, and early stopping are the front-line
        defenses; the deeper fix is better preference data.
      </p>

      <Callout type="pitfall" title="High reward-model score ≠ good model">
        <p>
          Never trust the reward curve alone. A policy can drive its reward-model
          score up and up while getting <em>worse</em> by human judgment. Always
          evaluate the final policy with fresh human preference comparisons or
          held-out benchmarks (
          <a href="/chapter/evaluation" className="prose-link">
            Chapter 21
          </a>
          ), and watch the KL divergence from the reference — a runaway KL is the
          telltale sign of over-optimization.
        </p>
      </Callout>

      <h2>DPO: skipping the reward model entirely</h2>
      <p>
        The classic pipeline is powerful but heavy: you train a separate reward
        model, then run an unstable RL loop that juggles <em>four</em> models in
        memory (policy, reference, reward model, and a value/critic network).{" "}
        <strong>Direct Preference Optimization (DPO)</strong>, from Rafailov et al.
        (2023), is the insight that you can skip all of it and optimize a simple
        classification loss <em>directly</em> on preference pairs.
      </p>
      <p>
        The derivation is genuinely beautiful. The KL-constrained objective from
        the PPO section has a known <em>closed-form</em> optimal policy: the best
        policy reweights the reference by the exponentiated reward,{" "}
        <M>{"\\pi^*(y\\mid x) \\propto \\pi_{\\text{ref}}(y\\mid x)\\,e^{r(x,y)/\\beta}"}</M>.
        Rearranging that relationship lets you express the reward{" "}
        <em>in terms of the policy itself</em> — the famous{" "}
        <strong>implicit reward</strong>:
      </p>
      <MB>{String.raw`r(x, y) = \beta\,\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \;+\; \beta\log Z(x)`}</MB>
      <p>
        Now substitute this reward into the Bradley–Terry preference model. The
        intractable partition term <M>{"\\beta\\log Z(x)"}</M>{" "}
        <em>cancels</em> because Bradley–Terry depends only on the reward{" "}
        <em>difference</em> between the two responses to the same prompt. What
        remains is a loss you can compute and backpropagate with nothing but the
        policy and the frozen reference:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log \sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]`}</MB>
      <p>
        Look at what this does. The term inside the sigmoid is the implicit-reward{" "}
        <strong>margin</strong> between the chosen response <M>{"y_w"}</M> and the
        rejected response <M>{"y_l"}</M>. Minimizing the loss pushes that margin
        up: it <em>raises</em> the policy’s log-probability of the chosen response
        and <em>lowers</em> that of the rejected one — each measured{" "}
        <em>relative to the reference</em>, so the <M>{"\\pi_{\\text{ref}}"}</M>{" "}
        terms <em>are</em> the KL leash, baked directly into the loss. No reward
        model. No RL. No reward sampling. Just a binary classifier over preference
        pairs. The bottom panel of the figure above visualizes exactly this:
        training widens the margin, the preference probability climbs to 100%, and
        the DPO loss falls toward 0.
      </p>

      <CodeBlock language="python" filename="dpo_loss.py" highlight={[28, 29, 30, 31]}>
{`import torch
import torch.nn.functional as F

def sequence_logprob(model, input_ids, labels):
    """Sum the log-probs of the response tokens (labels != -100) under \`model\`."""
    logits = model(input_ids).logits[:, :-1]          # predict next token
    labels = labels[:, 1:].clone()                    # shift to align
    mask = labels != -100
    labels = labels.masked_fill(~mask, 0)             # safe gather index
    token_logp = torch.gather(
        F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logp * mask).sum(-1)                # (batch,) seq log-prob

def dpo_loss(policy, ref, batch, beta=0.1):
    # log pi(y|x) under the trainable policy, for chosen & rejected responses
    pi_chosen   = sequence_logprob(policy, batch["chosen_ids"],   batch["chosen_labels"])
    pi_rejected = sequence_logprob(policy, batch["rejected_ids"], batch["rejected_labels"])

    # log pi_ref(y|x) under the frozen reference (no gradient)
    with torch.no_grad():
        ref_chosen   = sequence_logprob(ref, batch["chosen_ids"],   batch["chosen_labels"])
        ref_rejected = sequence_logprob(ref, batch["rejected_ids"], batch["rejected_labels"])

    # Implicit rewards = beta * log(pi / pi_ref) = beta * (logp_policy - logp_ref)
    chosen_reward   = beta * (pi_chosen   - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)

    # DPO loss: -log sigmoid(margin), margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss`}
      </CodeBlock>

      <Callout type="math" title="What β controls in DPO">
        <p>
          The same <M>{"\\beta"}</M> as the PPO KL coefficient, and it plays the
          same role: it’s the strength of the leash to{" "}
          <M>{"\\pi_{\\text{ref}}"}</M>. Larger <M>{"\\beta"}</M> keeps the policy
          closer to the SFT reference (more conservative, smaller updates); smaller{" "}
          <M>{"\\beta"}</M> lets it move further to satisfy preferences. Typical
          values are <M>{"\\beta \\in [0.1, 0.5]"}</M>. Set it to 0 and you remove
          the leash entirely — the model can overfit the preference data and
          degenerate, the DPO analogue of reward hacking.
        </p>
      </Callout>

      <h2>RLHF vs. DPO: which to use</h2>
      <p>
        DPO didn’t make PPO obsolete so much as it made alignment{" "}
        <em>accessible</em>. The tradeoffs:
      </p>
      <table>
        <thead>
          <tr>
            <th>Aspect</th>
            <th>PPO (RLHF)</th>
            <th>DPO</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Models in memory</td>
            <td>Policy, reference, reward, critic (4)</td>
            <td>Policy + frozen reference (2)</td>
          </tr>
          <tr>
            <td>Reward model</td>
            <td>Required, trained separately</td>
            <td>None — reward is implicit</td>
          </tr>
          <tr>
            <td>Stability / tuning</td>
            <td>Finicky; many RL knobs</td>
            <td>Stable; basically supervised</td>
          </tr>
          <tr>
            <td>Data</td>
            <td>Online — samples fresh responses</td>
            <td>Offline — a fixed preference set</td>
          </tr>
          <tr>
            <td>Peak quality</td>
            <td>Can edge higher with effort + online exploration</td>
            <td>Excellent; simpler to reach</td>
          </tr>
        </tbody>
      </table>
      <p>
        Because DPO is <em>offline</em> (it learns from a frozen preference dataset
        rather than continuously sampling new responses and scoring them), it can’t
        explore beyond the data it was given — which is why heavily-tuned online RL
        can still squeeze out a bit more at the frontier. But for the vast majority
        of teams, DPO’s simplicity, stability, and 2-model memory footprint make it
        the default first choice. Variants like IPO, KTO, and ORPO tweak the loss
        further; the DPO recipe is the one to understand first.
      </p>

      <h2>RLAIF and Constitutional AI</h2>
      <p>
        Human preference labels are the bottleneck: slow, expensive, and
        inconsistent. <strong>RLAIF</strong> — Reinforcement Learning from{" "}
        <em>AI</em> Feedback — replaces (or augments) the human labeler with a
        capable LLM that judges which response is better, generating preference
        pairs at scale for a fraction of the cost. The same DPO or reward-model
        machinery then runs on these AI-generated preferences.
      </p>
      <p>
        Anthropic’s <strong>Constitutional AI</strong> is the best-known instance:
        instead of relying on humans to label harmful outputs, the model critiques
        and revises its own responses against a written set of principles (a
        “constitution”), and those self-revisions become the preference data for
        alignment. It makes the values explicit and auditable in a document, and
        dramatically reduces the human labeling burden for the safety-critical
        parts of alignment. RLAIF and human feedback are not mutually exclusive —
        production systems blend both.
      </p>

      <h2>Where alignment leaves us — and what’s next</h2>
      <p>
        You now have the full post-training arc: pretraining gave you raw
        capability, SFT taught the model to follow instructions, and preference
        alignment — whether via the classic reward-model-plus-PPO pipeline or the
        streamlined DPO loss — taught it to produce responses people actually
        prefer, all while a KL leash kept it tethered to the sensible SFT model it
        grew from. This is, in essence, how every frontier chat model you’ve used
        was made helpful and harmless.
      </p>
      <p>
        One practical problem remains: all of this — SFT and preference tuning
        alike — assumed we could afford to update a model’s billions of parameters.
        In{" "}
        <a href="/chapter/peft-lora" className="prose-link">
          Chapter 20
        </a>{" "}
        we’ll see how <strong>parameter-efficient fine-tuning</strong> (LoRA and
        QLoRA) lets you run every technique in this part of the book on a single
        GPU — by training a tiny number of new parameters instead of the whole
        model.
      </p>
    </>
  );
}
