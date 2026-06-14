import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import LRSchedule from "../components/viz/optimization/LRSchedule";

export default function Chapter() {
  return (
    <>
      <p>
        In <a href="/chapter/training-objective" className="prose-link">Chapter 12</a> we pinned down{" "}
        <em>what</em> to minimize — the cross-entropy loss — and saw that its gradient with respect to
        the logits is the elegant <M>{"\\text{softmax}(z) - y"}</M>. Backpropagation hands us the
        gradient of that loss with respect to <em>every</em> parameter. This chapter is about the
        other half of learning: <em>how to take steps</em> with those gradients. We trace the family
        tree from plain SGD up to <strong>AdamW</strong> — the optimizer that trains essentially every
        modern LLM — and then cover the three knobs that make or break a run in practice: the{" "}
        <strong>learning rate</strong>, the <strong>warmup + cosine schedule</strong>, and{" "}
        <strong>gradient clipping</strong>.
      </p>

      <h2>Gradient descent, recapped</h2>
      <p>
        The loss <M>{"\\mathcal{L}(\\theta)"}</M> is a surface over the high-dimensional space of
        parameters; we want its lowest point. The gradient <M>{"g = \\nabla_\\theta \\mathcal{L}"}</M>{" "}
        points in the direction of steepest <em>increase</em>, so we step the opposite way, scaled by a{" "}
        <strong>learning rate</strong> <M>{"\\eta"}</M>:
      </p>
      <MB>{String.raw`\theta_{t+1} \;=\; \theta_t \;-\; \eta\, g_t, \qquad g_t = \nabla_\theta \mathcal{L}(\theta_t).`}</MB>
      <p>
        That is the entire idea. Everything below is a smarter answer to one question: given a noisy
        stream of gradients, what is the best <em>direction and size</em> for the next step?
      </p>

      <h2>SGD and its problems</h2>
      <p>
        We never compute the gradient over the whole dataset — that would be one step per epoch.
        Instead we estimate it on a random <strong>minibatch</strong>, giving{" "}
        <strong>stochastic gradient descent</strong> (SGD). The minibatch gradient is an unbiased but{" "}
        <em>noisy</em> estimate of the true gradient, and that noise, plus the geometry of the loss
        surface, creates two failure modes:
      </p>
      <ul>
        <li>
          <strong>Noise.</strong> Each step points in a slightly wrong direction, so the path
          jitters. A learning rate large enough to make progress can also amplify the noise into
          instability.
        </li>
        <li>
          <strong>Ravines (ill-conditioning).</strong> Loss surfaces are rarely round bowls; they are
          long, narrow valleys — steep across, shallow along. A single global step size is then a
          terrible compromise: small enough to not explode up the steep walls means painfully slow
          along the gentle floor. SGD <em>zig-zags</em> across the valley while barely advancing
          toward the minimum.
        </li>
      </ul>
      <p>
        Switch to the "Optimizers on a bowl" tab in the figure further down to watch SGD bounce
        between the walls of exactly such a ravine. Every optimizer that follows is a fix for one or
        both of these problems.
      </p>

      <h2>Momentum: an EMA of gradients</h2>
      <p>
        The fix for zig-zag is memory. <strong>Momentum</strong> accumulates an{" "}
        <strong>exponential moving average</strong> (EMA) of past gradients and steps along that
        running average instead of the raw, jittery gradient:
      </p>
      <MB>{String.raw`v_t = \beta\, v_{t-1} + (1-\beta)\, g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t.`}</MB>
      <p>
        with <M>{"\\beta \\approx 0.9"}</M>. The intuition is a heavy ball rolling downhill: in the
        shallow direction, consecutive gradients <em>agree</em> and reinforce, so velocity builds and
        the ball accelerates along the floor; in the steep direction, gradients flip sign on
        alternating steps and <em>cancel</em> in the average, so the oscillation is damped. Same
        gradients, far less zig-zag, much faster descent.
      </p>

      <h2>Adaptive rates: RMSProp</h2>
      <p>
        Momentum smooths the <em>direction</em>; it still uses one global step size. The
        complementary idea is to give <strong>each parameter its own effective learning rate</strong>,
        scaled down for parameters whose gradients have been large and up for those whose gradients
        have been small. <strong>RMSProp</strong> tracks an EMA of <em>squared</em> gradients and
        divides by its square root:
      </p>
      <MB>{String.raw`s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t.`}</MB>
      <p>
        Here <M>{"g_t^2"}</M> is element-wise, so <M>{"s_t"}</M> is a per-parameter estimate of recent
        gradient magnitude. Dividing by <M>{"\\sqrt{s_t}"}</M> <em>normalizes</em> every coordinate to
        roughly unit scale: steep directions get throttled, flat directions get amplified, and the
        ravine effectively becomes a round bowl. The tiny <M>{"\\epsilon"}</M> (e.g.{" "}
        <M>{"10^{-8}"}</M>) just prevents division by zero.
      </p>

      <h2>Adam = momentum + RMSProp</h2>
      <p>
        <strong>Adam</strong> (Adaptive Moment Estimation) is the obvious, brilliant combination:
        smooth the <em>direction</em> with momentum <em>and</em> normalize the <em>per-parameter
        scale</em> with RMSProp, simultaneously. It keeps two EMAs — the first moment{" "}
        <M>{"m"}</M> (mean of gradients) and the second moment <M>{"v"}</M> (mean of squared
        gradients):
      </p>
      <MB>{String.raw`\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2
\end{aligned}`}</MB>
      <p>
        Both <M>{"m"}</M> and <M>{"v"}</M> start at zero, which biases them toward zero in the early
        steps. Adam corrects this exactly with a <strong>bias correction</strong> that divides out the
        EMA's startup transient:
      </p>
      <MB>{String.raw`\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}.`}</MB>
      <p>
        The update then steps along the bias-corrected mean, scaled per-parameter by the
        bias-corrected root-mean-square:
      </p>
      <MB>{String.raw`\theta_{t+1} \;=\; \theta_t \;-\; \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.`}</MB>
      <p>
        The default hyperparameters — <M>{"\\beta_1 = 0.9"}</M>, <M>{"\\beta_2 = 0.999"}</M>,{" "}
        <M>{"\\epsilon = 10^{-8}"}</M> — work across an astonishing range of problems, which is a big
        part of why Adam is the default. (LLM practitioners often nudge <M>{"\\beta_2"}</M> down to{" "}
        <M>{"0.95"}</M> for better stability with the large, noisy gradients of giant batches.)
      </p>

      <Callout type="key" title="Why Adam(W) dominates LLM training">
        <p>
          A transformer has parameters with wildly different gradient scales — embeddings,
          attention projections, LayerNorm gains, the unembedding. A single global learning rate
          serves none of them well. Adam's <strong>per-parameter normalization</strong> makes it
          robust to this heterogeneity and to noisy minibatch gradients, so it trains stably with far
          less tuning than SGD. At the scale of billions of parameters and trillions of tokens, that
          robustness is worth the extra memory of storing <M>{"m"}</M> and <M>{"v"}</M>.
        </p>
      </Callout>

      <h2>AdamW: decoupled weight decay</h2>
      <p>
        <strong>Weight decay</strong> is regularization that gently shrinks parameters toward zero
        each step, discouraging the model from relying on large weights. The historically standard way
        to get it was to add an <M>{"L_2"}</M> penalty <M>{"\\tfrac{\\lambda}{2}\\lVert\\theta\\rVert^2"}</M>{" "}
        to the loss, which adds <M>{"\\lambda \\theta"}</M> to the gradient. With plain SGD those are
        identical. With Adam they are <em>not</em> — and the difference matters.
      </p>
      <p>
        The trouble: if the decay term is folded into <M>{"g_t"}</M>, it flows through Adam's
        per-parameter <M>{"1/\\sqrt{\\hat v_t}"}</M> rescaling along with everything else. Parameters
        with large gradient history get their decay <em>shrunk</em>; parameters with small history get
        it <em>amplified</em>. The amount of regularization a weight receives ends up coupled to its
        gradient magnitude — an unintended, scale-dependent mess.{" "}
        <strong>AdamW</strong> fixes this by <em>decoupling</em> weight decay from the gradient: it
        applies the Adam step, then shrinks the weights as a separate, direct operation:
      </p>
      <MB>{String.raw`\theta_{t+1} \;=\; \theta_t \;-\; \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} \;+\; \lambda\, \theta_t\right).`}</MB>
      <p>
        Now every parameter decays by the same proportional amount{" "}
        <M>{"\\eta\\lambda"}</M> per step, independent of its gradient statistics — exactly the
        regularization we intended. This decoupling consistently improves generalization, and{" "}
        <strong>AdamW is the standard optimizer for LLM pretraining</strong> (a typical{" "}
        <M>{"\\lambda \\approx 0.1"}</M>). One subtlety practitioners observe: weight decay is usually
        applied only to the matrix weights, <em>not</em> to biases or LayerNorm/RMSNorm gains, which
        are excluded from the decay group.
      </p>

      <Callout type="industry" title="The standard recipe">
        <p>
          A frontier pretraining run almost always uses <strong>AdamW</strong> with{" "}
          <M>{"\\beta_1 = 0.9"}</M>, <M>{"\\beta_2 = 0.95"}</M>, <M>{"\\epsilon = 10^{-8}"}</M>,
          weight decay <M>{"\\approx 0.1"}</M> (on weights only), a peak learning rate found by
          sweeping, linear <strong>warmup</strong> over the first ~0.5–2% of steps, <strong>cosine
          decay</strong> to ~10% of the peak, and <strong>gradient clipping</strong> at global norm
          1.0. If you remember one configuration from this book, remember that one.
        </p>
      </Callout>

      <h2>The learning rate: the one hyperparameter that matters most</h2>
      <p>
        If you can tune exactly one thing, tune the learning rate <M>{"\\eta"}</M>. It sets the size of
        every step, and its window between "too small" and "too large" is narrow:
      </p>
      <ul>
        <li>
          <strong>Too high</strong> → the steps overshoot the minimum and the loss <em>diverges</em>{" "}
          (climbs, spikes to <code>NaN</code>, or oscillates without settling). The single most common
          cause of a blown-up training run.
        </li>
        <li>
          <strong>Too low</strong> → the loss falls, but glacially; you waste compute crawling, and may
          stall in a poor region before the budget runs out.
        </li>
        <li>
          <strong>Just right</strong> → fast, steady descent. Because the ideal value spans orders of
          magnitude across problems, it is found by sweeping on a <em>log</em> scale (e.g.{" "}
          <M>{"1\\!\\times\\!10^{-4},\\ 3\\!\\times\\!10^{-4},\\ 1\\!\\times\\!10^{-3}"}</M>), not by
          nudging linearly.
        </li>
      </ul>

      <Callout type="pitfall" title="If your loss explodes, the learning rate is the first suspect">
        <p>
          A loss that shoots up or turns to <code>NaN</code> in the first few hundred steps almost
          always means the learning rate is too high <em>or</em> there is no warmup. Before touching
          the architecture, the data, or the numerics, <strong>halve the peak LR and add warmup</strong>.
          That one change resolves the large majority of "my transformer won't train" reports. Adam's
          adaptivity helps but does <em>not</em> make it immune — a too-large <M>{"\\eta"}</M> still
          diverges.
        </p>
      </Callout>

      <h2>Warmup + cosine schedule</h2>
      <p>
        We do not hold <M>{"\\eta"}</M> constant. The near-universal schedule for LLMs has two phases.
      </p>
      <p>
        <strong>Linear warmup.</strong> At initialization the weights are essentially random, so the
        first gradients are large and unreliable, and Adam's second-moment estimate <M>{"\\hat v"}</M>{" "}
        has barely any history to normalize with. A full-size step now can knock the model into a bad
        region it never recovers from. So we ramp <M>{"\\eta"}</M> <em>linearly from 0</em> up to the
        peak over the first <M>{"W"}</M> steps, letting the optimizer's moment estimates stabilize
        before we trust them with big steps.
      </p>
      <p>
        <strong>Cosine decay.</strong> After warmup, we anneal the learning rate down following a half
        cosine — fast at first, then gently easing into a small final value (often 10% of the peak, or
        zero). Large steps early make rapid progress; small steps late let the model settle precisely
        into a minimum without bouncing out of it. Writing <M>{"\\eta_{\\max}"}</M> for the peak,{" "}
        <M>{"\\eta_{\\min}"}</M> for the floor, <M>{"W"}</M> for warmup steps and <M>{"T"}</M> for total
        steps, the full schedule is:
      </p>
      <MB>{String.raw`\eta_t \;=\; \begin{cases} \dfrac{t}{W}\,\eta_{\max}, & t < W \\[1.4em] \eta_{\min} + \tfrac{1}{2}\bigl(\eta_{\max}-\eta_{\min}\bigr)\!\left(1 + \cos\!\Big(\pi\,\dfrac{t - W}{T - W}\Big)\right), & t \ge W. \end{cases}`}</MB>
      <p>
        At <M>{"t = W"}</M> the cosine argument is 0, giving <M>{"\\eta = \\eta_{\\max}"}</M>; at{" "}
        <M>{"t = T"}</M> the argument is <M>{"\\pi"}</M> and <M>{"\\cos\\pi = -1"}</M>, giving{" "}
        <M>{"\\eta = \\eta_{\\min}"}</M> — a smooth, continuous ride from peak to floor. The figure
        below is this exact formula; drag the sliders and watch the curve reshape.
      </p>

      <Figure
        n="13.1"
        title="Warmup + cosine learning-rate schedule (and optimizers on a bowl)"
        caption="Tab 1: linear warmup to max_lr, then a half-cosine decay to min_lr — the schedule behind GPT-3, LLaMA and friends. Tab 2: SGD, Momentum and Adam descending the same anisotropic 'ravine' loss surface, showing why adaptive, normalized steps win. Both are deterministic."
      >
        <LRSchedule />
      </Figure>

      <h2>Gradient clipping</h2>
      <p>
        Even with a good schedule, a single unlucky minibatch — a weird document, a numerical
        hiccup — can produce a gigantic gradient that, taken at face value, would launch the
        parameters into oblivion and spike the loss. <strong>Gradient clipping by global norm</strong>{" "}
        is the seatbelt. Compute the norm of the <em>entire</em> gradient vector (all parameters
        concatenated); if it exceeds a threshold <M>{"\\tau"}</M> (commonly 1.0), rescale the whole
        gradient down to that norm, preserving its <em>direction</em> but capping its{" "}
        <em>magnitude</em>:
      </p>
      <MB>{String.raw`\hat{g} \;=\; g \cdot \min\!\left(1,\; \frac{\tau}{\lVert g \rVert_2}\right).`}</MB>
      <p>
        Below the threshold the gradient passes through untouched; above it, the step is reined in.
        This costs almost nothing and prevents rare gradient spikes from undoing thousands of good
        steps — it is standard in virtually every large training run. Watching the <em>pre-clip</em>{" "}
        gradient norm is also one of the best health signals you have: a sudden, sustained jump in it
        is an early warning that something is going wrong.
      </p>

      <h2>It all in code</h2>
      <p>
        Two listings. First, the AdamW update written out in plain NumPy for one parameter array, so
        every term — the two moments, the bias correction, the decoupled decay — is visible with no
        framework hiding it. Second, the real-world PyTorch setup: construct <code>AdamW</code> and
        attach a warmup-then-cosine schedule via <code>LambdaLR</code>.
      </p>

      <CodeBlock language="python" filename="adamw_numpy.py">
{`import numpy as np

# One AdamW step for a single parameter array 'w' given its gradient 'g'.
# State (m, v) persists across steps and starts at zero.
def adamw_step(w, g, m, v, t, lr=3e-4, beta1=0.9, beta2=0.95,
               eps=1e-8, weight_decay=0.1):
    # 1) update biased first & second moment estimates (EMAs)
    m = beta1 * m + (1 - beta1) * g          # mean of gradients
    v = beta2 * v + (1 - beta2) * (g * g)    # mean of squared gradients

    # 2) bias-correct: counteract the zero-initialization of m, v
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # 3) the Adam step: direction m_hat, per-parameter scale 1/sqrt(v_hat)
    update = m_hat / (np.sqrt(v_hat) + eps)

    # 4) DECOUPLED weight decay — applied to w directly, NOT mixed into g.
    #    This is the single line that distinguishes AdamW from Adam + L2.
    w = w - lr * (update + weight_decay * w)
    return w, m, v

# quick sanity loop on a tiny quadratic  f(w) = 0.5 * ||w - target||^2
target = np.array([1.0, -2.0, 0.5])
w = np.zeros_like(target)
m = np.zeros_like(target)
v = np.zeros_like(target)
for t in range(1, 401):                      # t starts at 1 (bias correction needs it)
    g = (w - target)                         # gradient of f
    w, m, v = adamw_step(w, g, m, v, t)
print(w)                                     # -> close to [1.0, -2.0, 0.5]`}
      </CodeBlock>

      <CodeBlock language="python" filename="train_setup.py" highlight={[12, 22]}>
{`import math
import torch

# --- 1) the optimizer: AdamW, with weight decay only on matrix weights ------
def make_optimizer(model, lr=3e-4, weight_decay=0.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # biases and 1-D params (LayerNorm/RMSNorm gains) are NOT decayed
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

# --- 2) the schedule: linear warmup -> cosine decay, via LambdaLR ------------
def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_ratio=0.1):
    def lr_lambda(step):                       # returns a MULTIPLIER on the base lr
        if step < warmup_steps:                # linear warmup: 0 -> 1
            return step / max(1, warmup_steps)
        # cosine decay: 1 -> min_ratio over the remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

optimizer = make_optimizer(model)             # 'model' is your transformer
scheduler = cosine_with_warmup(optimizer, warmup_steps=2000, total_steps=100_000)

# --- 3) the inner loop, with gradient clipping ------------------------------
for step, (x, y) in enumerate(loader):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                             y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # global-norm clip
    optimizer.step()
    scheduler.step()                           # advance the LR schedule every step`}
      </CodeBlock>

      <Callout type="warning" title="Schedule length must match your real budget">
        <p>
          A cosine schedule decays to its floor exactly at <code>total_steps</code>. Set that number
          too small and the LR bottoms out while you still have data left, so the model crawls for the
          rest of the run; set it too large and the LR is still high when you stop, leaving the model
          under-annealed and the final loss worse than it should be. Decide your token budget{" "}
          <em>first</em>, then size the schedule to it.
        </p>
      </Callout>

      <h2>Bridge: from optimizer to the training loop</h2>
      <p>
        We can now turn gradients into a trained model. SGD steps downhill; momentum smooths the
        direction with an EMA of gradients; RMSProp normalizes the per-parameter scale; Adam fuses the
        two with bias correction; and AdamW adds <em>decoupled</em> weight decay — the configuration
        behind every modern LLM. Around it sit the three decisions that actually determine whether a
        run succeeds: a well-swept <strong>learning rate</strong>, a <strong>warmup + cosine
        schedule</strong>, and <strong>gradient clipping</strong> at global norm 1.0.
      </p>
      <p>
        With the objective (Chapter 12) and the optimizer (this chapter) in hand, we have every
        ingredient of a training step. Next we assemble them — data loading, the forward and backward
        pass, mixed precision, checkpointing, and logging — into the complete{" "}
        <strong>training loop</strong> that runs for days on a cluster, in{" "}
        <a href="/chapter/training-loop" className="prose-link">Chapter 15</a>.
      </p>
    </>
  );
}
