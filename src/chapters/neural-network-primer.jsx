import { M, MB } from "../components/Math";
import Callout from "../components/Callout";
import Figure from "../components/Figure";
import CodeBlock from "../components/CodeBlock";
import MLPForward from "../components/viz/neural-network-primer/MLPForward";
import Activations from "../components/viz/neural-network-primer/Activations";

export default function Chapter() {
  return (
    <>
      <p>
        An LLM is a neural network — a very large one, with a clever architecture — but the engine
        underneath is the same one that powers a digit classifier you could train on a laptop. If
        you have never built a neural net, this chapter is your foundation. We start from a{" "}
        <em>single neuron</em>, stack neurons into layers, define what it means for the network to be
        “wrong,” and then derive the two ideas that let it teach itself:{" "}
        <strong>gradient descent</strong> and <strong>backpropagation</strong>. By the end you will
        have trained a tiny network by hand, in NumPy, and seen the same thing in PyTorch. Every term
        is defined on first use; the only prerequisite is that you can read a for-loop.
      </p>

      <Callout type="key" title="The whole game in one sentence">
        <p>
          A neural network is a stack of simple, differentiable functions with adjustable knobs
          (<em>parameters</em>); learning means nudging those knobs in the direction that most
          reduces a measure of error, using gradients computed by the chain rule. Everything else —
          attention, transformers, trillion-parameter models — is this idea, scaled up.
        </p>
      </Callout>

      <h2>From a single neuron</h2>
      <p>
        Start with the simplest predictor imaginable: a <strong>linear model</strong>. Given two
        input numbers <M>{"x_1, x_2"}</M> it outputs a weighted sum plus a constant:
      </p>
      <MB>{String.raw`y = w_1 x_1 + w_2 x_2 + b`}</MB>
      <p>
        The numbers <M>{"w_1, w_2"}</M> are <strong>weights</strong> — how much each input matters —
        and <M>{"b"}</M> is the <strong>bias</strong>, a baseline that shifts the output up or down.
        That’s it; that’s a line (in higher dimensions, a hyperplane). You have probably met this as
        linear or logistic regression. It is genuinely useful, but it can only represent
        relationships that are <em>straight</em>: no matter how you set the weights, the output is a
        flat tilted plane over the inputs.
      </p>
      <p>
        A <strong>neuron</strong> (or <em>unit</em>) is a linear model with one extra step bolted on
        the end — a nonlinear <strong>activation function</strong> <M>{"\\phi"}</M>:
      </p>
      <MB>{String.raw`a = \phi\!\big(\underbrace{w_1 x_1 + w_2 x_2 + b}_{\text{the pre-activation } z}\big)`}</MB>
      <p>
        We call the quantity inside the parentheses the <strong>pre-activation</strong>{" "}
        <M>{"z"}</M> (sometimes the “logit” or “net input”), and <M>{"a = \\phi(z)"}</M> the{" "}
        <strong>activation</strong> — the neuron’s output. The activation is what lets a network bend.
        Without it, you could stack a thousand neurons and still only ever produce a straight line
        (we prove this in the next section). With it, neurons become the building blocks of a
        function approximator that can represent almost anything.
      </p>

      <Callout type="history" title="Where the metaphor comes from">
        <p>
          The name borrows from biology: a real neuron integrates incoming signals and “fires” once
          they cross a threshold. McCulloch and Pitts formalized this in 1943, and Rosenblatt’s 1958{" "}
          <em>perceptron</em> made it learnable. The biological analogy is loose — artificial neurons
          are just <M>{"\\phi(\\,w\\cdot x + b)"}</M> — but the intuition of “sum the evidence, then
          decide” is a good one to keep.
        </p>
      </Callout>

      <h2>Why nonlinearity? Activation functions</h2>
      <p>
        Here is the central reason activations exist. Suppose you stacked two linear layers with no
        activation between them: <M>{"h = x W_1"}</M> then <M>{"y = h W_2"}</M>. Substituting,{" "}
        <M>{"y = x W_1 W_2 = x W'"}</M> for some single matrix <M>{"W' = W_1 W_2"}</M>. The two
        layers <em>collapse</em> into one linear layer. No matter how many you stack, a network of
        pure linear layers is mathematically equivalent to a single linear layer — it can only draw
        straight boundaries. The activation function is the one ingredient that breaks this collapse
        and unlocks <em>depth</em>.
      </p>
      <p>
        Four activations show up constantly. The plot below puts them on the same axes — drag your
        eye from far left to far right and notice the shapes:
      </p>

      <Figure
        n="2.1"
        title="The four activations you’ll meet most"
        caption="ReLU and GELU keep a healthy slope for positive inputs; sigmoid and tanh flatten (saturate) at both ends, where their gradient vanishes. Toggle the saturating pair off to compare ReLU and GELU directly."
      >
        <Activations />
      </Figure>

      <ul>
        <li>
          <strong>ReLU</strong> — <M>{"\\operatorname{ReLU}(z) = \\max(0, z)"}</M>. Zero for negative
          inputs, identity for positive ones. Cheap (a single comparison), and its gradient is a
          clean 1 for active units, so signal flows freely. The default for most deep nets.
        </li>
        <li>
          <strong>GELU</strong> — <em>Gaussian Error Linear Unit</em>, a smooth cousin of ReLU:{" "}
          <M>{"\\operatorname{GELU}(z) = z\\,\\Phi(z)"}</M>, where <M>{"\\Phi"}</M> is the standard
          normal CDF. It curves gently through the origin instead of kinking sharply. This is the
          activation inside GPT-2/3 and BERT, and the one your transformer will use.
        </li>
        <li>
          <strong>sigmoid</strong> — <M>{"\\sigma(z) = 1/(1 + e^{-z})"}</M>. Squashes any real number
          into <M>{"(0, 1)"}</M>, so it reads as a probability. Still used for binary gates and
          outputs, but rarely as a hidden activation in deep nets.
        </li>
        <li>
          <strong>tanh</strong> — <M>{"\\tanh(z)"}</M>. A rescaled sigmoid mapping to{" "}
          <M>{"(-1, 1)"}</M>, zero-centered. Common in older RNNs.
        </li>
      </ul>

      <p>
        Why do ReLU and GELU dominate deep networks while sigmoid and tanh have faded? Look at the
        tails. For large <M>{"|z|"}</M>, sigmoid and tanh go <em>flat</em> — their output barely
        changes, so their <em>derivative</em> is nearly zero. When you chain many such layers (you’ll
        see why in the backprop section), these near-zero slopes multiply together and the gradient
        reaching the early layers shrinks toward nothing. This is the infamous{" "}
        <strong>vanishing-gradient problem</strong>: early layers stop learning. ReLU’s derivative is
        exactly 1 wherever a unit is active, so it doesn’t shrink the signal, and GELU keeps a
        similar, well-behaved slope while staying smooth. That single property — a non-vanishing
        gradient — is most of why deep learning works at all.
      </p>

      <Callout type="pitfall" title="Dead ReLUs">
        <p>
          ReLU’s flat left half has a downside: a unit whose pre-activation is always negative
          outputs 0 for every input and receives <em>zero</em> gradient — it can never recover. We
          call it a <em>dead</em> neuron. In the visualization below you can watch hidden units go
          dark exactly when their <M>{"z < 0"}</M>. Variants like Leaky ReLU and GELU soften this by
          letting a little signal through on the negative side.
        </p>
      </Callout>

      <h2>Layers and the forward pass</h2>
      <p>
        One neuron is weak. The power comes from putting many side by side into a{" "}
        <strong>layer</strong>, and stacking layers into a network. A layer with, say, 3 neurons
        takes the same inputs and produces 3 activations. Instead of writing three separate weighted
        sums, we collect the weights into a matrix and the biases into a vector, and the whole layer
        becomes a single matrix multiply:
      </p>
      <MB>{String.raw`\mathbf{h} = \phi\!\big(\mathbf{x}\,W + \mathbf{b}\big)`}</MB>
      <p>
        Getting the <strong>shapes</strong> right is half of understanding deep learning, so let’s be
        explicit. If the input <M>{"\\mathbf{x}"}</M> is a row vector of length{" "}
        <M>{"d_{\\text{in}}"}</M> and the layer has <M>{"d_{\\text{out}}"}</M> neurons, then:
      </p>
      <ul>
        <li>
          <M>{"\\mathbf{x}"}</M> has shape <M>{"(1, d_{\\text{in}})"}</M>,
        </li>
        <li>
          the weight matrix <M>{"W"}</M> has shape <M>{"(d_{\\text{in}}, d_{\\text{out}})"}</M> — one
          column of weights per output neuron,
        </li>
        <li>
          the bias <M>{"\\mathbf{b}"}</M> has shape <M>{"(1, d_{\\text{out}})"}</M>,
        </li>
        <li>
          so the output <M>{"\\mathbf{h} = \\phi(\\mathbf{x}W + \\mathbf{b})"}</M> has shape{" "}
          <M>{"(1, d_{\\text{out}})"}</M>, ready to feed the next layer.
        </li>
      </ul>
      <p>
        In practice you push a whole <strong>batch</strong> of <M>{"N"}</M> examples through at once:{" "}
        <M>{"X"}</M> becomes <M>{"(N, d_{\\text{in}})"}</M> and <M>{"H = \\phi(XW + \\mathbf{b})"}</M>{" "}
        becomes <M>{"(N, d_{\\text{out}})"}</M>, with the bias added to every row (this is{" "}
        <em>broadcasting</em>). Running an input forward through every layer to produce a prediction
        is called the <strong>forward pass</strong>. A network of two such layers — the canonical{" "}
        <strong>multilayer perceptron</strong> (MLP) — is:
      </p>
      <MB>{String.raw`\hat{\mathbf{y}} = \phi_2\!\big(\,\phi_1(\mathbf{x}\,W_1 + \mathbf{b}_1)\,W_2 + \mathbf{b}_2\,\big)`}</MB>
      <p>
        The intermediate layers are called <strong>hidden layers</strong> (you never observe their
        activations directly), and their width <M>{"d_{\\text{hidden}}"}</M> is a knob you choose. The
        figure below is exactly this kind of network — 2 inputs, a 3-unit hidden layer with ReLU, and
        a single output — drawn so you can watch a forward pass happen. Move the input sliders and
        every weighted sum, activation, and the final prediction recompute live. Edges are{" "}
        <span className="text-cyan-300">cyan when the weight is positive</span> and{" "}
        <span className="text-rose-300">rose when negative</span>, and thicker for larger magnitude;
        each node glows brighter the larger its activation.
      </p>

      <Figure
        n="2.2"
        title="A tiny MLP, computing live"
        caption="2 inputs → 3 hidden (ReLU) → 1 output. The numbers inside the nodes are real activations recomputed as you drag. Notice a hidden unit go dark the instant its pre-activation z turns negative — ReLU has zeroed it out. Flip on the backward pass to see which way gradients travel during training."
      >
        <MLPForward />
      </Figure>

      <Callout type="industry" title="An LLM is mostly these layers">
        <p>
          A transformer block is two sublayers: attention, then a position-wise MLP — literally the{" "}
          <M>{"\\phi(\\mathbf{x}W_1 + \\mathbf{b}_1)\\,W_2 + \\mathbf{b}_2"}</M> you see here, applied
          to every token. In most large models that MLP holds roughly two-thirds of all parameters.
          Master this little two-layer network and you have understood the bulk of where a model’s
          weights actually live.
        </p>
      </Callout>

      <h2>Measuring wrongness: the loss function</h2>
      <p>
        A fresh network has random weights and makes nonsense predictions. To improve it we first
        need a single number that says <em>how wrong</em> it is on the training data — the{" "}
        <strong>loss</strong> (or cost / objective). Lower is better; learning is the act of making
        this number small.
      </p>
      <p>
        For predicting real-valued targets (regression), the workhorse is{" "}
        <strong>mean squared error</strong> (MSE): the average squared gap between prediction{" "}
        <M>{"\\hat{y}"}</M> and truth <M>{"y"}</M> over <M>{"N"}</M> examples:
      </p>
      <MB>{String.raw`\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{n=1}^{N}\big(\hat{y}_n - y_n\big)^2`}</MB>
      <p>
        Squaring does two jobs: it makes every error positive (over- and under-shooting both count),
        and it punishes large mistakes far more than small ones, so the network prioritizes fixing
        its worst predictions. The choice of loss is not arbitrary — it encodes <em>what you want</em>.
        MSE is perfect for intuition because it’s just “distance squared.”
      </p>

      <Callout type="note" title="LLMs don’t use MSE">
        <p>
          A language model predicts a <em>probability distribution</em> over the next token, not a
          single number, so it’s trained with <strong>cross-entropy loss</strong> instead — the
          negative log-probability the model assigns to the correct token. The learning machinery
          (gradients, backprop, the update rule) is identical; only the final loss differs. We derive
          cross-entropy carefully in{" "}
          <a href="/chapter/training-objective" className="prose-link">
            Chapter 12
          </a>
          . For now, MSE keeps the picture clean.
        </p>
      </Callout>

      <h2>Learning by gradient descent</h2>
      <p>
        We have a loss <M>{"\\mathcal{L}"}</M> that depends on all the parameters{" "}
        <M>{"\\theta = \\{W_1, \\mathbf{b}_1, W_2, \\mathbf{b}_2, \\dots\\}"}</M>. We want the{" "}
        <M>{"\\theta"}</M> that makes <M>{"\\mathcal{L}"}</M> smallest. With millions of parameters
        we can’t solve for it algebraically — but we can <em>walk downhill</em>.
      </p>
      <p>
        Picture the loss as a landscape: the parameters are your position, the height is the loss.
        You’re blindfolded on this surface and want to reach a valley. The trick: feel the slope
        under your feet and step in the steepest <em>downhill</em> direction. The mathematical
        “slope” of <M>{"\\mathcal{L}"}</M> with respect to the parameters is the{" "}
        <strong>gradient</strong> <M>{"\\nabla_\\theta \\mathcal{L}"}</M> — a vector holding the
        partial derivative <M>{"\\partial \\mathcal{L} / \\partial \\theta_i"}</M> for every
        parameter. It points in the direction of steepest <em>increase</em>, so we step the opposite
        way. That single rule is <strong>gradient descent</strong>:
      </p>
      <MB>{String.raw`\theta \;\leftarrow\; \theta \;-\; \eta\,\nabla_\theta \mathcal{L}`}</MB>
      <p>
        Read it as: “new parameters equal old parameters, minus a small step in the uphill
        direction.” The scalar <M>{"\\eta"}</M> (eta) is the <strong>learning rate</strong> — the
        size of each step, and arguably the single most important number you’ll tune:
      </p>
      <ul>
        <li>
          <strong>Too small</strong> and training crawls, taking forever (or stalling in a shallow
          dip).
        </li>
        <li>
          <strong>Too large</strong> and you overshoot the valley, bouncing off the walls or
          diverging entirely — the loss blows up to <code>NaN</code>.
        </li>
      </ul>
      <p>
        Repeat the update thousands of times — each on a fresh mini-batch of data, which makes it{" "}
        <em>stochastic</em> gradient descent (SGD) — and the parameters slide steadily toward a region
        of low loss. The one piece we haven’t explained is the hardest: how do you actually compute{" "}
        <M>{"\\nabla_\\theta \\mathcal{L}"}</M> for a deep network? That’s backpropagation.
      </p>

      <Callout type="tip" title="Why the gradient is the steepest direction">
        <p>
          Among all unit-length directions you could step, the one that increases{" "}
          <M>{"\\mathcal{L}"}</M> fastest is exactly <M>{"\\nabla \\mathcal{L}"}</M> (this falls out
          of the dot product <M>{"\\nabla\\mathcal{L}\\cdot \\mathbf{u}"}</M> being maximized when{" "}
          <M>{"\\mathbf{u}"}</M> aligns with the gradient). Negating it gives steepest{" "}
          <em>descent</em>. Gradient descent is greedy and local — it follows the slope right under
          your feet — which is why the learning rate and the shape of the landscape matter so much.
        </p>
      </Callout>

      <h2>Backpropagation and the chain rule</h2>
      <p>
        The loss depends on the parameters through a long chain of operations: weights →
        pre-activations → activations → next layer → … → prediction → loss. To know how a weight buried
        in the first layer affects the final loss, we need the derivative of a deeply nested
        composition. Calculus has exactly the right tool: the <strong>chain rule</strong>. For a
        composition <M>{"L = f(g(h(\\theta)))"}</M>, the derivative is the product of the local
        derivatives along the way:
      </p>
      <MB>{String.raw`\frac{\partial L}{\partial \theta} \;=\; \frac{\partial L}{\partial f}\cdot\frac{\partial f}{\partial g}\cdot\frac{\partial g}{\partial h}\cdot\frac{\partial h}{\partial \theta}`}</MB>
      <p>
        <strong>Backpropagation</strong> is nothing more than the chain rule applied systematically,
        from the loss backward to every parameter — reusing shared sub-results so nothing is computed
        twice. The algorithm has two phases:
      </p>
      <ol>
        <li>
          <strong>Forward pass.</strong> Run the input through the network, computing and{" "}
          <em>caching</em> every intermediate value (each <M>{"z"}</M> and <M>{"a"}</M>), and finally
          the loss.
        </li>
        <li>
          <strong>Backward pass.</strong> Start at the loss with{" "}
          <M>{"\\partial \\mathcal{L}/\\partial \\hat{y}"}</M>, then walk <em>backward</em> through
          the layers. At each step, multiply the gradient flowing in by that operation’s{" "}
          <em>local</em> derivative to get the gradient flowing out. A matrix multiply{" "}
          <M>{"\\mathbf{x}W"}</M> passes gradient back as <M>{"W^{\\top}"}</M>; an activation{" "}
          <M>{"\\phi"}</M> multiplies by its slope <M>{"\\phi'(z)"}</M> (for ReLU, that’s a 0/1 gate
          — closed wherever the unit was inactive).
        </li>
      </ol>
      <p>
        Toggle the backward pass in Figure 2.2 to see the direction reverse: during the forward pass,
        numbers flow left → right to produce a prediction; during the backward pass, the{" "}
        <span className="text-violet-300">gradient signal</span> flows right → left, from the loss
        back toward the inputs, depositing a gradient on every weight it crosses.
      </p>

      <Callout type="math" title="Why one backward pass is enough">
        <p>
          The naive way to estimate a gradient is finite differences: nudge one parameter, re-run the
          whole forward pass, see how the loss changed — then repeat for <em>every</em> parameter. For{" "}
          <M>{"P"}</M> parameters that costs <M>{"P"}</M> forward passes, hopeless when{" "}
          <M>{"P"}</M> is in the billions. Backprop computes the gradient with respect to{" "}
          <em>all</em> <M>{"P"}</M> parameters in a single backward pass — about the same cost as one
          forward pass — by reusing the cached intermediates and pushing one shared error signal
          backward. This efficiency (often called <em>reverse-mode automatic differentiation</em>) is
          the algorithm that makes training large networks tractable.
        </p>
      </Callout>

      <h2>Putting it together: training a tiny network</h2>
      <p>
        Let’s make all of this concrete. Below is a complete training step for the same network you’ve
        been watching — 2 inputs → 3 hidden (ReLU) → 1 output — written in plain NumPy so every
        gradient is visible. We do a forward pass, compute the MSE loss, backpropagate by hand, and
        take one gradient-descent step. The comments name each shape; the gradients are exactly what
        the chain rule prescribes (and they match a finite-difference check to six decimals).
      </p>

      <CodeBlock language="python" filename="tiny_nn.py">
{`import numpy as np

# One training example: 2 inputs -> target.  (batch of N=1 for clarity)
x = np.array([[1.0, -2.0]])     # shape (1, 2)
y = np.array([[1.0]])           # shape (1, 1)  the target

# Parameters: 2 -> 3 -> 1.  Fixed numbers so the run is reproducible.
W1 = np.array([[ 0.5,  0.3, -0.2],     # (2, 3)
               [-0.4,  0.1,  0.6]])
b1 = np.array([[0.1, 0.0, 0.2]])       # (1, 3)
W2 = np.array([[ 0.7], [-0.5], [0.3]]) # (3, 1)
b2 = np.array([[0.05]])                # (1, 1)
lr = 0.1                               # learning rate (eta)

def relu(z):       return np.maximum(0.0, z)
def relu_grad(z):  return (z > 0).astype(z.dtype)   # ReLU'(z): 1 if z>0 else 0

# ---------- forward pass (cache the intermediates we'll reuse) ----------
z1   = x @ W1 + b1          # (1, 3)  hidden pre-activations
h    = relu(z1)            # (1, 3)  hidden activations
yhat = h @ W2 + b2         # (1, 1)  prediction (linear output head)
loss = np.mean((yhat - y) ** 2)        # scalar  MSE
print(f"prediction={yhat.item():.4f}  loss={loss:.4f}")

# ---------- backward pass (chain rule, loss -> parameters) ----------
N = x.shape[0]
dyhat = (2.0 / N) * (yhat - y)         # (1, 1)  d loss / d yhat
dW2   = h.T @ dyhat                    # (3, 1)
db2   = dyhat.sum(axis=0, keepdims=True)
dh    = dyhat @ W2.T                   # (1, 3)  push back through W2
dz1   = dh * relu_grad(z1)             # (1, 3)  through the ReLU gate
dW1   = x.T @ dz1                      # (2, 3)
db1   = dz1.sum(axis=0, keepdims=True)

# ---------- one gradient-descent update: theta <- theta - lr * grad ----------
W1 -= lr * dW1;  b1 -= lr * db1
W2 -= lr * dW2;  b2 -= lr * db2

# verify the step reduced the loss
new = np.mean(((relu(x @ W1 + b1) @ W2 + b2) - y) ** 2)
print(f"loss after one step={new:.4f}   (down from {loss:.4f})")`}
      </CodeBlock>

      <p>
        That is a neural network learning, with no library doing anything for you. Running it prints a
        loss that drops after a single step. Repeat the forward/backward/update loop a few hundred
        times and the prediction converges on the target. Now here is the same network in PyTorch —
        which builds the layers for you, and (crucially) computes the entire backward pass
        automatically with <code>loss.backward()</code>. You define only the forward computation;
        PyTorch records it and differentiates it for you.
      </p>

      <CodeBlock language="python" filename="tiny_nn_torch.py" highlight={[22, 23]}>
{`import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.tensor([[1.0, -2.0]])        # (1, 2)
y = torch.tensor([[1.0]])              # (1, 1)

# Same architecture: Linear(2->3) -> ReLU -> Linear(3->1)
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   # the update rule

for step in range(500):
    optimizer.zero_grad()        # clear gradients from the previous step
    yhat = model(x)              # forward pass
    loss = loss_fn(yhat, y)      # measure wrongness
    loss.backward()              # backprop: fill every .grad via the chain rule
    optimizer.step()             # theta <- theta - lr * grad
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.5f}")

print("final prediction:", model(x).item())`}
      </CodeBlock>

      <p>
        Look at the highlighted lines: <code>loss.backward()</code> is the entire backward pass we
        wrote by hand above, and <code>optimizer.step()</code> is{" "}
        <M>{"\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta\\mathcal{L}"}</M>. This
        five-line training loop — zero gradients, forward, loss, backward, step — is the{" "}
        <em>exact</em> loop used to pretrain a frontier LLM. The model gets vastly bigger and the loss
        becomes cross-entropy, but the choreography never changes.
      </p>

      <h2>Where this goes next</h2>
      <p>
        You now own the complete learning engine: neurons combine a linear map with a nonlinearity;
        layers stack them into a forward pass of matrix multiplies; a loss measures wrongness;
        gradient descent walks the parameters downhill; and backpropagation supplies the gradients in
        a single efficient backward pass. This machinery is not specific to language — it is{" "}
        <em>the</em> way every modern neural network is trained.
      </p>
      <p>
        Everything ahead is built on it. The <a href="/chapter/embeddings" className="prose-link">embedding</a>{" "}
        tables, the <a href="/chapter/self-attention" className="prose-link">self-attention</a> layers,
        the feed-forward MLPs inside every transformer block — all of them are just differentiable
        functions whose parameters this same loop adjusts. When we run{" "}
        <a href="/chapter/training-loop" className="prose-link">the training loop</a> at scale in
        Chapter 15, it will be the very five-line loop you just read — forward, loss, backward, step —
        driving trillions of tokens through a model. Next, in{" "}
        <a href="/chapter/language-modeling" className="prose-link">Chapter 3</a>, we reframe “predict
        the next token” as the precise probability problem an LLM is trained to solve.
      </p>
    </>
  );
}
