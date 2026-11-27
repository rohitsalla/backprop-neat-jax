# Algorithm and implementation notes

## Representation

A genome has two dictionaries keyed by historical ID: nodes and connections.
Input node IDs are 0 and 1, the output ID is 2, and hidden IDs start at 3.
A node stores its type, immutable depth, and trainable bias. A connection stores
its source, target, innovation number, trainable weight, and enabled flag.
Disabled genes remain available to compatibility and crossover, but are excluded
from forward evaluation, gradients, and the enabled-connection complexity cost.

For every edge `u -> v`, `depth(u) < depth(v)`. Inputs have depth 0; output depth
is 1. Splitting an edge creates a midpoint-depth hidden node. This invariant is
checked after every reproduction mutation and when loading a saved genome.
If floating-point subdivision cannot produce a strictly intermediate depth,
that edge cannot be split further. Node IDs need not be contiguous within a
genome; packing maps historical IDs to dense topological slots.

## Differentiable graph execution

For each non-input node in topological order:

```math
z_j = b_j + \sum_i M_{ij} w_{ij} a_i,\qquad
a_j = \begin{cases} z_j & j\text{ is output},\\ \tanh(z_j) & \text{otherwise}.\end{cases}
```

`M` is a binary edge mask. Padded and input biases are masked out of the
objective; input values are copied from the data. The output slot is stored
explicitly, since different genomes have different node counts. Padding after
the output cannot change its value. The forward pass uses a fixed-capacity
`lax.fori_loop`; its static bounds support reverse-mode autodifferentiation.

The loss for training is

```math
L = \frac{1}{n}\sum_k \operatorname{BCEWithLogits}(z_k,y_k)
  + \lambda\frac{\sum_{ij}(M_{ij}w_{ij})^2+\sum_j(m_j b_j)^2}
                     {\max(1,\sum_{ij}M_{ij}+\sum_j m_j)}.
```

Each genome has independent Adam moments and a per-genome gradient norm clip
of 5. `vmap` batches genomes; `scan` batches training steps into a compiled
loop. Full-batch training is intentional: no additional minibatch random state
is needed, and the benchmark datasets are small. Moments reset after topology
mutation because historical parameters move to different packed slots; trained
gene values survive via unpacking. Adam uses Optax defaults for its moments.

Fixed array capacity makes the compiled shape independent of the graph's actual
size. It does not make topology differentiable: structural evolution is a Python
and NumPy operation outside the JAX graph. For capacity `C`, population `P`,
batch `B`, and `S` Adam steps, the dense padded arithmetic is approximately
`O(S * P * B * C^2)`. Small, growing graphs still pay for the allocated capacity.

## Historical innovations and crossover

One registry is shared across the population for the full run:

- `(source ID, target ID) -> connection innovation ID`.
- `split connection innovation ID -> hidden node ID`.

Matching innovations align independently evolved genomes. A split disables the
old edge and creates weights 1 and the old weight around a new `tanh` node with
zero bias. This encourages useful inheritance but changes the function.
Historical node depths remain consistent because the original endpoints have
immutable depths.

Crossover chooses a fitter primary parent (random primary on exact ties),
copies its node/connection inventory, and samples weights/enabled states for
matching connection genes from either parent. If either matching gene is
disabled, the child gene is disabled with probability 0.75. Matching node biases
are sampled independently. Primary-only inheritance of disjoint genes preserves
the configured capacity and all required endpoint nodes.

## Compatibility and species

Let `E` and `D` be excess and disjoint connection counts, including disabled
genes, and `W` the mean absolute matching-weight difference. Then

```math
\delta = c_E E/N + c_D D/N + c_W W.
```

`N` is the larger connection-gene count, or 1 when that count is less than 20.
No matching genes gives `W=0`. Genomes join the first compatible species under
the configured threshold; otherwise they start a species. The next
representative is the species champion. Best fitness and last-improved
generation persist even when the representative changes. The compatibility
threshold is fixed; this implementation does not target an automatic species
count. Species without members are dropped.

## Fitness sharing and reproduction

Validation BCE, without L2, supplies selection fitness:

```math
f_i = -\operatorname{BCE}_{val,i}-\alpha\,\#\operatorname{enabledEdges}_i.
```

Gradient training never uses validation data. Repeated validation selection
means validation is a search objective, not a final unbiased performance score.

Negative fitness is transformed into a positive reproduction score:

```math
q_i = \exp\left(\operatorname{clip}((f_i-f_{best})/T,-50,0)\right),\qquad
Q_s = \frac{1}{|s|}\sum_{i\in s}q_i.
```

Species offspring quotas are proportional to `Q_s`, which is explicit fitness
sharing of the transformed scores. One slot is reserved for the global
population champion. The remaining `P-1` slots use floor allocation followed
by largest fractional remainders; ties follow stable species order. A species
with two or more allocated slots retains a species elite. Other slots are
children of uniformly sampled parents from the species' top survivor fraction.
Single-slot species mutate their child, so a population of singleton species
does not freeze. The global elite can also appear as a species elite; duplicate
elites are permitted and the total population size remains exactly `P`.

Stagnant species receive no allocation unless they contain the current
population champion. Reproduction is within species. Weight mutation perturbs
each connection with probability 0.9 or resets it otherwise, conditional on
the genome's weight-mutation event; non-input biases receive Gaussian noise.
Node and edge mutation events are independently sampled afterwards.

## Reproducibility and model selection

Train/validation/test use three child `SeedSequence` streams. Evolution uses a
separate seed derived from `[seed, 947]`. JAX training is deterministic on a
given backend for these operations, subject to normal floating-point/backend
differences. No JAX random operation is required after NumPy initialization.

The best penalized validation-fitness genome across all evaluated generations
is cloned immediately, protecting it from later mutation or further training.
After evolution ends, final metrics are computed for this exact clone on all
splits. Test accuracy/loss is never logged during the search. The plot then
overlays test inputs and labels for visual inspection. `selected_generation`
is zero-indexed and saved in metrics.

Saving includes the exact dataset, complete settings, package versions, history,
and champion genes. It does not include every population or registry state;
resuming evolution is deliberately outside the current checkpoint contract.

## Limits of the controls

The `evolution` control disables all gradient calls but retains weight and
topology mutations. The `fixed` control fixes a fully connected one-hidden-layer
graph while retaining the hybrid population search and weight operations.
These controls isolate algorithmic components; they are not compute-matched
comparisons. Larger hidden layers and topology capacities increase dense JAX
execution costs, and evolution-only training omits those gradient costs.

For a serious comparison, predeclare settings and seed lists, report all runs,
use test data only after selection, compare wall-clock budgets as well as
generations, and include a separately implemented single-model MLP baseline.

## References

- [Original NEAT paper](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [JAX automatic differentiation and vectorization](https://docs.jax.dev/en/latest/101/transformations.html)
- [JAX control flow](https://docs.jax.dev/en/latest/control-flow.html)
- [Optax Adam](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam)
