# lau-ricci-curvature-agents

**Ollivier-Ricci and Forman-Ricci curvature on agent interaction graphs** — detect bottlenecks, predict consensus time, bound graph diameter via Bonnet-Myers, evolve topology via curvature flow, and design optimal fleet communication networks.

## What This Does

Ricci curvature on graphs measures how "spread out" or "pinched" the local geometry is:

- **Positive curvature** → neighbors overlap heavily → information flows well → fast consensus
- **Zero curvature** → random mixing (like a grid or tree)
- **Negative curvature** → neighborhoods diverge → bottlenecks, over-squashing, slow convergence

This crate provides:

- **Ollivier-Ricci curvature** — κ(x,y) = 1 − W₁(μ_x, μ_y)/d(x,y) via optimal transport
- **Forman-Ricci curvature** — simpler combinatorial: F(u,v) = 4 − deg(u) − deg(v)
- **Bottleneck detection** — find negative-curvature edges and suggest fixes
- **Curvature flow** — evolve topology to increase curvature (triangle closing, rewiring)
- **Bonnet-Myers theorem** — positive curvature → bounded diameter
- **Concentration inequality** — curvature → mixing/consensus time bounds
- **Belief space curvature** — Fisher information metric on agent belief distributions
- **Fleet topology design** — engineer graphs with target minimum curvature

## Key Idea

The Ollivier-Ricci curvature κ(x,y) = 1 − W₁(μ_x, μ_y)/d(x,y) measures how much the probability distributions around two adjacent nodes overlap. High overlap (high κ) means information spreads efficiently. The key applications:

1. **Bottleneck detection**: Edges with κ < 0 are communication bottlenecks
2. **Consensus time**: If κ_min > 0, consensus time is O(n/κ_min)
3. **Diameter bound**: If κ_min > 0, diameter ≤ 1/κ_min (Bonnet-Myers)
4. **Topology design**: Optimize the graph to maximize minimum curvature

## Install

```toml
[dependencies]
lau-ricci-curvature-agents = "0.1.0"
```

## Quick Start

```rust
use lau_ricci_curvature_agents::*;

// Build an agent interaction graph
let mut graph = AgentGraph::new(6);
graph.add_edge(0, 1, 1.0);
graph.add_edge(1, 2, 1.0);
graph.add_edge(2, 3, 1.0);
graph.add_edge(3, 4, 1.0);
graph.add_edge(4, 5, 1.0);

// Compute Ollivier-Ricci curvature
let orc = OllivierRicciCurvature::default();
let curvatures = orc.all_curvatures(&graph);
for ((a, b), k) in &curvatures {
    println!("Edge ({}, {}): κ = {:.4}", a, b, k);
}
println!("Average curvature: {:.4}", orc.average_curvature(&graph));

// Compute Forman-Ricci curvature (much faster)
let forman = FormanRicciCurvature::default();
println!("Forman curvature (0,1): {:.1}", forman.edge_curvature(&graph, 0, 1));

// Detect bottlenecks
let detector = BottleneckDetector::new(0.0);
let bottlenecks = detector.detect(&graph);
for bn in &bottlenecks {
    println!("Bottleneck ({}, {}): severity = {:.2}", bn.edge.0, bn.edge.1, bn.severity);
}

// Bonnet-Myers analysis
let bm = BonnetMyers::default();
let result = bm.analyze(&graph);
println!("Diameter bound: {:?}, actual: {}", result.diameter_bound, result.actual_diameter);
```

## API Reference

### `graph` — Agent Interaction Graph

| Type | Description |
|------|-------------|
| `AgentGraph` | Weighted undirected graph. Agents are nodes 0..n, edges have weights. |
| `AgentId` | Type alias: `usize` |
| `EdgeId` | Type alias: `(usize, usize)` |

**`AgentGraph` methods:**
- `new(n)`, `num_agents()`, `num_edges()`
- `add_edge(a, b, w)`, `remove_edge(a, b)`
- `has_edge(a, b) → bool`, `edge_weight(a, b) → Option<f64>`
- `neighbors(a) → Vec<(AgentId, Weight)>`, `degree(a) → usize`
- `edges() → Vec<(AgentId, AgentId)>`
- `adjacency_matrix() → DMatrix<f64>`, `laplacian() → DMatrix<f64>`
- `diameter() → usize`, `shortest_path(a, b) → Option<Vec<usize>>`
- `complete(n)`, `ring(n)`, `path(n)`, `star(n)` — Constructors

---

### `ollivier_ricci` — Ollivier-Ricci Curvature

| Type | Description |
|------|-------------|
| `MeasureStrategy` | How to build neighborhood measure: LazyRandomWalk{alpha}, UniformNeighbors, Custom{probs}. |
| `TransportSolver` | Optimal transport solver: Exact, Sinkhorn{reg, iterations}. |
| `OllivierRicciCurvature` | Main curvature computer. |

**`OllivierRicciCurvature` methods:**
- `new(measure, solver)`, `default()` — α=0.5 lazy random walk, exact solver
- `measure_at(&graph, a) → Vec<(AgentId, f64)>` — Neighborhood probability distribution
- `wasserstein_1(&support_a, &support_b, &graph) → f64` — W₁ distance between measures
- `edge_curvature(&graph, a, b) → f64` — κ(a,b) = 1 − W₁/d
- `all_curvatures(&graph) → Vec<((AgentId, AgentId), f64)>` — All edges
- `average_curvature(&graph) → f64`, `min_curvature(&graph) → f64`

---

### `forman_ricci` — Forman-Ricci Curvature

| Type | Description |
|------|-------------|
| `FormanRicciCurvature` | Combinatorial curvature: F(u,v) = 4 − deg(u) − deg(v). |

**Methods:**
- `new(weighted)`, `default()` — Unweighted
- `edge_curvature(&graph, a, b) → f64` — Single edge
- `all_curvatures(&graph) → Vec<((AgentId, AgentId), f64)>`
- `average_curvature(&graph) → f64`, `min_curvature(&graph) → f64`

---

### `bottleneck` — Bottleneck Detection

| Type | Description |
|------|-------------|
| `Bottleneck` | A detected bottleneck: edge, curvatures (OR + Forman), severity, suggested fixes. |
| `BottleneckDetector` | Detect edges with curvature below threshold. |

**`BottleneckDetector`:**
- `new(threshold)`, `default()` — threshold = 0.0
- `detect(&graph) → Vec<Bottleneck>` — All bottleneck edges with suggested fixes

---

### `curvature_flow` — Curvature-Driven Topology Evolution

| Type | Description |
|------|-------------|
| `FlowStrategy` | Enum: TriangleClosing, Rewire, NeighborhoodFilling. |
| `FlowStep` | Result: average curvature before/after, edges added/removed, improvement. |
| `CurvatureFlow` | Evolve graph to increase curvature. |

**`CurvatureFlow`:**
- `new(orc, strategy, learning_rate)`, `default()` — Triangle closing, lr=0.1
- `step(&mut graph) → FlowStep` — One evolution step
- `run(&mut graph, steps) → Vec<FlowStep>` — Multiple steps

---

### `bonnet_myers` — Bonnet-Myers Theorem

| Type | Description |
|------|-------------|
| `BonnetMyersResult` | Analysis: min curvature, diameter bound, actual diameter, satisfaction flag. |
| `BonnetMyers` | Checker: if κ ≥ κ₀ > 0, then diam(G) ≤ 1/κ₀. |

**Methods:**
- `new(orc)`, `default()`
- `diameter_bound(&graph) → Option<usize>` — Upper bound if κ_min > 0
- `analyze(&graph) → BonnetMyersResult`

---

### `concentration` — Curvature-Concentration Inequality

| Type | Description |
|------|-------------|
| `ConcentrationResult` | Bounds: variance, mixing time, consensus time. |
| `CurvatureConcentration` | Analyzer: high curvature → fast concentration. |

**Methods:**
- `new(orc)`, `default()`
- `analyze(&graph) → ConcentrationResult` — If κ_min > 0: variance ≤ n/(4κ), mixing ≤ O(log(n)/κ), consensus ≤ O(n/κ)

---

### `belief_curvature` — Belief Space Curvature

| Type | Description |
|------|-------------|
| `BeliefState` | A probability distribution over categories. Methods: `uniform(n)`, `from_probs(v)`, `dirac(n,k)`, `is_valid()`. |
| `BeliefSpaceCurvature` | Fisher information metric curvature on belief space. |

**`BeliefSpaceCurvature`:**
- `new(categories)`, `fisher_information_matrix(&belief) → DMatrix<f64>`
- `fisher_rao_distance(&b1, &b2) → f64` — Geodesic distance in belief space
- `sectional_curvature(&b1, &b2, &b3) → f64` — Curvature of the 2D section

---

### `fleet` — Fleet Topology Design

| Type | Description |
|------|-------------|
| `FleetDesignGoals` | Target: n_agents, target_min_curvature, max_degree, ensure_bonnet_myers. |
| `FleetDesignResult` | Output: graph, curvatures, diameter, bottlenecks, consensus time. |
| `FleetDesigner` | Design graphs that satisfy curvature targets. |

**`FleetDesigner`:**
- `default()`
- `design(&goals) → FleetDesignResult` — Design from scratch
- `optimize(&mut graph, steps) → FleetDesignResult` — Optimize existing graph
- `compare_topologies(n, topologies) → Vec<FleetDesignResult>` — Compare designs

## How It Works

1. **Neighborhood measures**: For each agent, build a probability distribution μ_x over its neighborhood (lazy random walk: weight α on self, (1−α)/deg on each neighbor).

2. **Optimal transport**: Compute W₁(μ_x, μ_y) — the cost of transporting the measure at x to the measure at y. The exact solver uses the network simplex; the Sinkhorn solver uses entropic regularization.

3. **Curvature**: κ(x,y) = 1 − W₁(μ_x, μ_y)/d(x,y). If κ > 0, measures overlap more than expected (positive curvature, tree-like). If κ < 0, they diverge (negative curvature, expander-like).

4. **Forman-Ricci**: A simpler formula F(u,v) = 4 − deg(u) − deg(v) that captures the combinatorial curvature without optimal transport. Fast to compute, useful as a first pass.

5. **Bottleneck detection**: Edges with κ < threshold are bottlenecks. Suggested fixes: add triangles (close 2-paths into 3-cycles) or rewire edges.

6. **Curvature flow**: Iteratively modify the graph to increase average curvature. Triangle closing is the most effective: for every pair of agents that share a neighbor, add an edge between them.

7. **Concentration**: If κ_min > 0, the graph satisfies a concentration inequality — any 1-Lipschitz function has bounded variance, and random walks mix in O(log(n)/κ_min) steps.

## The Math

### Ollivier-Ricci Curvature

For adjacent nodes x, y on a metric graph with probability measures μ_x, μ_y:
κ(x,y) = 1 − W₁(μ_x, μ_y)/d(x,y)

where W₁ is the 1-Wasserstein (earth mover's) distance. This is the discrete analog of the Ricci curvature lower bound in Riemannian geometry.

### Forman-Ricci Curvature

For an edge e = (u,v) in an unweighted graph:
F(e) = 4 − deg(u) − deg(v)

This counts "parallel edges" (common neighbors) minus "crossing edges" (other edges from u and v). High-degree nodes connected by an edge have negative Forman curvature.

### Bonnet-Myers Theorem

If a Riemannian manifold has Ric ≥ (n−1)κ > 0, then diam(M) ≤ π/√κ. The discrete analog: if all Ollivier-Ricci curvatures are ≥ κ₀ > 0, then diam(G) ≤ 1/κ₀.

### Concentration of Measure

On a graph with κ_min > 0, for any 1-Lipschitz function f:
Var(f) ≤ n/(4·κ_min)
This means agent states concentrate rapidly around the mean — fast consensus.

### Fisher Information Metric

The Fisher-Rao metric on the simplex of probability distributions gives a Riemannian structure on belief space. Its geodesic distance between beliefs p, q is:
d_FR(p, q) = 2·arccos(Σᵢ √(pᵢ qᵢ))
The sectional curvature of this space governs how beliefs diverge/converge.

## License

MIT
