# lau-ricci-curvature-agents

**Ollivier-Ricci and Forman-Ricci curvature on agent interaction graphs.** Detect bottlenecks, prove consensus bounds, design optimal fleet topologies — all via discrete curvature.

96 tests · MIT license · `nalgebra` + `serde`

---

## What This Does

This crate treats an agent fleet as a graph and computes **discrete curvature** on its edges. Curvature tells you:

- **Positive curvature** → agents agree, information flows freely, fast consensus
- **Zero curvature** → random mixing, neutral convergence
- **Negative curvature** → bottlenecks, information doesn't cross, slow/divergent consensus

The crate implements:

1. **Ollivier-Ricci curvature** via optimal transport (Wasserstein-1 distance between neighborhood measures)
2. **Forman-Ricci curvature** — a simpler combinatorial alternative
3. **Belief space curvature** — Fisher-Rao metric on probability simplex, Riemann tensor, sectional/scalar curvature
4. **Concentration of measure** — curvature → variance/mixing/consensus time bounds
5. **Bonnet-Myers theorem** — positive curvature → bounded diameter → finite graph
6. **Curvature flow** — evolve topology to increase curvature (triangle closing, rewiring)
7. **Bottleneck detection** — find negative-curvature edges, suggest fixes
8. **Fleet design** — engineer topologies that maximize curvature for fast consensus

---

## Key Idea

**Ollivier-Ricci curvature** of an edge (x,y):

**κ(x,y) = 1 − W₁(μₓ, μᵧ) / d(x,y)**

where μₓ is the probability measure around x (lazy random walk), μᵧ around y, and W₁ is the 1-Wasserstein (Earth Mover's) distance.

- κ > 0: neighbors overlap significantly → information flows well
- κ < 0: neighbors are far apart → bottleneck

---

## Install

```toml
[dependencies]
lau-ricci-curvature-agents = "0.1.0"
```

Dependencies: `nalgebra = "0.33"` (serde), `serde = "1"`, `serde_json = "1"`, `approx = "0.5"` (dev).

---

## Quick Start

```rust
use lau_ricci_curvature_agents::{
    AgentGraph, OllivierRicciCurvature, FormanRicciCurvature,
    MeasureStrategy, TransportSolver, BottleneckDetector, FleetDesigner,
};

fn main() {
    // Build a fleet graph
    let mut graph = AgentGraph::new(8);
    graph.add_edge(0, 1, 1.0);
    graph.add_edge(1, 2, 1.0);
    graph.add_edge(2, 3, 1.0);
    graph.add_edge(3, 4, 1.0);
    graph.add_edge(4, 5, 1.0);
    graph.add_edge(5, 6, 1.0);
    graph.add_edge(6, 7, 1.0);
    // This is a path graph — has negative curvature in the middle

    // Compute Ollivier-Ricci curvature
    let orc = OllivierRicciCurvature::new(
        MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
        TransportSolver::Exact,
    );
    let avg_k = orc.average_curvature(&graph);
    let min_k = orc.min_curvature(&graph);
    println!("Average curvature: {:.4}", avg_k);
    println!("Min curvature:     {:.4}", min_k);

    // Detect bottlenecks
    let detector = BottleneckDetector::new(0.0);
    let bottlenecks = detector.detect(&graph);
    for b in &bottlenecks {
        println!("Bottleneck: edge {:?}, severity {:.4}", b.edge, b.severity);
    }

    // Design a better fleet topology
    let designer = FleetDesigner::default();
    let result = designer.design(8, 4, 0.1);
    println!("Designed fleet: {} edges, curvature {:.4}, diameter {}",
        result.graph.num_edges(), result.min_curvature, result.diameter);
}
```

---

## API Reference

### Graph (`graph`)

| Type | Description |
|------|-------------|
| `AgentGraph` | Weighted undirected graph with BFS distance, Laplacian, connected components |
| `AgentId` | `usize` — node identifier |
| `EdgeId` | `(AgentId, AgentId)` — sorted pair |

**`AgentGraph`** methods:
- `new(n)`, `complete(n)`, `path(n)`, `cycle(n)`, `star(n)`, `grid(rows, cols)`
- `add_edge(a, b, w)`, `edge_weight(a, b)`, `neighbors(a)`, `degree(a)`
- `edges()`, `agents()`, `has_edge(a, b)`, `num_agents()`, `num_edges()`
- `distance(a, b)` — BFS shortest path
- `diameter()` — longest shortest path
- `laplacian()` — `DMatrix<f64>` combinatorial Laplacian (D − A)
- `connected_components()` — `Vec<Vec<AgentId>>`

### Ollivier-Ricci Curvature (`ollivier_ricci`)

| Type | Description |
|------|-------------|
| `OllivierRicciCurvature` | Main curvature computer: measure strategy + transport solver |
| `MeasureStrategy` | `LazyRandomWalk{alpha}`, `UniformNeighbors`, `Custom{probs}` |
| `TransportSolver` | `Exact` (greedy LP), `Sinkhorn{reg, iterations}` |

**`OllivierRicciCurvature`** methods:
- `new(measure, solver)` — construct with chosen strategy
- `measure_at(graph, a)` → `Vec<(AgentId, f64)>` — neighborhood probability measure
- `wasserstein_1(graph, mu_x, mu_y)` → `f64` — optimal transport distance
- `edge_curvature(graph, a, b)` → `f64` — κ(a,b) = 1 − W₁/d
- `all_curvatures(graph)` → `Vec<((AgentId, AgentId), f64)>`
- `average_curvature(graph)` → `f64`
- `min_curvature(graph)` → `f64`
- `max_curvature(graph)` → `f64`

### Forman-Ricci Curvature (`forman_ricci`)

| Type | Description |
|------|-------------|
| `FormanRicciCurvature` | Combinatorial curvature: F(u,v) = 4 − deg(u) − deg(v) (unweighted) |

**Methods:**
- `new(weighted)`, `edge_curvature(graph, a, b)`, `all_curvatures(graph)`, `average_curvature(graph)`, `min_curvature(graph)`, `max_curvature(graph)`

### Belief Space Curvature (`belief_curvature`)

| Type | Description |
|------|-------------|
| `BeliefState` | Probability distribution: `uniform(n)`, `from_probs(v)`, `dirac(n, k)` |
| `BeliefSpaceCurvature` | Fisher-Rao metric, Riemann tensor, sectional/scalar curvature on simplex |

**`BeliefSpaceCurvature`** methods:
- `new(categories)` — construct for n-category belief space
- `fisher_matrix(belief)` → `(n−1)×(n−1)` Fisher information matrix
- `christoffel_symbols(belief)` → `Vec<Vec<Vec<f64>>>` — Γᵏᵢⱼ connection coefficients
- `riemann_tensor(belief)` → 4D array Rˡᵢⱼₘ
- `sectional_curvature(belief, i, j)` → `f64`
- `scalar_curvature(belief)` → `f64`
- `fisher_rao_distance(b1, b2)` → `f64` — √(2 Σ(√pᵢ − √qᵢ)²)
- `fleet_belief_curvature(beliefs)` → `f64` — average scalar curvature across fleet

### Concentration of Measure (`concentration`)

| Type | Description |
|------|-------------|
| `CurvatureConcentration` | Converts curvature → variance/mixing/consensus bounds |
| `ConcentrationResult` | `min_curvature`, `variance_bound`, `mixing_time_bound`, `consensus_time_bound` |

**Theorem**: If κ ≥ κ₀ > 0, then:
- Variance of any 1-Lipschitz f ≤ n/(4κ₀)
- Mixing time ≤ O(log(n)/κ₀)
- Consensus time ≤ O(n/κ₀)

### Bonnet-Myers (`bonnet_myers`)

| Type | Description |
|------|-------------|
| `BonnetMyers` | Checks Bonnet-Myers diameter bound |
| `BonnetMyersResult` | `min_curvature`, `diameter_bound`, `actual_diameter`, `satisfies_bonnet_myers` |

**Theorem**: If κ ≥ κ₀ > 0 everywhere, then diam(G) ≤ ⌊1/κ₀⌋.

### Curvature Flow (`curvature_flow`)

| Type | Description |
|------|-------------|
| `CurvatureFlow` | Evolve graph topology to increase curvature |
| `FlowStrategy` | `TriangleClosing`, `Rewire`, `NeighborhoodFilling` |
| `FlowStep` | `avg_before`, `avg_after`, `edges_added`, `edges_removed`, `improvement` |

**Methods:**
- `step(graph)` → `(AgentGraph, FlowStep)` — one topology evolution step
- `evolve(graph, steps)` → `(AgentGraph, Vec<FlowStep>)` — multi-step evolution

### Bottleneck Detection (`bottleneck`)

| Type | Description |
|------|-------------|
| `BottleneckDetector` | Finds edges with curvature below threshold |
| `Bottleneck` | `edge`, `ollivier_curvature`, `forman_curvature`, `severity`, `suggested_fixes` |

**Methods:**
- `new(threshold)`, `detect(graph)` → `Vec<Bottleneck>`, `fix_all(graph)` → `AgentGraph`

### Fleet Design (`fleet`)

| Type | Description |
|------|-------------|
| `FleetDesigner` | Design topologies maximizing curvature |
| `FleetDesignGoals` | `n_agents`, `target_min_curvature`, `max_degree`, `ensure_bonnet_myers` |
| `FleetDesignResult` | `graph`, `avg_curvature`, `min_curvature`, `diameter`, `bottleneck_count`, `bonnet_myers_satisfied`, `estimated_consensus_time` |

**Methods:**
- `design(n, max_degree, target_curvature)` → `FleetDesignResult`
- `design_with_goals(goals)` → `FleetDesignResult`

---

## How It Works

### Pipeline

```
AgentGraph
    │
    ├─→ OllivierRicciCurvature ──→ edge curvatures (via optimal transport)
    ├─→ FormanRicciCurvature   ──→ edge curvatures (combinatorial)
    │
    ├─→ CurvatureConcentration ──→ variance/mixing/consensus bounds
    ├─→ BonnetMyers            ──→ diameter bound
    │
    ├─→ BottleneckDetector     ──→ negative-curvature edges + fixes
    ├─→ CurvatureFlow          ──→ topology evolution
    └─→ FleetDesigner          ──→ optimal topology from scratch
```

### Optimal Transport (Wasserstein-1)

The core computation: given two probability measures μₓ, μᵧ on graph nodes, find the minimum-cost transport plan.

**Exact solver**: Greedy matching — sort (source, target) pairs by graph distance, transport mass in cost order. Exact for discrete measures.

**Sinkhorn solver**: Iterative entropic regularization — Kᵢⱼ = exp(−d(xᵢ,yⱼ)/ε), alternating projections: u ← a./(Kv), v ← b./(Kᵀu). Approximate but smooth.

### Curvature Flow Strategies

- **TriangleClosing**: For each negative-curvature edge (u,v), find shared neighbors w. If u and v share a neighbor but aren't directly connected to that neighbor's neighbors, add those edges.
- **Rewire**: Remove the worst-curvature edge. Add an edge that most improves curvature.
- **NeighborhoodFilling**: For each negative-curvature edge (u,v), connect the neighbors of u to the neighbors of v.

### Belief Space Geometry

Agents hold beliefs p ∈ Δⁿ⁻¹ (probability simplex). The Fisher information metric:

gᵢⱼ(p) = δᵢⱼ/pᵢ + 1/pₙ

This induces Riemannian curvature via Christoffel symbols → Riemann tensor → sectional curvature → scalar curvature.

---

## The Math

### Ollivier-Ricci Curvature

For adjacent nodes x, y with neighborhood measures μₓ, μᵧ:

**κ(x,y) = 1 − W₁(μₓ, μᵧ) / d(x,y)**

where W₁(μ,ν) = inf{Σ cᵢⱼ d(xᵢ,yⱼ) : T is a transport plan from μ to ν}

Range: κ ∈ [−1, 1]
- κ = 1: μₓ = μᵧ (perfect overlap, e.g. complete graph)
- κ = 0: tree-like / flat
- κ = −1: maximally divergent neighborhoods

### Forman-Ricci Curvature

For edge e = (u,v) with weight wₑ:

**F(e) = wₑ(wᵤ/wₑ + wᵥ/wₑ − deg(u) − deg(v))**

Unweighted: F(u,v) = 4 − deg(u) − deg(v)

Range: (−∞, 4]

### Curvature-Concentration Inequality

If κ(x,y) ≥ κ₀ > 0 for all edges:

- **Variance bound**: Var(f) ≤ n/(4κ₀) for any 1-Lipschitz f
- **Mixing time**: τ_mix ≤ O(log(n)/κ₀)
- **Consensus time**: τ_consensus ≤ O(n/κ₀)

### Bonnet-Myers Theorem

If κ ≥ κ₀ > 0 everywhere, then **diam(G) ≤ ⌊1/κ₀⌋**.

Classical analog: positive Ricci curvature on a Riemannian manifold bounds the diameter (Bonnet-Myers theorem in Riemannian geometry).

### Fisher-Rao Metric

On the probability simplex Δⁿ⁻¹, the Fisher information metric is:

gᵢⱼ(θ) = δᵢⱼ/θᵢ + 1/θₙ    (for i,j ∈ {1,...,n−1})

Fisher-Rao distance: d(p,q) = 2√(Σ(√pᵢ − √qᵢ)²)

This is the **Hellinger distance** scaled by 2, and equals the geodesic distance on the simplex under the Fisher metric.

---

## License

MIT
