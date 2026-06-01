# lau-ricci-curvature-agents

**Ollivier-Ricci and Forman-Ricci curvature on agent interaction graphs for fleet topology analysis.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-96-green.svg)]()

## What This Does

This crate applies **discrete Ricci curvature** — a notion of curvature for graphs — to agent interaction networks. The key insight:

- **High curvature** = agents agree, fast consensus, robust information flow
- **Low/negative curvature** = agents disagree, information bottlenecks, slow convergence
- **Zero curvature** = random mixing, no structure

The crate computes two types of discrete Ricci curvature:

1. **Ollivier-Ricci curvature** — via optimal transport between neighborhood measures: `κ(x,y) = 1 - W₁(μₓ, μᵧ)/d(x,y)`
2. **Forman-Ricci curvature** — a simpler combinatorial formula: `F(u,v) = 4 - deg(u) - deg(v)`

It then uses curvature to detect bottlenecks, bound consensus times, verify Bonnet-Myers diameter bounds, and evolve fleet topology to improve convergence.

**96 tests** cover graph construction, both curvature types, concentration inequalities, Bonnet-Myers, curvature flow, and bottleneck detection.

## Key Idea

On a Riemannian manifold, positive Ricci curvature means geodesics converge (think: a sphere). Negative curvature means they diverge (think: a saddle). On a graph, Ollivier-Ricci curvature captures the same idea using probability measures at each node:

```
κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
```

where `W₁` is the Wasserstein-1 (earth mover's) distance between the neighborhood distributions `μₓ` and `μᵧ`. If neighbors of `x` and `y` are similar (mass doesn't need to move far), `κ` is high — information flows easily.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-ricci-curvature-agents = { git = "https://github.com/SuperInstance/lau-ricci-curvature-agents" }
```

### Dependencies

- `nalgebra` — linear algebra
- `serde` / `serde_json` — serialization

## Quick Start

```rust
use lau_ricci_curvature_agents::*;

// Build an agent interaction graph
let mut graph = AgentGraph::new(5);
graph.add_edge(0, 1, 1.0);
graph.add_edge(1, 2, 1.0);
graph.add_edge(2, 3, 1.0);
graph.add_edge(3, 4, 1.0);
graph.add_edge(0, 4, 1.0);  // cycle = positive curvature

// Compute Ollivier-Ricci curvature
let orc = OllivierRicciCurvature::default();
let curvature_01 = orc.edge_curvature(&graph, 0, 1);
let all = orc.all_curvatures(&graph);
let avg = orc.average_curvature(&graph);
let min = orc.min_curvature(&graph);

// Compute Forman-Ricci curvature (much cheaper)
let forman = FormanRicciCurvature::new(false);  // unweighted
let f01 = forman.edge_curvature(&graph, 0, 1);
// F(0,1) = 4 - deg(0) - deg(1) = 4 - 2 - 3 = -1

// Detect bottlenecks (edges with negative curvature)
let detector = BottleneckDetector::new(0.0);  // threshold
let bottlenecks = detector.detect(&graph);
for bn in &bottlenecks {
    println!("Bottleneck at edge {:?}: κ_OR = {:?}, severity = {:.3}",
        bn.edge, bn.ollivier_curvature, bn.severity);
}

// Check Bonnet-Myers: if κ ≥ κ₀ > 0, then diam(G) ≤ 1/κ₀
let bm = BonnetMyers::default();
let result = bm.analyze(&graph);
println!("Diameter bound: {:?}", result.diameter_bound);
println!("Actual diameter: {}", result.actual_diameter);

// Concentration bounds: how fast does consensus happen?
let conc = CurvatureConcentration::default();
let bounds = conc.analyze(&graph);
println!("Mixing time ≤ {}", bounds.mixing_time_bound);
println!("Consensus time ≤ {}", bounds.consensus_time_bound);

// Improve topology via curvature flow
let mut flow = CurvatureFlow::default();
let step = flow.step(&mut graph);
println!("Average curvature: {:.3} → {:.3}", step.avg_before, step.avg_after);
```

## API Reference

### `graph` — Agent Interaction Graph

| Type/Method | Description |
|-------------|-------------|
| `AgentGraph::new(n)` | Create graph with `n` agent nodes (0-indexed) |
| `.add_edge(a, b, w)` | Add undirected weighted edge |
| `.edge_weight(a, b)` | Get weight of edge (a,b) |
| `.neighbors(a)` | List of `(neighbor, weight)` pairs |
| `.degree(a)` | Number of edges incident to agent |
| `.edges()` | All edges as sorted pairs |
| `.diameter()` | Graph diameter (max shortest path) |
| `.adjacency_matrix()` | N×N weighted adjacency matrix |
| `.laplacian()` | Graph Laplacian L = D - A |
| `AgentId` | Type alias for `usize` |

### `ollivier_ricci` — Ollivier-Ricci Curvature

`κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)`

| Type/Method | Description |
|-------------|-------------|
| `OllivierRicciCurvature::new(measure, solver)` | Configure curvature computation |
| `MeasureStrategy` | How to build neighborhood measures: `LazyRandomWalk{α}`, `UniformNeighbors`, `Custom` |
| `TransportSolver` | How to solve optimal transport: `Exact`, `Sinkhorn{reg, iters}` |
| `.edge_curvature(graph, a, b)` | κ for a single edge |
| `.all_curvatures(graph)` | κ for all edges |
| `.average_curvature(graph)` | Mean curvature |
| `.min_curvature(graph)` | Minimum curvature (the bottleneck) |
| `.measure_at(graph, a)` | Neighborhood probability distribution |

Default: lazy random walk with α=0.5 and exact transport solver.

### `forman_ricci` — Forman-Ricci Curvature

```
F(u,v) = 4 - deg(u) - deg(v)           (unweighted)
F(u,v) = wₑ(wᵤ/wₑ + wᵥ/wₑ) - deg(u) - deg(v)  (weighted)
```

| Method | Description |
|--------|-------------|
| `FormanRicciCurvature::new(weighted)` | Configure weighted or unweighted |
| `.edge_curvature(graph, a, b)` | Forman curvature for one edge |
| `.all_curvatures(graph)` | All edges |
| `.average_curvature(graph)` | Mean |
| `.min_curvature(graph)` / `.max_curvature(graph)` | Extremes |

Much cheaper to compute than Ollivier-Ricci (no optimal transport needed), but less informative.

### `belief_curvature` — Belief Space Curvature

Curvature of agent belief space using the Fisher information metric on the probability simplex.

| Type/Method | Description |
|-------------|-------------|
| `BeliefState` | A probability distribution over categories |
| `BeliefSpaceCurvature::new(n)` | Curvature computation for n categories |
| `.fisher_matrix(belief)` | Fisher information matrix at a belief |
| `.sectional_curvature(b1, b2, b3)` | Sectional curvature of belief space |
| `.ricci_curvature(belief, dir1, dir2)` | Ricci curvature in a 2-plane |

### `concentration` — Curvature-Concentration Inequalities

If κ ≥ κ₀ > 0 everywhere, then:
- Variance of any 1-Lipschitz function ≤ n/(4κ₀)
- Mixing time ≤ O(log(n)/κ₀)
- Consensus time ≤ O(n/κ₀)

| Method | Description |
|--------|-------------|
| `CurvatureConcentration::analyze(graph)` | Compute all concentration bounds |
| `.spectral_gap_bound(graph)` | Lower bound on spectral gap from curvature |

### `bonnet_myers` — Bonnet-Myers Theorem for Graphs

If κ ≥ κ₀ > 0 everywhere, then:
- `diam(G) ≤ ⌊1/κ₀⌋`
- The graph is finite and has bounded size

| Method | Description |
|--------|-------------|
| `BonnetMyers::diameter_bound(graph)` | Upper bound from curvature (None if κ ≤ 0) |
| `BonnetMyers::analyze(graph)` | Full analysis: bound, actual diameter, satisfaction |
| `BonnetMyers::check(graph)` | Boolean: does the graph satisfy BM conditions? |

### `curvature_flow` — Topology Evolution

Evolve fleet topology to increase curvature (improve consensus).

| Strategy | Description |
|----------|-------------|
| `TriangleClosing` | Add edges between agents that share neighbors |
| `Rewire` | Remove worst-curvature edge, add better one |
| `NeighborhoodFilling` | Add edges to negative-curvature neighborhoods |

| Method | Description |
|--------|-------------|
| `CurvatureFlow::step(graph)` | Execute one flow step, returns curvature change and edges modified |

### `bottleneck` — Bottleneck Detection

Edges with negative curvature are communication bottlenecks.

| Type/Method | Description |
|-------------|-------------|
| `Bottleneck` | A detected bottleneck: edge, curvatures, severity, suggested fixes |
| `BottleneckDetector::new(threshold)` | Detect edges below curvature threshold |
| `.detect(graph)` | Find all bottlenecks |
| `.severity_ranking(graph)` | Rank bottlenecks by severity |

### `fleet` — Fleet Topology Design

| Type | Description |
|------|-------------|
| `FleetDesignGoals` | Design parameters: n_agents, target curvature, max degree, BM enforcement |
| `FleetDesignResult` | Designed graph with curvature statistics and consensus estimates |
| `FleetDesigner` | End-to-end fleet topology optimizer |

## How It Works

### 1. Model Agents as a Graph

Agents are nodes, communication/interaction channels are weighted edges. The graph captures who talks to whom.

### 2. Compute Discrete Ricci Curvature

**Ollivier-Ricci**: At each node `x`, define a probability measure `μₓ` over its neighbors (lazy random walk). For each edge `(x,y)`, compute:
```
κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
```

This requires solving an optimal transport problem (exact LP or Sinkhorn approximation).

**Forman-Ricci**: Purely combinatorial:
```
F(x,y) = 4 - deg(x) - deg(y)
```

### 3. Interpret Curvature

| Curvature | Meaning | Consensus |
|-----------|---------|-----------|
| κ > 0 | Neighbors overlap heavily | Fast (bounded by 1/κ) |
| κ = 0 | Random mixing | Moderate |
| κ < 0 | Neighbors diverge | Slow or impossible |

### 4. Apply Structural Theorems

- **Concentration**: κ ≥ κ₀ → variance of any Lipschitz function is O(1/κ₀)
- **Bonnet-Myers**: κ ≥ κ₀ → diameter ≤ 1/κ₀
- **Spectral gap**: λ₁ ≥ 2κ₀ (Cheeger-type inequality)

### 5. Improve Topology

Use curvature flow to:
- Close triangles around negative-curvature edges
- Rewire edges to increase average curvature
- Eliminate bottlenecks

## The Math

### Ollivier-Ricci Curvature

For a graph G = (V, E) with a metric d and probability measures {μₓ} on V:

```
κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
```

where `W₁(μ, ν) = inf_{π} Σ d(x,y) π(x,y)` is the optimal transport cost.

**Interpretation**:
- κ(x,y) = 1: identical neighborhoods (e.g., complete graph)
- κ(x,y) = 0: independent neighborhoods (e.g., tree-like)
- κ(x,y) < 0: diverging neighborhoods (e.g., a bridge)

### Forman-Ricci Curvature

For an edge e = (u,v):
```
F(e) = wₑ( wᵤ/wₑ + wᵥ/wₑ ) - deg(u) - deg(v)
```

Unweighted: `F(u,v) = 4 - deg(u) - deg(v)`

- Complete graph Kₙ: F = 4 - 2(n-1) (very negative for large n)
- Cycle Cₙ: F = 4 - 4 = 0
- Path: F = 4 - 1 - 2 = 1 for endpoints, 4 - 2 - 2 = 0 for interior

### Bonnet-Myers Theorem (Discrete)

**Theorem**: If G is a connected graph with κ(x,y) ≥ κ₀ > 0 for all edges, then:
```
diam(G) ≤ 1/κ₀
```

This is the graph analog of the classical Bonnet-Myers theorem: positive Ricci curvature bounds the diameter of a manifold.

### Concentration of Measure

**Theorem**: If κ ≥ κ₀ > 0, then for any 1-Lipschitz function f: V → ℝ:
```
Var(f) ≤ |V| / (4κ₀)
```

This bounds how much any agent's opinion can deviate from the fleet average.

### Curvature Flow

Given a graph G, iteratively modify edges to increase curvature:
1. Find the edge with most negative curvature
2. Either add triangles around it (triangle closing) or rewire it
3. Recompute curvature
4. Repeat until convergence or budget exhausted

## Project Structure

```
src/
├── lib.rs              # Crate root, module declarations
├── graph.rs            # Agent interaction graph: AgentGraph, adjacency, Laplacian
├── ollivier_ricci.rs   # Ollivier-Ricci curvature via optimal transport
├── forman_ricci.rs     # Forman-Ricci curvature (combinatorial)
├── belief_curvature.rs # Curvature of belief space via Fisher information
├── concentration.rs    # Curvature-concentration inequalities
├── bonnet_myers.rs     # Bonnet-Myers diameter bounds
├── curvature_flow.rs   # Topology evolution to increase curvature
├── bottleneck.rs       # Negative-curvature bottleneck detection
└── fleet.rs            # Fleet topology design and optimization
```

## License

MIT
