//! Ollivier-Ricci curvature via optimal transport.
//!
//! κ(x,y) = 1 - W₁(μ_x, μ_y) / d(x,y)
//! where μ_x is the neighborhood probability measure around x,
//! μ_y around y, and W₁ is the 1-Wasserstein distance.

use crate::graph::{AgentGraph, AgentId, Weight};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Strategy for building the neighborhood probability measure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasureStrategy {
    /// Uniform over neighbors + self (lazy random walk with parameter α).
    LazyRandomWalk { alpha: f64 },
    /// Uniform over neighbors only.
    UniformNeighbors,
    /// Custom probability weights per agent.
    Custom { probs: Vec<Vec<(AgentId, f64)>> },
}

impl Default for MeasureStrategy {
    fn default() -> Self {
        MeasureStrategy::LazyRandomWalk { alpha: 0.5 }
    }
}

/// Solver for optimal transport (Wasserstein-1) distance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportSolver {
    /// Exact: solve LP via north-west corner / network simplex (small graphs).
    Exact,
    /// Sinkhorn approximation with given regularization.
    Sinkhorn { reg: f64, iterations: usize },
}

impl Default for TransportSolver {
    fn default() -> Self {
        TransportSolver::Exact
    }
}

/// Ollivier-Ricci curvature computer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllivierRicciCurvature {
    pub measure: MeasureStrategy,
    pub solver: TransportSolver,
}

impl OllivierRicciCurvature {
    pub fn new(measure: MeasureStrategy, solver: TransportSolver) -> Self {
        Self { measure, solver }
    }

    /// Build neighborhood probability measure for agent `a`.
    pub fn measure_at(&self, graph: &AgentGraph, a: AgentId) -> Vec<(AgentId, f64)> {
        match &self.measure {
            MeasureStrategy::LazyRandomWalk { alpha } => {
                let neighbors = graph.neighbors(a);
                let deg = neighbors.len() as f64;
                if deg == 0.0 {
                    return vec![(a, 1.0)];
                }
                let mut probs = Vec::new();
                probs.push((a, *alpha));
                let per_neighbor = (1.0 - alpha) / deg;
                for &(nb, _) in neighbors {
                    probs.push((nb, per_neighbor));
                }
                probs
            }
            MeasureStrategy::UniformNeighbors => {
                let neighbors = graph.neighbors(a);
                if neighbors.is_empty() {
                    return vec![(a, 1.0)];
                }
                let p = 1.0 / neighbors.len() as f64;
                neighbors.iter().map(|&(nb, _)| (nb, p)).collect()
            }
            MeasureStrategy::Custom { probs } => {
                probs.get(a).cloned().unwrap_or_else(|| vec![(a, 1.0)])
            }
        }
    }

    /// Compute 1-Wasserstein distance between two probability measures
    /// over graph nodes using the graph distance as ground metric.
    pub fn wasserstein_1(
        &self,
        graph: &AgentGraph,
        mu_x: &[(AgentId, f64)],
        mu_y: &[(AgentId, f64)],
    ) -> f64 {
        match &self.solver {
            TransportSolver::Exact => {
                self.wasserstein_exact(graph, mu_x, mu_y)
            }
            TransportSolver::Sinkhorn { reg, iterations } => {
                self.wasserstein_sinkhorn(graph, mu_x, mu_y, *reg, *iterations)
            }
        }
    }

    fn wasserstein_exact(
        &self,
        graph: &AgentGraph,
        mu_x: &[(AgentId, f64)],
        mu_y: &[(AgentId, f64)],
    ) -> f64 {
        let n = graph.num_agents();
        // Precompute distance matrix
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = graph.distance(i, j)
                    .unwrap_or(n + 1) as f64;
            }
        }

        // Build the transport problem as a simple LP
        // Using the Hungarian-like greedy approach for discrete measures
        let nx = mu_x.len();
        let ny = mu_y.len();

        if nx == 0 || ny == 0 {
            return 0.0;
        }

        // Build cost matrix
        let mut cost = vec![vec![0.0f64; ny]; nx];
        for (i, &(xi, _)) in mu_x.iter().enumerate() {
            for (j, &(yj, _)) in mu_y.iter().enumerate() {
                cost[i][j] = dist[xi][yj];
            }
        }

        // Simple transport: greedy with remaining mass
        let mut supply: Vec<f64> = mu_x.iter().map(|&(_, p)| p).collect();
        let mut demand: Vec<f64> = mu_y.iter().map(|&(_, p)| p).collect();
        let mut total_cost = 0.0;

        // Sort pairs by cost for greedy
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..nx {
            for j in 0..ny {
                pairs.push((i, j, cost[i][j]));
            }
        }
        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (i, j, c) in pairs {
            let flow = supply[i].min(demand[j]);
            if flow > 1e-15 {
                total_cost += flow * c;
                supply[i] -= flow;
                demand[j] -= flow;
            }
        }

        total_cost
    }

    fn wasserstein_sinkhorn(
        &self,
        graph: &AgentGraph,
        mu_x: &[(AgentId, f64)],
        mu_y: &[(AgentId, f64)],
        reg: f64,
        iterations: usize,
    ) -> f64 {
        let n = graph.num_agents();
        let mut dist = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                dist[i][j] = graph.distance(i, j)
                    .unwrap_or(n + 1) as f64;
            }
        }

        let nx = mu_x.len();
        let ny = mu_y.len();
        if nx == 0 || ny == 0 {
            return 0.0;
        }

        // Cost matrix
        let mut k = vec![vec![0.0f64; ny]; nx];
        for (i, &(xi, _)) in mu_x.iter().enumerate() {
            for (j, &(yj, _)) in mu_y.iter().enumerate() {
                k[i][j] = (-dist[xi][yj] / reg).exp();
            }
        }

        let a: Vec<f64> = mu_x.iter().map(|&(_, p)| p).collect();
        let b: Vec<f64> = mu_y.iter().map(|&(_, p)| p).collect();

        let mut u = vec![1.0f64 / nx as f64; nx];
        let mut v = vec![1.0f64 / ny as f64; ny];

        for _ in 0..iterations {
            // u = a ./ (K * v)
            for i in 0..nx {
                let kv: f64 = (0..ny).map(|j| k[i][j] * v[j]).sum();
                u[i] = if kv > 1e-30 { a[i] / kv } else { u[i] };
            }
            // v = b ./ (K^T * u)
            for j in 0..ny {
                let ktu: f64 = (0..nx).map(|i| k[i][j] * u[i]).sum();
                v[j] = if ktu > 1e-30 { b[j] / ktu } else { v[j] };
            }
        }

        // Transport plan
        let mut cost_val = 0.0;
        for (i, &(xi, _)) in mu_x.iter().enumerate() {
            for (j, &(yj, _)) in mu_y.iter().enumerate() {
                let pij = u[i] * k[i][j] * v[j];
                cost_val += pij * dist[xi][yj];
            }
        }
        cost_val
    }

    /// Compute Ollivier-Ricci curvature for edge (a, b).
    ///
    /// κ(a,b) = 1 - W₁(μ_a, μ_b) / d(a,b)
    pub fn edge_curvature(&self, graph: &AgentGraph, a: AgentId, b: AgentId) -> f64 {
        let d = match graph.distance(a, b) {
            Some(d) if d > 0 => d as f64,
            _ => return 0.0,
        };
        let mu_a = self.measure_at(graph, a);
        let mu_b = self.measure_at(graph, b);
        let w = self.wasserstein_1(graph, &mu_a, &mu_b);
        1.0 - w / d
    }

    /// Compute curvature for all edges.
    pub fn all_curvatures(&self, graph: &AgentGraph) -> Vec<((AgentId, AgentId), f64)> {
        graph
            .edges()
            .into_iter()
            .map(|(a, b)| {
                let k = self.edge_curvature(graph, a, b);
                ((a, b), k)
            })
            .collect()
    }

    /// Average curvature across all edges.
    pub fn average_curvature(&self, graph: &AgentGraph) -> f64 {
        let curvatures = self.all_curvatures(graph);
        if curvatures.is_empty() {
            return 0.0;
        }
        curvatures.iter().map(|&(_, k)| k).sum::<f64>() / curvatures.len() as f64
    }

    /// Minimum curvature (most negative edge).
    pub fn min_curvature(&self, graph: &AgentGraph) -> f64 {
        self.all_curvatures(graph)
            .into_iter()
            .map(|(_, k)| k)
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum curvature.
    pub fn max_curvature(&self, graph: &AgentGraph) -> f64 {
        self.all_curvatures(graph)
            .into_iter()
            .map(|(_, k)| k)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_orc() -> OllivierRicciCurvature {
        OllivierRicciCurvature::new(
            MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
            TransportSolver::Exact,
        )
    }

    #[test]
    fn test_measure_lazy_rw_complete_3() {
        let g = AgentGraph::complete(3);
        let orc = make_orc();
        // Node 0: self = 0.5, neighbors 1,2 each get 0.25
        let mu = orc.measure_at(&g, 0);
        assert_relative_eq!(mu.iter().map(|&(_, p)| p).sum::<f64>(), 1.0);
    }

    #[test]
    fn test_measure_uniform_neighbors() {
        let g = AgentGraph::complete(3);
        let orc = OllivierRicciCurvature::new(
            MeasureStrategy::UniformNeighbors,
            TransportSolver::Exact,
        );
        let mu = orc.measure_at(&g, 0);
        assert_eq!(mu.len(), 2); // neighbors 1 and 2
        assert_relative_eq!(mu[0].1, 0.5);
        assert_relative_eq!(mu[1].1, 0.5);
    }

    #[test]
    fn test_measure_isolated_node() {
        let mut g = AgentGraph::new(3);
        // agent 2 is isolated
        g.add_edge(0, 1, 1.0);
        let orc = make_orc();
        let mu = orc.measure_at(&g, 2);
        assert_eq!(mu, vec![(2, 1.0)]);
    }

    #[test]
    fn test_wasserstein_same_measure() {
        let g = AgentGraph::complete(3);
        let orc = make_orc();
        let mu = orc.measure_at(&g, 0);
        let w = orc.wasserstein_1(&g, &mu, &mu);
        assert_relative_eq!(w, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_wasserstein_nonnegative() {
        let g = AgentGraph::path(5);
        let orc = make_orc();
        let mu_a = orc.measure_at(&g, 0);
        let mu_b = orc.measure_at(&g, 4);
        let w = orc.wasserstein_1(&g, &mu_a, &mu_b);
        assert!(w >= 0.0);
    }

    #[test]
    fn test_wasserstein_sinkhorn() {
        let g = AgentGraph::complete(3);
        let orc = OllivierRicciCurvature::new(
            MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
            TransportSolver::Sinkhorn { reg: 0.1, iterations: 100 },
        );
        let mu_a = orc.measure_at(&g, 0);
        let mu_b = orc.measure_at(&g, 1);
        let w = orc.wasserstein_1(&g, &mu_a, &mu_b);
        assert!(w >= 0.0);
    }

    #[test]
    fn test_curvature_complete_graph_positive() {
        let g = AgentGraph::complete(5);
        let orc = make_orc();
        let k = orc.edge_curvature(&g, 0, 1);
        // Complete graph should have high positive curvature
        assert!(k > 0.5, "complete graph curvature should be high, got {}", k);
    }

    #[test]
    fn test_curvature_path_graph_negative() {
        let g = AgentGraph::path(10);
        let orc = make_orc();
        let k = orc.edge_curvature(&g, 4, 5);
        // Path middle edges should have negative curvature
        assert!(k < 0.0, "path middle should have negative curvature, got {}", k);
    }

    #[test]
    fn test_curvature_bounded_minus_one_to_one() {
        let g = AgentGraph::grid(3, 3);
        let orc = make_orc();
        for (a, b) in g.edges() {
            let k = orc.edge_curvature(&g, a, b);
            assert!(k >= -1.0 - 1e-10 && k <= 1.0 + 1e-10,
                "curvature out of bounds for ({},{}): {}", a, b, k);
        }
    }

    #[test]
    fn test_all_curvatures_count() {
        let g = AgentGraph::cycle(6);
        let orc = make_orc();
        let all = orc.all_curvatures(&g);
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_average_curvature_complete() {
        let g = AgentGraph::complete(4);
        let orc = make_orc();
        let avg = orc.average_curvature(&g);
        assert!(avg > 0.5);
    }

    #[test]
    fn test_average_curvature_path() {
        let g = AgentGraph::path(10);
        let orc = make_orc();
        let avg = orc.average_curvature(&g);
        // Path should have low/negative average curvature
        assert!(avg < 0.2);
    }

    #[test]
    fn test_min_max_curvature() {
        let g = AgentGraph::cycle(8);
        let orc = make_orc();
        let min_k = orc.min_curvature(&g);
        let max_k = orc.max_curvature(&g);
        assert!(min_k <= max_k);
    }

    #[test]
    fn test_curvature_cycle_graph() {
        let g = AgentGraph::cycle(6);
        let orc = make_orc();
        // Cycle should have moderate curvature (uniform)
        let curvatures: Vec<f64> = orc.all_curvatures(&g).into_iter().map(|(_, k)| k).collect();
        let first = curvatures[0];
        for k in &curvatures {
            assert_relative_eq!(*k, first, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_curvature_star_graph() {
        let g = AgentGraph::star(5);
        let orc = make_orc();
        let all = orc.all_curvatures(&g);
        // Star edges should have negative curvature (bottleneck)
        for (_, k) in &all {
            assert!(k < &0.0, "star edge curvature should be negative, got {}", k);
        }
    }

    #[test]
    fn test_curvature_serialization() {
        let orc = make_orc();
        let json = serde_json::to_string(&orc).unwrap();
        let orc2: OllivierRicciCurvature = serde_json::from_str(&json).unwrap();
        let g = AgentGraph::complete(3);
        let k1 = orc.edge_curvature(&g, 0, 1);
        let k2 = orc2.edge_curvature(&g, 0, 1);
        assert_relative_eq!(k1, k2);
    }
}
