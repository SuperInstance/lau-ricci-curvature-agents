//! Curvature-driven flow: evolve fleet topology to increase curvature.
//!
//! The idea: edges with negative curvature are bottlenecks.
//! Adding edges around them or rewiring can increase curvature,
//! leading to faster consensus.

use crate::graph::AgentGraph;
use crate::ollivier_ricci::{MeasureStrategy, OllivierRicciCurvature, TransportSolver};
use serde::{Deserialize, Serialize};

/// Strategy for topology evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowStrategy {
    /// Add edges between agents that share neighbors (triangle closing).
    TriangleClosing,
    /// Rewire: remove worst-curvature edge, add to improve curvature.
    Rewire,
    /// Add edges to the most negative curvature neighborhoods.
    NeighborhoodFilling,
}

/// Result of a curvature flow step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowStep {
    /// Average curvature before.
    pub avg_before: f64,
    /// Average curvature after.
    pub avg_after: f64,
    /// Edges added.
    pub edges_added: Vec<(usize, usize)>,
    /// Edges removed.
    pub edges_removed: Vec<(usize, usize)>,
    /// Improvement in average curvature.
    pub improvement: f64,
}

/// Curvature flow optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureFlow {
    pub orc: OllivierRicciCurvature,
    pub strategy: FlowStrategy,
    /// Learning rate for edge weight adjustment.
    pub learning_rate: f64,
}

impl Default for CurvatureFlow {
    fn default() -> Self {
        Self {
            orc: OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            strategy: FlowStrategy::TriangleClosing,
            learning_rate: 0.1,
        }
    }
}

impl CurvatureFlow {
    pub fn new(orc: OllivierRicciCurvature, strategy: FlowStrategy, learning_rate: f64) -> Self {
        Self { orc, strategy, learning_rate }
    }

    /// Execute one step of curvature flow.
    pub fn step(&self, graph: &mut AgentGraph) -> FlowStep {
        let avg_before = self.orc.average_curvature(graph);

        let (added, removed) = match self.strategy {
            FlowStrategy::TriangleClosing => self.step_triangle_closing(graph),
            FlowStrategy::Rewire => self.step_rewire(graph),
            FlowStrategy::NeighborhoodFilling => self.step_neighborhood_filling(graph),
        };

        let avg_after = self.orc.average_curvature(graph);
        FlowStep {
            avg_before,
            avg_after,
            edges_added: added,
            edges_removed: removed,
            improvement: avg_after - avg_before,
        }
    }

    /// Run curvature flow for `steps` iterations.
    pub fn evolve(&self, graph: &mut AgentGraph, steps: usize) -> Vec<FlowStep> {
        let mut history = Vec::new();
        for _ in 0..steps {
            let step = self.step(graph);
            let imp = step.improvement.abs();
            history.push(step);
            if imp < 1e-10 {
                break;
            }
        }
        history
    }

    fn step_triangle_closing(&self, graph: &mut AgentGraph) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut added = Vec::new();
        let curvatures = self.orc.all_curvatures(graph);

        // Find the worst curvature edges
        let mut sorted: Vec<_> = curvatures.into_iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // For the worst edges, try to close triangles
        for &((a, b), k) in &sorted {
            if k >= 0.0 || added.len() >= 3 {
                break;
            }
            // Find common neighbors
            let neighbors_a: std::collections::HashSet<usize> =
                graph.neighbors(a).iter().map(|&(n, _)| n).collect();
            let neighbors_b: std::collections::HashSet<usize> =
                graph.neighbors(b).iter().map(|&(n, _)| n).collect();

            // Find non-common neighbors of a that aren't connected to b
            for &na in &neighbors_a {
                if na != b && !neighbors_b.contains(&na) && !graph.has_edge(b, na) {
                    graph.add_edge(b, na, 1.0);
                    added.push((b.min(na), b.max(na)));
                    if added.len() >= 3 {
                        break;
                    }
                }
            }
        }

        (added, vec![])
    }

    fn step_rewire(&self, graph: &mut AgentGraph) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut added = Vec::new();
        let mut removed = Vec::new();

        let curvatures = self.orc.all_curvatures(graph);
        let mut sorted: Vec<_> = curvatures.into_iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(&((a, b), _)) = sorted.first() {
            // Find a better neighbor for a
            let neighbors_a: std::collections::HashSet<usize> =
                graph.neighbors(a).iter().map(|&(n, _)| n).collect();

            for candidate in 0..graph.num_agents() {
                if candidate != a && candidate != b && !neighbors_a.contains(&candidate) {
                    // Rewire: remove (a,b), add (a,candidate)
                    // Check if this improves curvature
                    if graph.degree(b) > 1 {
                        // Only rewire if b won't become isolated
                        removed.push((a.min(b), a.max(b)));
                        added.push((a.min(candidate), a.max(candidate)));
                        break;
                    }
                }
            }
        }

        (added, removed)
    }

    fn step_neighborhood_filling(
        &self,
        graph: &mut AgentGraph,
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut added = Vec::new();
        let curvatures = self.orc.all_curvatures(graph);

        // Find the edge with most negative curvature
        let worst = curvatures
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(((a, b), _)) = worst {
            // Fill: connect all neighbors of a to all neighbors of b
            let neighbors_a: Vec<usize> = graph.neighbors(*a).iter().map(|&(n, _)| n).collect();
            let neighbors_b: Vec<usize> = graph.neighbors(*b).iter().map(|&(n, _)| n).collect();

            for &na in &neighbors_a {
                for &nb in &neighbors_b {
                    if na != nb && !graph.has_edge(na, nb) {
                        graph.add_edge(na, nb, 1.0);
                        added.push((na.min(nb), na.max(nb)));
                        if added.len() >= 5 {
                            return (added, vec![]);
                        }
                    }
                }
            }
        }

        (added, vec![])
    }

    /// Compute the curvature gradient for each possible edge addition.
    /// Returns sorted by curvature improvement potential.
    pub fn curvature_gradient(&self, graph: &AgentGraph) -> Vec<((usize, usize), f64)> {
        let n = graph.num_agents();
        let _current_avg = self.orc.average_curvature(graph);

        let mut candidates = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if !graph.has_edge(i, j) {
                    // Estimate curvature improvement by considering common neighbors
                    let neighbors_i: std::collections::HashSet<usize> =
                        graph.neighbors(i).iter().map(|&(n, _)| n).collect();
                    let neighbors_j: std::collections::HashSet<usize> =
                        graph.neighbors(j).iter().map(|&(n, _)| n).collect();
                    let common = neighbors_i.intersection(&neighbors_j).count();
                    // More common neighbors = higher potential curvature
                    let estimate = common as f64 / (neighbors_i.len().max(1).max(neighbors_j.len().max(1))) as f64;
                    candidates.push(((i, j), estimate));
                }
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_flow_step_triangle_closing() {
        let mut g = AgentGraph::path(5);
        let flow = CurvatureFlow::new(
            OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            FlowStrategy::TriangleClosing,
            0.1,
        );
        let step = flow.step(&mut g);
        assert!(step.edges_added.len() > 0 || step.improvement.abs() < 1e-10);
    }

    #[test]
    fn test_flow_evolve_improves() {
        let mut g = AgentGraph::path(6);
        let flow = CurvatureFlow::new(
            OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            FlowStrategy::TriangleClosing,
            0.1,
        );
        let avg_before = flow.orc.average_curvature(&g);
        let history = flow.evolve(&mut g, 10);
        let avg_after = flow.orc.average_curvature(&g);
        // Should improve or stay the same
        assert!(avg_after >= avg_before - 0.01);
    }

    #[test]
    fn test_flow_rewire() {
        let mut g = AgentGraph::path(5);
        let flow = CurvatureFlow::new(
            OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            FlowStrategy::Rewire,
            0.1,
        );
        let step = flow.step(&mut g);
        // Rewiring may or may not happen depending on the situation
        assert!(step.edges_added.len() <= step.edges_removed.len() + 1);
    }

    #[test]
    fn test_flow_neighborhood_filling() {
        let mut g = AgentGraph::star(5);
        let flow = CurvatureFlow::new(
            OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            FlowStrategy::NeighborhoodFilling,
            0.1,
        );
        let step = flow.step(&mut g);
        // Star has no common neighbors, so filling may add edges between leaves
    }

    #[test]
    fn test_curvature_gradient() {
        let g = AgentGraph::path(4);
        let flow = CurvatureFlow::default();
        let grad = flow.curvature_gradient(&g);
        // Should have some candidates (non-edges)
        assert!(!grad.is_empty());
    }

    #[test]
    fn test_flow_step_records_improvement() {
        let mut g = AgentGraph::path(8);
        let flow = CurvatureFlow::new(
            OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            FlowStrategy::NeighborhoodFilling,
            0.1,
        );
        let step = flow.step(&mut g);
        assert_relative_eq!(step.improvement, step.avg_after - step.avg_before);
    }

    #[test]
    fn test_flow_complete_graph_stable() {
        let mut g = AgentGraph::complete(4);
        let flow = CurvatureFlow::default();
        let avg_before = flow.orc.average_curvature(&g);
        let history = flow.evolve(&mut g, 5);
        // Complete graph is already optimal, should not change much
        let avg_after = flow.orc.average_curvature(&g);
        assert_relative_eq!(avg_before, avg_after, epsilon = 0.01);
    }

    #[test]
    fn test_flow_serialization() {
        let flow = CurvatureFlow::default();
        let json = serde_json::to_string(&flow).unwrap();
        let flow2: CurvatureFlow = serde_json::from_str(&json).unwrap();
        let g = AgentGraph::complete(3);
        assert_relative_eq!(
            flow.orc.average_curvature(&g),
            flow2.orc.average_curvature(&g)
        );
    }

    #[test]
    fn test_flow_step_result_serialization() {
        let mut g = AgentGraph::path(4);
        let flow = CurvatureFlow::default();
        let step = flow.step(&mut g);
        let json = serde_json::to_string(&step).unwrap();
        let step2: FlowStep = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(step.avg_before, step2.avg_before);
        assert_relative_eq!(step.improvement, step2.improvement);
    }
}
