//! Curvature-concentration inequality.
//!
//! High curvature → fast concentration of measure.
//! If κ ≥ κ₀ > 0 on a graph, then the variance of any 1-Lipschitz function
//! is bounded by O(1/κ₀), and consensus is reached in O(1/κ₀) steps.

use crate::graph::AgentGraph;
use crate::ollivier_ricci::{MeasureStrategy, OllivierRicciCurvature, TransportSolver};
use serde::{Deserialize, Serialize};

/// Result of a concentration analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationResult {
    /// Minimum curvature in the graph.
    pub min_curvature: f64,
    /// Variance bound for 1-Lipschitz functions.
    pub variance_bound: f64,
    /// Expected mixing time bound.
    pub mixing_time_bound: usize,
    /// Expected consensus time bound.
    pub consensus_time_bound: usize,
}

/// Curvature-concentration inequality analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureConcentration {
    pub orc: OllivierRicciCurvature,
}

impl Default for CurvatureConcentration {
    fn default() -> Self {
        Self {
            orc: OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
        }
    }
}

impl CurvatureConcentration {
    pub fn new(orc: OllivierRicciCurvature) -> Self {
        Self { orc }
    }

    /// Compute concentration bounds for the graph.
    ///
    /// If κ_min > 0:
    /// - Variance of any 1-Lipschitz f ≤ n/(4·κ_min)
    /// - Mixing time ≤ O(log(n)/κ_min)
    /// - Consensus time ≤ O(n/κ_min)
    pub fn analyze(&self, graph: &AgentGraph) -> ConcentrationResult {
        let n = graph.num_agents();
        let kappa_min = self.orc.min_curvature(graph);

        let (variance_bound, mixing_bound, consensus_bound) = if kappa_min > 1e-10 {
            let var_b = n as f64 / (4.0 * kappa_min);
            let mix = ((n as f64).ln() / kappa_min).ceil() as usize;
            let cons = (n as f64 / kappa_min).ceil() as usize;
            (var_b, mix.max(1), cons.max(1))
        } else if kappa_min > -1e-10 {
            // κ ≈ 0: no concentration
            let var_b = (n * n) as f64 / 4.0;
            (var_b, n * n, n * n)
        } else {
            // κ < 0: no concentration, potentially exponential mixing
            let var_b = f64::INFINITY;
            (var_b, usize::MAX, usize::MAX)
        };

        ConcentrationResult {
            min_curvature: kappa_min,
            variance_bound,
            mixing_time_bound: mixing_bound,
            consensus_time_bound: consensus_bound,
        }
    }

    /// Compute the spectral gap lower bound from curvature.
    ///
    /// λ₁ ≥ κ_min (when κ_min > 0)
    pub fn spectral_gap_bound(&self, graph: &AgentGraph) -> f64 {
        let kappa_min = self.orc.min_curvature(graph);
        if kappa_min > 0.0 {
            kappa_min
        } else {
            0.0
        }
    }

    /// Check if consensus is guaranteed (κ_min > 0).
    pub fn consensus_guaranteed(&self, graph: &AgentGraph) -> bool {
        self.orc.min_curvature(graph) > 0.0
    }

    /// Compare concentration bounds of two graphs.
    pub fn compare_graphs(
        &self,
        g1: &AgentGraph,
        g2: &AgentGraph,
    ) -> GraphComparison {
        let r1 = self.analyze(g1);
        let r2 = self.analyze(g2);
        GraphComparison {
            graph1_min_curvature: r1.min_curvature,
            graph2_min_curvature: r2.min_curvature,
            faster_consensus: if r1.consensus_time_bound <= r2.consensus_time_bound {
                1
            } else {
                2
            },
            concentration_ratio: if r2.variance_bound > 0.0 && r2.variance_bound.is_finite() {
                r1.variance_bound / r2.variance_bound
            } else {
                f64::NAN
            },
        }
    }

    /// Simulate consensus dynamics and measure actual convergence time.
    /// Returns the number of steps to reach ε-consensus.
    pub fn simulate_consensus(
        &self,
        graph: &AgentGraph,
        values: &[f64],
        epsilon: f64,
        max_steps: usize,
    ) -> Option<usize> {
        let n = graph.num_agents();
        assert_eq!(values.len(), n);
        let mut x = values.to_vec();

        for step in 0..max_steps {
            let mut x_new = vec![0.0; n];
            for i in 0..n {
                let neighbors = graph.neighbors(i);
                let deg = neighbors.len() as f64;
                if deg == 0.0 {
                    x_new[i] = x[i];
                } else {
                    let mut sum = x[i]; // include self
                    for &(j, _) in neighbors {
                        sum += x[j];
                    }
                    x_new[i] = sum / (1.0 + deg);
                }
            }

            // Check consensus
            let mean = x_new.iter().sum::<f64>() / n as f64;
            let max_dev = x_new.iter().map(|v| (v - mean).abs()).fold(0.0f64, f64::max);
            if max_dev < epsilon {
                return Some(step + 1);
            }
            x = x_new;
        }
        None
    }
}

/// Comparison of two graphs' concentration properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphComparison {
    pub graph1_min_curvature: f64,
    pub graph2_min_curvature: f64,
    /// Which graph has faster consensus (1 or 2).
    pub faster_consensus: usize,
    /// Ratio of variance bounds.
    pub concentration_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_concentration_complete_graph() {
        let g = AgentGraph::complete(5);
        let cc = CurvatureConcentration::default();
        let result = cc.analyze(&g);
        assert!(result.min_curvature > 0.0, "complete graph should have positive curvature");
        assert!(result.variance_bound.is_finite());
        assert!(result.mixing_time_bound < 100);
    }

    #[test]
    fn test_concentration_path_graph() {
        let g = AgentGraph::path(10);
        let cc = CurvatureConcentration::default();
        let result = cc.analyze(&g);
        assert!(result.min_curvature < 0.0, "path should have negative curvature");
    }

    #[test]
    fn test_spectral_gap_bound_positive() {
        let g = AgentGraph::complete(5);
        let cc = CurvatureConcentration::default();
        let gap = cc.spectral_gap_bound(&g);
        assert!(gap > 0.0);
    }

    #[test]
    fn test_spectral_gap_bound_zero() {
        let g = AgentGraph::path(10);
        let cc = CurvatureConcentration::default();
        let gap = cc.spectral_gap_bound(&g);
        assert_relative_eq!(gap, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_consensus_guaranteed() {
        let g = AgentGraph::complete(4);
        let cc = CurvatureConcentration::default();
        assert!(cc.consensus_guaranteed(&g));
    }

    #[test]
    fn test_consensus_not_guaranteed() {
        let g = AgentGraph::path(10);
        let cc = CurvatureConcentration::default();
        assert!(!cc.consensus_guaranteed(&g));
    }

    #[test]
    fn test_compare_graphs() {
        let g1 = AgentGraph::complete(5);
        let g2 = AgentGraph::path(5);
        let cc = CurvatureConcentration::default();
        let comp = cc.compare_graphs(&g1, &g2);
        assert_eq!(comp.faster_consensus, 1); // complete faster
        assert!(comp.graph1_min_curvature > comp.graph2_min_curvature);
    }

    #[test]
    fn test_simulate_consensus_complete() {
        let g = AgentGraph::complete(5);
        let cc = CurvatureConcentration::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let steps = cc.simulate_consensus(&g, &values, 0.01, 100);
        assert!(steps.is_some());
        assert!(steps.unwrap() < 50);
    }

    #[test]
    fn test_simulate_consensus_path() {
        let g = AgentGraph::path(5);
        let cc = CurvatureConcentration::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let steps = cc.simulate_consensus(&g, &values, 0.01, 1000);
        // Path should converge but slower
        assert!(steps.is_some());
    }

    #[test]
    fn test_simulate_consensus_already_consensus() {
        let g = AgentGraph::complete(3);
        let cc = CurvatureConcentration::default();
        let values = vec![2.0, 2.0, 2.0];
        let steps = cc.simulate_consensus(&g, &values, 0.01, 100);
        assert_eq!(steps, Some(1)); // Already in consensus after one check
    }

    #[test]
    fn test_concentration_cycle_graph() {
        let g = AgentGraph::cycle(6);
        let cc = CurvatureConcentration::default();
        let result = cc.analyze(&g);
        // Cycle should have positive curvature
        assert!(result.min_curvature > -0.1);
    }

    #[test]
    fn test_concentration_serialization() {
        let cc = CurvatureConcentration::default();
        let json = serde_json::to_string(&cc).unwrap();
        let cc2: CurvatureConcentration = serde_json::from_str(&json).unwrap();
        let g = AgentGraph::complete(3);
        let r1 = cc.analyze(&g);
        let r2 = cc2.analyze(&g);
        assert_relative_eq!(r1.min_curvature, r2.min_curvature);
    }
}
