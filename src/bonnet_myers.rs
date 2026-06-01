//! Bonnet-Myers theorem for graphs.
//!
//! If κ ≥ κ₀ > 0 everywhere, then:
//! - Graph diameter ≤ 1/κ₀
//! - Graph is finite (bounded size)
//! - Fundamental group is finite

use crate::graph::AgentGraph;
use crate::ollivier_ricci::{MeasureStrategy, OllivierRicciCurvature, TransportSolver};
use serde::{Deserialize, Serialize};

/// Result of a Bonnet-Myers analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonnetMyersResult {
    /// Minimum curvature in the graph.
    pub min_curvature: f64,
    /// Bonnet-Myers diameter bound (1/κ if κ > 0, else ∞).
    pub diameter_bound: Option<usize>,
    /// Actual graph diameter.
    pub actual_diameter: usize,
    /// Whether the graph satisfies Bonnet-Myers conditions.
    pub satisfies_bonnet_myers: bool,
}

/// Bonnet-Myers theorem checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonnetMyers {
    pub orc: OllivierRicciCurvature,
}

impl Default for BonnetMyers {
    fn default() -> Self {
        Self {
            orc: OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
        }
    }
}

impl BonnetMyers {
    pub fn new(orc: OllivierRicciCurvature) -> Self {
        Self { orc }
    }

    /// Compute the Bonnet-Myers diameter bound.
    ///
    /// If all edges have curvature ≥ κ > 0, then diam(G) ≤ ⌊1/κ⌋.
    /// Returns None if curvature is not uniformly positive.
    pub fn diameter_bound(&self, graph: &AgentGraph) -> Option<usize> {
        let kappa_min = self.orc.min_curvature(graph);
        if kappa_min > 1e-10 {
            let bound = (1.0 / kappa_min).floor() as usize;
            Some(bound.max(1))
        } else {
            None
        }
    }

    /// Full Bonnet-Myers analysis.
    pub fn analyze(&self, graph: &AgentGraph) -> BonnetMyersResult {
        let kappa_min = self.orc.min_curvature(graph);
        let diam_bound = self.diameter_bound(graph);
        let actual_diam = graph.diameter();
        let satisfies = kappa_min > 1e-10
            && diam_bound.map_or(false, |b| actual_diam <= b);

        BonnetMyersResult {
            min_curvature: kappa_min,
            diameter_bound: diam_bound,
            actual_diameter: actual_diam,
            satisfies_bonnet_myers: satisfies,
        }
    }

    /// Check if the graph satisfies the Bonnet-Myers conditions.
    pub fn check(&self, graph: &AgentGraph) -> bool {
        self.analyze(graph).satisfies_bonnet_myers
    }

    /// Maximum fleet size compatible with Bonnet-Myers.
    ///
    /// If curvature ≥ κ, max agents ≤ sum of (1/κ)-neighborhood sizes.
    /// Simplified: max agents grows exponentially with 1/κ.
    pub fn max_fleet_size(&self, kappa: f64, max_degree: usize) -> usize {
        if kappa <= 0.0 {
            return usize::MAX;
        }
        let radius = (1.0 / kappa).floor() as usize;
        // Upper bound: (max_degree)^radius
        if radius == 0 {
            return 1;
        }
        max_degree.pow(radius as u32)
    }

    /// Verify the diameter bound holds for the given graph.
    pub fn verify_diameter_bound(&self, graph: &AgentGraph) -> bool {
        match self.diameter_bound(graph) {
            Some(bound) => graph.diameter() <= bound,
            None => false,
        }
    }

    /// Check all edges have curvature ≥ threshold.
    pub fn all_curvatures_above(&self, graph: &AgentGraph, threshold: f64) -> bool {
        self.orc
            .all_curvatures(graph)
            .into_iter()
            .all(|(_, k)| k >= threshold - 1e-10)
    }

    /// Compute the minimum curvature required for a given diameter.
    pub fn required_curvature_for_diameter(&self, target_diameter: usize) -> f64 {
        if target_diameter == 0 {
            return f64::INFINITY;
        }
        1.0 / target_diameter as f64
    }

    /// Compactness check: if curvature ≥ κ, the graph has bounded covering number.
    pub fn covering_number_bound(&self, graph: &AgentGraph) -> Option<usize> {
        let bound = self.diameter_bound(graph)?;
        // In a graph with diameter ≤ D, covering number ≤ |V| but
        // we can bound it by the max-degree ball
        let n = graph.num_agents();
        let max_deg = (0..n).map(|i| graph.degree(i)).max().unwrap_or(0);
        if max_deg == 0 {
            return Some(n);
        }
        // Number of D-radius balls to cover: ≤ n / (max_deg + 1)
        Some((n + max_deg) / (max_deg + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bonnet_myers_complete_graph() {
        let g = AgentGraph::complete(5);
        let bm = BonnetMyers::default();
        let result = bm.analyze(&g);
        assert!(result.min_curvature > 0.0);
        assert!(result.diameter_bound.is_some());
        assert_eq!(result.actual_diameter, 1);
    }

    #[test]
    fn test_bonnet_myers_path_graph() {
        let g = AgentGraph::path(10);
        let bm = BonnetMyers::default();
        let result = bm.analyze(&g);
        assert!(result.min_curvature < 0.0);
        assert!(result.diameter_bound.is_none());
    }

    #[test]
    fn test_bonnet_myers_cycle() {
        let g = AgentGraph::cycle(6);
        let bm = BonnetMyers::default();
        let result = bm.analyze(&g);
        // Cycle should have positive curvature
        assert!(result.min_curvature > 0.0, "cycle curvature should be positive, got {}", result.min_curvature);
        assert!(result.diameter_bound.is_some());
    }

    #[test]
    fn test_diameter_bound_complete() {
        let g = AgentGraph::complete(4);
        let bm = BonnetMyers::default();
        let bound = bm.diameter_bound(&g).unwrap();
        assert!(bound >= 1); // diameter is 1, bound should be ≥ 1
    }

    #[test]
    fn test_verify_diameter_bound() {
        let g = AgentGraph::complete(4);
        let bm = BonnetMyers::default();
        assert!(bm.verify_diameter_bound(&g));
    }

    #[test]
    fn test_all_curvatures_above() {
        let g = AgentGraph::complete(4);
        let bm = BonnetMyers::default();
        assert!(bm.all_curvatures_above(&g, 0.0));
    }

    #[test]
    fn test_required_curvature() {
        let bm = BonnetMyers::default();
        let k = bm.required_curvature_for_diameter(5);
        assert_relative_eq!(k, 0.2);
    }

    #[test]
    fn test_max_fleet_size() {
        let bm = BonnetMyers::default();
        let max = bm.max_fleet_size(0.5, 4);
        assert!(max > 0);
        assert!(max < 1000);
    }

    #[test]
    fn test_max_fleet_size_zero_curvature() {
        let bm = BonnetMyers::default();
        assert_eq!(bm.max_fleet_size(0.0, 4), usize::MAX);
    }

    #[test]
    fn test_covering_number_bound() {
        let g = AgentGraph::complete(4);
        let bm = BonnetMyers::default();
        let bound = bm.covering_number_bound(&g);
        assert!(bound.is_some());
        assert!(bound.unwrap() <= 4);
    }

    #[test]
    fn test_star_graph_negative_curvature() {
        let g = AgentGraph::star(6);
        let bm = BonnetMyers::default();
        let result = bm.analyze(&g);
        // Star has negative curvature edges → no Bonnet-Myers
        assert!(!result.satisfies_bonnet_myers);
    }

    #[test]
    fn test_serialization() {
        let bm = BonnetMyers::default();
        let json = serde_json::to_string(&bm).unwrap();
        let bm2: BonnetMyers = serde_json::from_str(&json).unwrap();
        let g = AgentGraph::complete(3);
        let r1 = bm.analyze(&g);
        let r2 = bm2.analyze(&g);
        assert_relative_eq!(r1.min_curvature, r2.min_curvature);
    }
}
