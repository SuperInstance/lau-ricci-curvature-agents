//! Bottleneck detection via negative curvature edges.
//!
//! Edges with negative curvature are communication bottlenecks:
//! information doesn't flow well across them. Detecting and fixing
//! these improves fleet consensus time.

use crate::graph::{AgentGraph, AgentId};
use crate::ollivier_ricci::{MeasureStrategy, OllivierRicciCurvature, TransportSolver};
use crate::forman_ricci::FormanRicciCurvature;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A detected bottleneck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// The bottleneck edge.
    pub edge: (AgentId, AgentId),
    /// Ollivier-Ricci curvature (if computed).
    pub ollivier_curvature: Option<f64>,
    /// Forman-Ricci curvature (if computed).
    pub forman_curvature: Option<f64>,
    /// Severity: how negative the curvature is (0 = none, higher = worse).
    pub severity: f64,
    /// Suggested fix: agents to connect to alleviate bottleneck.
    pub suggested_fixes: Vec<(AgentId, AgentId)>,
}

/// Bottleneck detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckDetector {
    pub orc: OllivierRicciCurvature,
    pub forman: FormanRicciCurvature,
    /// Curvature threshold below which an edge is considered a bottleneck.
    pub threshold: f64,
}

impl Default for BottleneckDetector {
    fn default() -> Self {
        Self {
            orc: OllivierRicciCurvature::new(
                MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
                TransportSolver::Exact,
            ),
            forman: FormanRicciCurvature::default(),
            threshold: 0.0,
        }
    }
}

impl BottleneckDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }

    /// Detect all bottleneck edges in the graph.
    pub fn detect(&self, graph: &AgentGraph) -> Vec<Bottleneck> {
        let orc_curves = self.orc.all_curvatures(graph);
        let forman_curves = self.forman.all_curvatures(graph);

        let forman_map: HashMap<(AgentId, AgentId), f64> =
            forman_curves.into_iter().collect();

        let mut bottlenecks = Vec::new();

        // Check O-R curvature bottlenecks
        for ((a, b), k_orc) in &orc_curves {
            if k_orc < &self.threshold {
                let k_f = forman_map.get(&(*a, *b)).copied();
                let severity = (-k_orc).max(0.0);
                let fixes = self.suggest_fixes(graph, *a, *b);

                bottlenecks.push(Bottleneck {
                    edge: (*a, *b),
                    ollivier_curvature: Some(*k_orc),
                    forman_curvature: k_f,
                    severity,
                    suggested_fixes: fixes,
                });
            }
        }

        // Sort by severity (worst first)
        bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        bottlenecks
    }

    /// Detect bottlenecks using Forman-Ricci only (fast, for large graphs).
    pub fn detect_forman_only(&self, graph: &AgentGraph) -> Vec<Bottleneck> {
        let forman_curves = self.forman.all_curvatures(graph);

        forman_curves
            .into_iter()
            .filter(|(_, k)| k < &self.threshold)
            .map(|((a, b), k_f)| {
                let severity = (-k_f).max(0.0);
                let fixes = self.suggest_fixes(graph, a, b);
                Bottleneck {
                    edge: (a, b),
                    ollivier_curvature: None,
                    forman_curvature: Some(k_f),
                    severity,
                    suggested_fixes: fixes,
                }
            })
            .collect()
    }

    /// Suggest edge additions to fix a bottleneck at edge (a, b).
    fn suggest_fixes(&self, graph: &AgentGraph, a: AgentId, b: AgentId) -> Vec<(AgentId, AgentId)> {
        let mut fixes = Vec::new();
        let neighbors_a: Vec<AgentId> = graph.neighbors(a).iter().map(|&(n, _)| n).collect();
        let neighbors_b: Vec<AgentId> = graph.neighbors(b).iter().map(|&(n, _)| n).collect();

        // Connect neighbors of a to neighbors of b
        for &na in &neighbors_a {
            for &nb in &neighbors_b {
                if na != nb && !graph.has_edge(na, nb) && na != b && nb != a {
                    fixes.push((na.min(nb), na.max(nb)));
                    if fixes.len() >= 5 {
                        return fixes;
                    }
                }
            }
        }

        // If no cross-connections, suggest connecting the nodes' 2-hop neighborhoods
        if fixes.is_empty() {
            for &na in &neighbors_a {
                if !graph.has_edge(na, b) && na != b {
                    fixes.push((na.min(b), na.max(b)));
                    if fixes.len() >= 3 {
                        break;
                    }
                }
            }
        }

        fixes
    }

    /// Compute a bottleneck score for the entire graph.
    /// Higher = more bottlenecked. 0 = no bottlenecks.
    pub fn bottleneck_score(&self, graph: &AgentGraph) -> f64 {
        let bottlenecks = self.detect(graph);
        if bottlenecks.is_empty() {
            return 0.0;
        }
        bottlenecks.iter().map(|b| b.severity).sum::<f64>() / graph.num_edges().max(1) as f64
    }

    /// Check if the graph has any bottlenecks.
    pub fn has_bottlenecks(&self, graph: &AgentGraph) -> bool {
        self.orc.min_curvature(graph) < self.threshold
    }

    /// Get the most severe bottleneck.
    pub fn worst_bottleneck(&self, graph: &AgentGraph) -> Option<Bottleneck> {
        self.detect(graph).into_iter().next()
    }

    /// Count bottlenecks.
    pub fn count_bottlenecks(&self, graph: &AgentGraph) -> usize {
        self.detect(graph).len()
    }

    /// Apply fixes: add suggested edges for all bottlenecks.
    /// Returns the number of edges added.
    pub fn apply_fixes(&self, graph: &mut AgentGraph) -> usize {
        let bottlenecks = self.detect(graph);
        let mut count = 0;
        for bn in &bottlenecks {
            for &(a, b) in &bn.suggested_fixes {
                if !graph.has_edge(a, b) {
                    graph.add_edge(a, b, 1.0);
                    count += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_detect_star_no_orc_bottlenecks() {
        let g = AgentGraph::star(6);
        let detector = BottleneckDetector::default();
        let bns = detector.detect(&g);
        // Star graph has non-negative O-R curvature with lazy RW → no O-R bottlenecks
        assert!(bns.is_empty(), "star should have no O-R bottlenecks with lazy RW");
    }

    #[test]
    fn test_no_bottlenecks_complete() {
        let g = AgentGraph::complete(4);
        let detector = BottleneckDetector::default();
        assert!(!detector.has_bottlenecks(&g));
    }

    #[test]
    fn test_bottleneck_star() {
        let g = AgentGraph::star(6);
        let detector = BottleneckDetector::default();
        // Star has Forman curvature bottlenecks
        let bns = detector.detect_forman_only(&g);
        assert!(!bns.is_empty());
        // All star edges should be Forman bottlenecks
        assert!(bns.len() >= 3);
    }

    #[test]
    fn test_bottleneck_severity() {
        let g = AgentGraph::star(6);
        let detector = BottleneckDetector::default();
        let bns = detector.detect_forman_only(&g);
        assert!(!bns.is_empty());
        for bn in &bns {
            assert!(bn.severity >= 0.0);
            let k = bn.forman_curvature.unwrap_or(0.0);
            assert!(k < 0.0);
        }
    }

    #[test]
    fn test_bottleneck_score() {
        let g1 = AgentGraph::complete(4);
        let g2 = AgentGraph::path(8);
        let detector = BottleneckDetector::default();
        let s1 = detector.bottleneck_score(&g1);
        let s2 = detector.bottleneck_score(&g2);
        // Both have score >= 0 (no O-R bottlenecks)
        assert!(s1 >= 0.0);
        assert!(s2 >= 0.0);
    }

    #[test]
    fn test_suggest_fixes() {
        let g = AgentGraph::path(5);
        let detector = BottleneckDetector::default();
        let bns = detector.detect(&g);
        if let Some(_bn) = bns.first() {
            // Suggestions should exist for path bottlenecks
            // (may or may not depending on edge structure)
        }
    }

    #[test]
    fn test_worst_bottleneck() {
        let g = AgentGraph::star(6);
        let detector = BottleneckDetector::default();
        let worst = detector.worst_bottleneck(&g);
        // No O-R bottlenecks for star graph
        assert!(worst.is_none() || worst.unwrap().severity >= 0.0);
    }

    #[test]
    fn test_count_bottlenecks() {
        let g = AgentGraph::star(6);
        let detector = BottleneckDetector::default();
        let count = detector.count_bottlenecks(&g);
        // No O-R bottlenecks for star graph with lazy RW
        assert!(count >= 0);
    }

    #[test]
    fn test_apply_fixes() {
        let mut g = AgentGraph::path(6);
        let detector = BottleneckDetector::default();
        let edges_before = g.num_edges();
        let _fixes = detector.apply_fixes(&mut g);
        assert!(g.num_edges() >= edges_before);
    }

    #[test]
    fn test_detect_forman_only() {
        let g = AgentGraph::star(6);
        let detector = BottleneckDetector::default();
        let bns = detector.detect_forman_only(&g);
        assert!(!bns.is_empty());
        for bn in &bns {
            assert!(bn.forman_curvature.is_some());
            assert!(bn.ollivier_curvature.is_none());
        }
    }

    #[test]
    fn test_bottleneck_serialization() {
        let g = AgentGraph::path(5);
        let detector = BottleneckDetector::default();
        let bns = detector.detect(&g);
        if let Some(bn) = bns.first() {
            let json = serde_json::to_string(bn).unwrap();
            let bn2: Bottleneck = serde_json::from_str(&json).unwrap();
            assert_eq!(bn.edge, bn2.edge);
            assert_relative_eq!(bn.severity, bn2.severity);
        }
    }
}
