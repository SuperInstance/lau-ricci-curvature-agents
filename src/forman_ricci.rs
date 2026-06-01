//! Forman-Ricci curvature: simpler combinatorial curvature.
//!
//! For an edge e = (u,v) with weight w_e:
//! F(e) = w_e * (w_u/w_e + w_v/w_e - deg_u - deg_v)
//!
//! Simplified unweighted:
//! F(u,v) = 4 - deg(u) - deg(v)

use crate::graph::{AgentGraph, AgentId};
use serde::{Deserialize, Serialize};

/// Forman-Ricci curvature computer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormanRicciCurvature {
    /// Whether to use weighted version.
    pub weighted: bool,
}

impl Default for FormanRicciCurvature {
    fn default() -> Self {
        Self { weighted: false }
    }
}

impl FormanRicciCurvature {
    pub fn new(weighted: bool) -> Self {
        Self { weighted }
    }

    /// Compute Forman-Ricci curvature for edge (a, b).
    ///
    /// Unweighted: F(a,b) = 4 - deg(a) - deg(b)
    /// Weighted: includes edge and vertex weights.
    pub fn edge_curvature(&self, graph: &AgentGraph, a: AgentId, b: AgentId) -> f64 {
        if !self.weighted {
            // Unweighted Forman-Ricci
            let deg_a = graph.degree(a) as f64;
            let deg_b = graph.degree(b) as f64;
            4.0 - deg_a - deg_b
        } else {
            // Weighted version
            let w_e = graph.edge_weight(a, b).unwrap_or(1.0);
            let w_a: f64 = graph.neighbors(a).iter().map(|&(_, w)| w).sum();
            let w_b: f64 = graph.neighbors(b).iter().map(|&(_, w)| w).sum();
            w_e * (w_a / w_e + w_b / w_e) - graph.degree(a) as f64 - graph.degree(b) as f64
        }
    }

    /// Compute curvature for all edges.
    pub fn all_curvatures(&self, graph: &AgentGraph) -> Vec<((AgentId, AgentId), f64)> {
        graph
            .edges()
            .into_iter()
            .map(|(a, b)| ((a, b), self.edge_curvature(graph, a, b)))
            .collect()
    }

    /// Average curvature.
    pub fn average_curvature(&self, graph: &AgentGraph) -> f64 {
        let all = self.all_curvatures(graph);
        if all.is_empty() {
            return 0.0;
        }
        all.iter().map(|&(_, k)| k).sum::<f64>() / all.len() as f64
    }

    /// Minimum curvature.
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

    /// Edges sorted by curvature (ascending = most negative first = bottlenecks).
    pub fn edges_by_curvature(&self, graph: &AgentGraph) -> Vec<((AgentId, AgentId), f64)> {
        let mut all = self.all_curvatures(graph);
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_forman_complete_graph() {
        let g = AgentGraph::complete(4);
        let f = FormanRicciCurvature::default();
        // K4: each node degree 3, F = 4 - 3 - 3 = -2
        let k = f.edge_curvature(&g, 0, 1);
        assert_relative_eq!(k, -2.0);
    }

    #[test]
    fn test_forman_path_endpoints() {
        let g = AgentGraph::path(4);
        let f = FormanRicciCurvature::default();
        // Path: node 0 degree 1, node 1 degree 2 → F = 4-1-2 = 1
        let k = f.edge_curvature(&g, 0, 1);
        assert_relative_eq!(k, 1.0);
    }

    #[test]
    fn test_forman_path_middle() {
        let g = AgentGraph::path(4);
        let f = FormanRicciCurvature::default();
        // node 1 degree 2, node 2 degree 2 → F = 4-2-2 = 0
        let k = f.edge_curvature(&g, 1, 2);
        assert_relative_eq!(k, 0.0);
    }

    #[test]
    fn test_forman_cycle() {
        let g = AgentGraph::cycle(6);
        let f = FormanRicciCurvature::default();
        // Cycle: all degrees 2, F = 4-2-2 = 0
        let k = f.edge_curvature(&g, 0, 1);
        assert_relative_eq!(k, 0.0);
    }

    #[test]
    fn test_forman_star_center_edge() {
        let g = AgentGraph::star(5);
        let f = FormanRicciCurvature::default();
        // center degree 4, leaf degree 1 → F = 4-4-1 = -1
        let k = f.edge_curvature(&g, 0, 1);
        assert_relative_eq!(k, -1.0);
    }

    #[test]
    fn test_forman_all_curvatures_count() {
        let g = AgentGraph::cycle(5);
        let f = FormanRicciCurvature::default();
        let all = f.all_curvatures(&g);
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_forman_weighted() {
        let mut g = AgentGraph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 1.0);
        let f = FormanRicciCurvature::new(true);
        let k = f.edge_curvature(&g, 0, 1);
        // w_e=2, w_0=2, w_1=3, deg_0=1, deg_1=2
        // F = 2*(2/2 + 3/2) - 1 - 2 = 2*(1+1.5) - 3 = 5-3 = 2
        assert_relative_eq!(k, 2.0);
    }

    #[test]
    fn test_forman_edges_sorted() {
        let g = AgentGraph::path(5);
        let f = FormanRicciCurvature::default();
        let sorted = f.edges_by_curvature(&g);
        // Middle edges have lower curvature than endpoint edges
        assert!(sorted[0].1 <= sorted.last().unwrap().1);
    }

    #[test]
    fn test_forman_average() {
        let g = AgentGraph::cycle(6);
        let f = FormanRicciCurvature::default();
        assert_relative_eq!(f.average_curvature(&g), 0.0);
    }

    #[test]
    fn test_forman_serialization() {
        let f = FormanRicciCurvature::new(true);
        let json = serde_json::to_string(&f).unwrap();
        let f2: FormanRicciCurvature = serde_json::from_str(&json).unwrap();
        assert_eq!(f.weighted, f2.weighted);
    }

    #[test]
    fn test_forman_grid() {
        let g = AgentGraph::grid(3, 3);
        let f = FormanRicciCurvature::default();
        // Corner: degree 2, edge to degree-3 node: F = 4-2-3 = -1
        // Interior: degree 4, edge to degree-4: F = 4-4-4 = -4
        let all = f.all_curvatures(&g);
        assert!(!all.is_empty());
        assert!(f.min_curvature(&g) < 0.0);
    }
}
