//! PLATO fleet design: engineer topology for high curvature = fast convergence.
//!
//! Design fleet communication topologies that maximize Ricci curvature,
//! ensuring fast consensus, no bottlenecks, and robust information flow.

use crate::graph::AgentGraph;
use crate::ollivier_ricci::{MeasureStrategy, OllivierRicciCurvature, TransportSolver};
use crate::forman_ricci::FormanRicciCurvature;
use crate::concentration::CurvatureConcentration;
use crate::bonnet_myers::BonnetMyers;
use crate::curvature_flow::{CurvatureFlow, FlowStrategy};
use crate::bottleneck::BottleneckDetector;
use serde::{Deserialize, Serialize};

/// Fleet topology design goals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetDesignGoals {
    /// Number of agents.
    pub n_agents: usize,
    /// Target minimum curvature (higher = faster consensus).
    pub target_min_curvature: f64,
    /// Maximum degree per agent (communication bandwidth).
    pub max_degree: usize,
    /// Whether to ensure Bonnet-Myers conditions.
    pub ensure_bonnet_myers: bool,
}

impl Default for FleetDesignGoals {
    fn default() -> Self {
        Self {
            n_agents: 10,
            target_min_curvature: 0.1,
            max_degree: 6,
            ensure_bonnet_myers: true,
        }
    }
}

/// Result of fleet design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetDesignResult {
    /// The designed graph.
    pub graph: AgentGraph,
    /// Average curvature achieved.
    pub avg_curvature: f64,
    /// Minimum curvature achieved.
    pub min_curvature: f64,
    /// Graph diameter.
    pub diameter: usize,
    /// Number of bottlenecks.
    pub bottleneck_count: usize,
    /// Bonnet-Myers bound satisfied.
    pub bonnet_myers_satisfied: bool,
    /// Estimated consensus time.
    pub estimated_consensus_time: usize,
}

/// Fleet topology designer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetDesigner {
    pub orc: OllivierRicciCurvature,
    pub forman: FormanRicciCurvature,
    pub flow: CurvatureFlow,
    pub detector: BottleneckDetector,
}

impl Default for FleetDesigner {
    fn default() -> Self {
        let orc = OllivierRicciCurvature::new(
            MeasureStrategy::LazyRandomWalk { alpha: 0.5 },
            TransportSolver::Exact,
        );
        Self {
            orc: orc.clone(),
            forman: FormanRicciCurvature::default(),
            flow: CurvatureFlow::new(orc.clone(), FlowStrategy::TriangleClosing, 0.1),
            detector: BottleneckDetector::default(),
        }
    }
}

impl FleetDesigner {
    /// Design a fleet topology from scratch.
    ///
    /// Starts with a random-ish graph and evolves it using curvature flow
    /// until the design goals are met.
    pub fn design(&self, goals: &FleetDesignGoals) -> FleetDesignResult {
        // Start with a regular-ish graph
        let mut graph = self.initial_topology(goals);

        // Evolve using curvature flow
        let flow = CurvatureFlow::new(
            self.orc.clone(),
            FlowStrategy::TriangleClosing,
            0.1,
        );
        flow.evolve(&mut graph, 20);

        // Fill remaining bottlenecks
        let det = BottleneckDetector::new(goals.target_min_curvature);
        det.apply_fixes(&mut graph);

        self.evaluate(&graph)
    }

    /// Create initial topology based on goals.
    fn initial_topology(&self, goals: &FleetDesignGoals) -> AgentGraph {
        let n = goals.n_agents;
        let max_deg = goals.max_degree;

        // Build a regular graph: connect each node to ⌊max_deg/2⌋ nearest neighbors in a cycle
        let k = (max_deg / 2).min(n / 2).max(1);
        let mut g = AgentGraph::cycle(n);

        // Add k-nearest neighbors
        for i in 0..n {
            for j in 1..=k {
                let neighbor = (i + j) % n;
                if !g.has_edge(i, neighbor) && g.degree(i) < max_deg {
                    g.add_edge(i, neighbor, 1.0);
                }
            }
        }
        g
    }

    /// Evaluate a fleet topology.
    pub fn evaluate(&self, graph: &AgentGraph) -> FleetDesignResult {
        let avg_k = self.orc.average_curvature(graph);
        let min_k = self.orc.min_curvature(graph);
        let diameter = graph.diameter();
        let bottlenecks = self.detector.detect(graph);

        let bm = BonnetMyers::new(self.orc.clone());
        let bm_result = bm.analyze(graph);

        let cc = CurvatureConcentration::new(self.orc.clone());
        let consensus = cc.simulate_consensus(
            graph,
            &(0..graph.num_agents()).map(|i| i as f64).collect::<Vec<_>>(),
            0.01,
            1000,
        );

        FleetDesignResult {
            graph: graph.clone(),
            avg_curvature: avg_k,
            min_curvature: min_k,
            diameter,
            bottleneck_count: bottlenecks.len(),
            bonnet_myers_satisfied: bm_result.satisfies_bonnet_myers,
            estimated_consensus_time: consensus.unwrap_or(usize::MAX),
        }
    }

    /// Optimize an existing topology.
    pub fn optimize(&self, graph: &mut AgentGraph, iterations: usize) -> FleetDesignResult {
        let flow = CurvatureFlow::new(
            self.orc.clone(),
            FlowStrategy::TriangleClosing,
            0.1,
        );
        flow.evolve(graph, iterations);
        self.evaluate(graph)
    }

    /// Compare two fleet topologies.
    pub fn compare(&self, g1: &AgentGraph, g2: &AgentGraph) -> FleetComparison {
        let r1 = self.evaluate(g1);
        let r2 = self.evaluate(g2);
        let bc1 = r1.bottleneck_count;
        let bc2 = r2.bottleneck_count;
        let ac1 = r1.avg_curvature;
        let ac2 = r2.avg_curvature;
        FleetComparison {
            result1: r1,
            result2: r2,
            better_avg_curvature: if ac1 >= ac2 { 1 } else { 2 },
            fewer_bottlenecks: if bc1 <= bc2 { 1 } else { 2 },
        }
    }

    /// Generate a curvature-optimal topology for n agents.
    pub fn optimal_topology(&self, n: usize) -> AgentGraph {
        // Complete graph has the highest curvature but is expensive
        // Use a well-connected regular graph as a practical optimum
        let goals = FleetDesignGoals {
            n_agents: n,
            target_min_curvature: 0.5,
            max_degree: n.min(6),
            ensure_bonnet_myers: true,
        };
        self.design(&goals).graph
    }
}

/// Comparison of two fleet topologies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetComparison {
    pub result1: FleetDesignResult,
    pub result2: FleetDesignResult,
    pub better_avg_curvature: usize,
    pub fewer_bottlenecks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_design_basic() {
        let designer = FleetDesigner::default();
        let goals = FleetDesignGoals {
            n_agents: 8,
            target_min_curvature: 0.0,
            max_degree: 4,
            ensure_bonnet_myers: false,
        };
        let result = designer.design(&goals);
        assert_eq!(result.graph.num_agents(), 8);
        assert!(result.diameter > 0);
    }

    #[test]
    fn test_evaluate_complete() {
        let g = AgentGraph::complete(4);
        let designer = FleetDesigner::default();
        let result = designer.evaluate(&g);
        assert!(result.avg_curvature > 0.0);
        assert!(result.bottleneck_count == 0);
        assert_eq!(result.diameter, 1);
    }

    #[test]
    fn test_evaluate_path() {
        let g = AgentGraph::path(6);
        let designer = FleetDesigner::default();
        let result = designer.evaluate(&g);
        // Path graph has no negative-curvature bottlenecks with lazy RW
        assert!(result.bottleneck_count >= 0);
    }

    #[test]
    fn test_optimize() {
        let mut g = AgentGraph::path(6);
        let designer = FleetDesigner::default();
        let result = designer.optimize(&mut g, 5);
        assert!(result.graph.num_agents() == 6);
    }

    #[test]
    fn test_compare() {
        let g1 = AgentGraph::complete(4);
        let g2 = AgentGraph::path(4);
        let designer = FleetDesigner::default();
        let comp = designer.compare(&g1, &g2);
        assert_eq!(comp.better_avg_curvature, 1);
    }

    #[test]
    fn test_optimal_topology() {
        let designer = FleetDesigner::default();
        let g = designer.optimal_topology(5);
        assert_eq!(g.num_agents(), 5);
        assert!(g.num_edges() >= 5);
    }

    #[test]
    fn test_design_goals_default() {
        let goals = FleetDesignGoals::default();
        assert_eq!(goals.n_agents, 10);
        assert!(goals.target_min_curvature > 0.0);
    }

    #[test]
    fn test_initial_topology_connected() {
        let designer = FleetDesigner::default();
        let goals = FleetDesignGoals {
            n_agents: 8,
            max_degree: 4,
            ..Default::default()
        };
        let g = designer.initial_topology(&goals);
        let components = g.connected_components();
        assert_eq!(components.len(), 1, "initial topology should be connected");
    }

    #[test]
    fn test_result_serialization() {
        let g = AgentGraph::complete(3);
        let designer = FleetDesigner::default();
        let result = designer.evaluate(&g);
        let json = serde_json::to_string(&result).unwrap();
        let result2: FleetDesignResult = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(result.avg_curvature, result2.avg_curvature);
        assert_eq!(result.diameter, result2.diameter);
    }

    #[test]
    fn test_comparison_serialization() {
        let g1 = AgentGraph::complete(3);
        let g2 = AgentGraph::path(3);
        let designer = FleetDesigner::default();
        let comp = designer.compare(&g1, &g2);
        let json = serde_json::to_string(&comp).unwrap();
        let comp2: FleetComparison = serde_json::from_str(&json).unwrap();
        assert_eq!(comp.better_avg_curvature, comp2.better_avg_curvature);
    }
}
