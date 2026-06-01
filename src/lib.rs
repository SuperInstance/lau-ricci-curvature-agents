//! # lau-ricci-curvature-agents
//!
//! Ollivier-Ricci and Forman-Ricci curvature on agent interaction graphs.
//!
//! High curvature = agents agree, fast consensus. Low/negative curvature =
//! agents disagree, information bottlenecks. Zero curvature = random mixing.

#![allow(unused_variables)]

pub mod graph;
pub mod ollivier_ricci;
pub mod forman_ricci;
pub mod belief_curvature;
pub mod concentration;
pub mod bonnet_myers;
pub mod curvature_flow;
pub mod bottleneck;
pub mod fleet;

pub use graph::{AgentGraph, AgentId, EdgeId};
pub use ollivier_ricci::{OllivierRicciCurvature, TransportSolver};
pub use forman_ricci::FormanRicciCurvature;
pub use belief_curvature::BeliefSpaceCurvature;
pub use concentration::CurvatureConcentration;
pub use bonnet_myers::BonnetMyers;
pub use curvature_flow::CurvatureFlow;
pub use bottleneck::{BottleneckDetector, Bottleneck};
pub use fleet::FleetDesigner;
