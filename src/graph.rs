//! Agent interaction graph representation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for an agent node.
pub type AgentId = usize;

/// Unique identifier for an edge.
pub type EdgeId = (AgentId, AgentId);

/// Weight for edges and measures.
pub type Weight = f64;

mod edge_weights_serde {
    use super::{AgentId, Weight};
    use serde::{Serializer, Deserializer, Serialize, Deserialize};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(map: &HashMap<(AgentId, AgentId), Weight>, s: S) -> Result<S::Ok, S::Error> {
        let vec: Vec<((AgentId, AgentId), Weight)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        vec.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<HashMap<(AgentId, AgentId), Weight>, D::Error> {
        let vec: Vec<((AgentId, AgentId), Weight)> = Vec::deserialize(d)?;
        Ok(vec.into_iter().collect())
    }
}

/// An agent interaction graph — agents are nodes, interactions are weighted edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGraph {
    /// Number of agents (nodes are 0..n)
    n: usize,
    /// Adjacency: agent -> list of (neighbor, weight)
    adj: HashMap<AgentId, Vec<(AgentId, Weight)>>,
    /// Edge weights indexed by sorted pair
    #[serde(with = "edge_weights_serde")]
    edge_weights: HashMap<EdgeId, Weight>,
}

impl AgentGraph {
    /// Create an empty graph with `n` agents and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: HashMap::new(),
            edge_weights: HashMap::new(),
        }
    }

    /// Number of agents.
    pub fn num_agents(&self) -> usize {
        self.n
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edge_weights.len()
    }

    /// Add an undirected edge with weight (default 1.0).
    pub fn add_edge(&mut self, a: AgentId, b: AgentId, w: Weight) {
        assert!(a < self.n && b < self.n, "agent id out of range");
        if a == b {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self.edge_weights.insert((lo, hi), w);
        self.adj.entry(a).or_default().push((b, w));
        self.adj.entry(b).or_default().push((a, w));
    }

    /// Get edge weight.
    pub fn edge_weight(&self, a: AgentId, b: AgentId) -> Option<Weight> {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self.edge_weights.get(&(lo, hi)).copied()
    }

    /// Get neighbors of agent `a`.
    pub fn neighbors(&self, a: AgentId) -> &[(AgentId, Weight)] {
        self.adj.get(&a).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Degree of agent `a`.
    pub fn degree(&self, a: AgentId) -> usize {
        self.neighbors(a).len()
    }

    /// All edges as sorted pairs.
    pub fn edges(&self) -> Vec<EdgeId> {
        let mut e: Vec<_> = self.edge_weights.keys().copied().collect();
        e.sort();
        e
    }

    /// All agent IDs.
    pub fn agents(&self) -> Vec<AgentId> {
        (0..self.n).collect()
    }

    /// Check if edge exists.
    pub fn has_edge(&self, a: AgentId, b: AgentId) -> bool {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        self.edge_weights.contains_key(&(lo, hi))
    }

    /// Shortest path distance (unweighted BFS).
    pub fn distance(&self, a: AgentId, b: AgentId) -> Option<usize> {
        if a == b {
            return Some(0);
        }
        let mut visited = vec![false; self.n];
        let mut dist = vec![0usize; self.n];
        let mut queue = std::collections::VecDeque::new();
        visited[a] = true;
        queue.push_back(a);
        while let Some(u) = queue.pop_front() {
            for &(v, _) in self.neighbors(u) {
                if !visited[v] {
                    visited[v] = true;
                    dist[v] = dist[u] + 1;
                    if v == b {
                        return Some(dist[v]);
                    }
                    queue.push_back(v);
                }
            }
        }
        None
    }

    /// Graph diameter (longest shortest path).
    pub fn diameter(&self) -> usize {
        let mut max_d = 0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if let Some(d) = self.distance(i, j) {
                    max_d = max_d.max(d);
                }
            }
        }
        max_d
    }

    /// Build a complete graph on n agents.
    pub fn complete(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j, 1.0);
            }
        }
        g
    }

    /// Build a path graph on n agents.
    pub fn path(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    /// Build a cycle graph on n agents.
    pub fn cycle(n: usize) -> Self {
        let mut g = Self::path(n);
        if n > 2 {
            g.add_edge(0, n - 1, 1.0);
        }
        g
    }

    /// Build a star graph: agent 0 connected to all others.
    pub fn star(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 1..n {
            g.add_edge(0, i, 1.0);
        }
        g
    }

    /// Build a grid graph (rows x cols).
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut g = Self::new(n);
        for r in 0..rows {
            for c in 0..cols {
                let u = r * cols + c;
                if c + 1 < cols {
                    g.add_edge(u, u + 1, 1.0);
                }
                if r + 1 < rows {
                    g.add_edge(u, u + cols, 1.0);
                }
            }
        }
        g
    }

    /// Compute the Laplacian matrix (n x n).
    pub fn laplacian(&self) -> nalgebra::DMatrix<f64> {
        let n = self.n;
        let mut l = nalgebra::DMatrix::zeros(n, n);
        for (&(a, b), &w) in &self.edge_weights {
            l[(a, a)] += w;
            l[(b, b)] += w;
            l[(a, b)] -= w;
            l[(b, a)] -= w;
        }
        l
    }

    /// Connected components.
    pub fn connected_components(&self) -> Vec<Vec<AgentId>> {
        let mut visited = vec![false; self.n];
        let mut components = Vec::new();
        for start in 0..self.n {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(u) = stack.pop() {
                comp.push(u);
                for &(v, _) in self.neighbors(u) {
                    if !visited[v] {
                        visited[v] = true;
                        stack.push(v);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
}
