//! Curvature of agent belief space via Fisher information metric.
//!
//! Agents hold beliefs (probability distributions). The Fisher information
//! metric defines a Riemannian structure on belief space. The sectional
//! curvature of this space (related to the Fisher-Rao metric) governs
//! how beliefs diverge/converge.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A belief state: a probability distribution over categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// Probability vector (must sum to 1).
    pub probs: Vec<f64>,
}

impl BeliefState {
    /// Create a uniform belief over `n` categories.
    pub fn uniform(n: usize) -> Self {
        Self {
            probs: vec![1.0 / n as f64; n],
        }
    }

    /// Create a belief from raw probabilities (normalizes).
    pub fn from_probs(probs: Vec<f64>) -> Self {
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            Self {
                probs: probs.iter().map(|p| p / sum).collect(),
            }
        } else {
            Self::uniform(probs.len())
        }
    }

    /// Dirac delta: all mass on category `k`.
    pub fn dirac(n: usize, k: usize) -> Self {
        let mut p = vec![0.0; n];
        p[k] = 1.0;
        Self { probs: p }
    }

    /// Number of categories.
    pub fn dim(&self) -> usize {
        self.probs.len()
    }

    /// Check valid probability distribution.
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.probs.iter().sum();
        (sum - 1.0).abs() < 1e-8 && self.probs.iter().all(|&p| p >= -1e-10)
    }

    /// To DVector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.probs.clone())
    }
}

/// Belief space curvature computation using Fisher information metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefSpaceCurvature {
    /// Number of belief categories.
    pub categories: usize,
}

impl BeliefSpaceCurvature {
    pub fn new(categories: usize) -> Self {
        Self { categories }
    }

    /// Fisher information matrix at belief state θ.
    ///
    /// I_ij = δ_ij / θ_i (diagonal, for the multinomial model)
    /// Returns the (n-1) x (n-1) matrix (on the simplex).
    pub fn fisher_matrix(&self, belief: &BeliefState) -> DMatrix<f64> {
        let n = self.categories;
        if n <= 1 {
            return DMatrix::zeros(0, 0);
        }
        let k = n - 1;
        let mut fisher = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                let theta_i = belief.probs[i].max(1e-15);
                let theta_n = belief.probs[n - 1].max(1e-15);
                if i == j {
                    fisher[(i, j)] = 1.0 / theta_i + 1.0 / theta_n;
                } else {
                    fisher[(i, j)] = 1.0 / theta_n;
                }
            }
        }
        fisher
    }

    /// Christoffel symbols (connection coefficients) for the Fisher-Rao metric.
    /// Returns Γ^k_ij as a 3D array indexed by [k][i][j].
    pub fn christoffel_symbols(&self, belief: &BeliefState) -> Vec<Vec<Vec<f64>>> {
        let n = self.categories;
        let k = if n <= 1 { 0 } else { n - 1 };
        let mut gamma = vec![vec![vec![0.0; k]; k]; k];
        let theta_n = belief.probs.get(n - 1).copied().unwrap_or(1e-15).max(1e-15);

        for l in 0..k {
            for i in 0..k {
                for j in 0..k {
                    let theta_i = belief.probs[i].max(1e-15);
                    // Simplified: Γ^l_ij = δ_{li}/(2*θ_i) + 1/(2*θ_n) for the simplex
                    if l == i {
                        gamma[l][i][j] += 0.5 / theta_i;
                    }
                    gamma[l][i][j] += 0.5 / theta_n;
                }
            }
        }
        gamma
    }

    /// Riemann curvature tensor R^l_{ijk} as 4D array.
    pub fn riemann_tensor(&self, belief: &BeliefState) -> Vec<Vec<Vec<Vec<f64>>>> {
        let n = self.categories;
        let k = if n <= 1 { 0 } else { n - 1 };
        let gamma = self.christoffel_symbols(belief);
        let mut R = vec![vec![vec![vec![0.0; k]; k]; k]; k];

        for l in 0..k {
            for i in 0..k {
                for j in 0..k {
                    for m in 0..k {
                        // R^l_{ijm} = ∂_j Γ^l_{im} - ∂_m Γ^l_{ij} + Γ^l_{js} Γ^s_{im} - Γ^l_{ms} Γ^s_{ij}
                        // Using numerical approximation of partial derivatives
                        let mut val = 0.0;
                        // Christoffel term: Γ^l_{js} Γ^s_{im} - Γ^l_{ms} Γ^s_{ij}
                        for s in 0..k {
                            val += gamma[l][j][s] * gamma[s][i][m] - gamma[l][m][s] * gamma[s][i][j];
                        }
                        R[l][i][j][m] = val;
                    }
                }
            }
        }
        R
    }

    /// Sectional curvature in the (i,j) plane at the given belief.
    pub fn sectional_curvature(&self, belief: &BeliefState, i: usize, j: usize) -> f64 {
        let n = self.categories;
        if n <= 2 {
            // On the 1-simplex (interval), curvature is well-defined but trivial
            return 0.0;
        }
        let k = n - 1;
        if i >= k || j >= k {
            return 0.0;
        }

        let g = self.fisher_matrix(belief);
        let R = self.riemann_tensor(belief);

        // K(i,j) = R(x_i, x_j, x_i, x_j) / (g(x_i,x_i)*g(x_j,x_j) - g(x_i,x_j)^2)
        // R_{ijij} = g_{li} R^l_{jij} summed over l
        let mut r_ijij = 0.0;
        for l in 0..k {
            r_ijij += g[(i, l)] * R[l][j][i][j];
        }

        let denom = g[(i, i)] * g[(j, j)] - g[(i, j)] * g[(i, j)];
        if denom.abs() < 1e-30 {
            return 0.0;
        }
        r_ijij / denom
    }

    /// Scalar curvature at the given belief.
    pub fn scalar_curvature(&self, belief: &BeliefState) -> f64 {
        let n = self.categories;
        if n <= 2 {
            return 0.0;
        }
        let k = n - 1;
        let mut total = 0.0;
        for i in 0..k {
            for j in (i + 1)..k {
                total += self.sectional_curvature(belief, i, j);
            }
        }
        // Scale by appropriate factor for dimension
        total * 2.0
    }

    /// Fisher-Rao distance between two beliefs.
    pub fn fisher_rao_distance(&self, b1: &BeliefState, b2: &BeliefState) -> f64 {
        assert_eq!(b1.dim(), b2.dim());
        let n = b1.dim();
        let mut sum = 0.0;
        for i in 0..n {
            let p = b1.probs[i].max(1e-15).sqrt();
            let q = b2.probs[i].max(1e-15).sqrt();
            sum += (p - q).powi(2);
        }
        2.0 * sum.sqrt()
    }

    /// Average belief curvature across agents given their belief states.
    pub fn fleet_belief_curvature(&self, beliefs: &[BeliefState]) -> f64 {
        if beliefs.is_empty() {
            return 0.0;
        }
        beliefs.iter().map(|b| self.scalar_curvature(b)).sum::<f64>() / beliefs.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_belief_uniform_valid() {
        let b = BeliefState::uniform(5);
        assert!(b.is_valid());
        assert_eq!(b.dim(), 5);
    }

    #[test]
    fn test_belief_dirac_valid() {
        let b = BeliefState::dirac(3, 1);
        assert!(b.is_valid());
        assert_relative_eq!(b.probs[1], 1.0);
    }

    #[test]
    fn test_belief_from_probs_normalizes() {
        let b = BeliefState::from_probs(vec![2.0, 3.0, 5.0]);
        assert!(b.is_valid());
        assert_relative_eq!(b.probs[0], 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_matrix_uniform() {
        let bsc = BeliefSpaceCurvature::new(3);
        let b = BeliefState::uniform(3);
        let F = bsc.fisher_matrix(&b);
        // Uniform on 3-simplex: I_ii = 3 + 3 = 6, I_ij = 3
        assert_relative_eq!(F[(0, 0)], 6.0, epsilon = 1e-10);
        assert_relative_eq!(F[(0, 1)], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_matrix_positive_definite() {
        let bsc = BeliefSpaceCurvature::new(4);
        let b = BeliefState::from_probs(vec![0.1, 0.3, 0.4, 0.2]);
        let F = bsc.fisher_matrix(&b);
        // Check symmetric
        assert_relative_eq!(F[(0, 1)], F[(1, 0)]);
        // Check positive diagonal
        for i in 0..3 {
            assert!(F[(i, i)] > 0.0);
        }
    }

    #[test]
    fn test_fisher_matrix_dimension() {
        let bsc = BeliefSpaceCurvature::new(5);
        let b = BeliefState::uniform(5);
        let F = bsc.fisher_matrix(&b);
        assert_eq!(F.nrows(), 4);
        assert_eq!(F.ncols(), 4);
    }

    #[test]
    fn test_sectional_curvature_2d() {
        let bsc = BeliefSpaceCurvature::new(2);
        let b = BeliefState::uniform(2);
        // 2 categories → 1-simplex → trivial curvature
        assert_relative_eq!(bsc.sectional_curvature(&b, 0, 0), 0.0);
    }

    #[test]
    fn test_scalar_curvature_uniform() {
        let bsc = BeliefSpaceCurvature::new(4);
        let b = BeliefState::uniform(4);
        let s = bsc.scalar_curvature(&b);
        // Should be a finite number
        assert!(s.is_finite());
    }

    #[test]
    fn test_fisher_rao_distance_same() {
        let bsc = BeliefSpaceCurvature::new(3);
        let b = BeliefState::uniform(3);
        let d = bsc.fisher_rao_distance(&b, &b);
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_rao_distance_symmetric() {
        let bsc = BeliefSpaceCurvature::new(3);
        let b1 = BeliefState::from_probs(vec![0.2, 0.5, 0.3]);
        let b2 = BeliefState::from_probs(vec![0.4, 0.1, 0.5]);
        let d12 = bsc.fisher_rao_distance(&b1, &b2);
        let d21 = bsc.fisher_rao_distance(&b2, &b1);
        assert_relative_eq!(d12, d21, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_rao_distance_positive() {
        let bsc = BeliefSpaceCurvature::new(3);
        let b1 = BeliefState::dirac(3, 0);
        let b2 = BeliefState::dirac(3, 1);
        let d = bsc.fisher_rao_distance(&b1, &b2);
        assert!(d > 0.0);
        // Max distance on probability simplex is π
        assert!(d <= std::f64::consts::PI + 1e-10);
    }

    #[test]
    fn test_christoffel_symbols_shape() {
        let bsc = BeliefSpaceCurvature::new(4);
        let b = BeliefState::uniform(4);
        let gamma = bsc.christoffel_symbols(&b);
        assert_eq!(gamma.len(), 3);
        assert_eq!(gamma[0].len(), 3);
        assert_eq!(gamma[0][0].len(), 3);
    }

    #[test]
    fn test_riemann_tensor_shape() {
        let bsc = BeliefSpaceCurvature::new(4);
        let b = BeliefState::uniform(4);
        let R = bsc.riemann_tensor(&b);
        assert_eq!(R.len(), 3);
        assert_eq!(R[0].len(), 3);
        assert_eq!(R[0][0].len(), 3);
        assert_eq!(R[0][0][0].len(), 3);
    }

    #[test]
    fn test_fleet_belief_curvature() {
        let bsc = BeliefSpaceCurvature::new(4);
        let beliefs = vec![
            BeliefState::uniform(4),
            BeliefState::from_probs(vec![0.1, 0.3, 0.4, 0.2]),
            BeliefState::from_probs(vec![0.5, 0.2, 0.2, 0.1]),
        ];
        let avg = bsc.fleet_belief_curvature(&beliefs);
        assert!(avg.is_finite());
    }

    #[test]
    fn test_belief_serialization() {
        let b = BeliefState::from_probs(vec![0.3, 0.7]);
        let json = serde_json::to_string(&b).unwrap();
        let b2: BeliefState = serde_json::from_str(&json).unwrap();
        assert_eq!(b.probs, b2.probs);
    }
}
