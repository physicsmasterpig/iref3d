import Foundation

/// Manifold data returned from the Rust core.
struct ManifoldData: Codable {
    let name: String
    let n: Int
    let r: Int
    let hard: Int
    let easy: Int
    let nzLatex: String?
    let nuX: String?
    let nuP: String?
}

/// A single index entry (one (m, e) query result).
struct IndexEntry: Codable {
    /// Per-cusp charges: [[aCoeff, bCoeff], ...]
    let charges: [[Double]]
    let series: String
    let source: String
}

/// NC cycle data for a single cusp.
struct NCCycleCusp: Codable {
    let cuspIdx: Int
    let cycles: [NCCycle]
}

struct NCCycle: Codable {
    let gamma: String   // LaTeX for γ_i expression
    let delta: String   // LaTeX for δ_i expression
    let weyl: String    // LaTeX for (a, b)
    let q1proj: String  // LaTeX for q^1 coefficient value
    let marginal: Bool
}

/// A single filling result card.
struct FillingResult: Codable {
    let title: String
    let kernelLabel: String
    let kernelClass: String  // "refined" or "marginal"
    let muted: String?
    let indexNotation: String  // LaTeX for the I_{...} subscript
    let series: String
    let edgeToggles: [EdgeToggle]?
}

struct EdgeToggle: Codable {
    let idx: Int
    let disabled: Bool
}
