import SwiftUI

/// Pipeline stage — tracks what's been computed so far.
enum PipelineStage: Int, Comparable {
    case empty = 0
    case manifoldLoaded = 1
    case indexComputed = 2
    case fillingComputed = 3

    static func < (lhs: PipelineStage, rhs: PipelineStage) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// A manifold entry in the sidebar.
struct ManifoldEntry: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let n: Int
    let r: Int
}

@MainActor
final class AppState: ObservableObject {
    @Published var stage: PipelineStage = .empty
    @Published var selectedManifold: ManifoldEntry?
    @Published var nMax: Int = 10

    /// Set by sidebar to trigger a load in the WebView on next update cycle.
    @Published var pendingWebLoad: String?

    /// Sidebar: recent manifolds
    @Published var recentManifolds: [ManifoldEntry] = [
        ManifoldEntry(name: "m004", n: 2, r: 1),
        ManifoldEntry(name: "m003", n: 2, r: 1),
        ManifoldEntry(name: "s776", n: 5, r: 2),
        ManifoldEntry(name: "t12345", n: 7, r: 1),
    ]

    func loadManifold(name: String) {
        guard !name.isEmpty else { return }
        // Trigger WebView load — the WebView bridge will update AppState
        // when the Rust FFI returns real data.
        pendingWebLoad = name

        // Optimistic sidebar update (will be corrected by real data)
        let entry = ManifoldEntry(name: name, n: 0, r: 0)
        selectedManifold = entry
        stage = .manifoldLoaded

        // Push to recent (dedup)
        recentManifolds.removeAll { $0.name == name }
        recentManifolds.insert(entry, at: 0)
        if recentManifolds.count > 10 {
            recentManifolds.removeLast()
        }
    }
}
