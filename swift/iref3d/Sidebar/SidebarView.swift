import SwiftUI

struct SidebarView: View {
    @EnvironmentObject var appState: AppState
    @State private var manifoldName: String = ""

    var body: some View {
        List {
            // Active manifold
            if let m = appState.selectedManifold {
                Section("Active") {
                    ManifoldCard(entry: m, isActive: true)
                }
            }

            // Quick load
            Section("Load") {
                HStack(spacing: 6) {
                    TextField("Name", text: $manifoldName)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { loadManifold() }
                    Button("Load") { loadManifold() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(manifoldName.isEmpty)
                }
            }

            // Recent
            Section("Recent") {
                ForEach(appState.recentManifolds) { entry in
                    ManifoldCard(entry: entry, isActive: false)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            manifoldName = entry.name
                            loadManifold()
                        }
                }
            }

            Spacer()

            // Tools
            Section {
                Label("Kernel Builder", systemImage: "cpu")
                Label("Data Packs", systemImage: "externaldrive")
                Label("Settings", systemImage: "gear")
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("iref3d")
    }

    private func loadManifold() {
        guard !manifoldName.isEmpty else { return }
        appState.loadManifold(name: manifoldName)
    }
}

struct ManifoldCard: View {
    let entry: ManifoldEntry
    let isActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(entry.name)
                .font(.system(size: 14, weight: .semibold))
            Text("n = \(entry.n), r = \(entry.r)")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}
