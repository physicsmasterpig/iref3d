import SwiftUI

@main
struct iref3dApp: App {
    @StateObject private var appState = AppState()

    init() {
        // Force-disable state restoration
        UserDefaults.standard.removeObject(forKey: "NSQuitAlwaysKeepsWindows")
        UserDefaults.standard.set(false, forKey: "NSQuitAlwaysKeepsWindows")
    }

    var body: some Scene {
        WindowGroup("iref3d") {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 900, minHeight: 600)
                .onAppear {
                    // Ensure the window is visible
                    DispatchQueue.main.async {
                        NSApp.activate(ignoringOtherApps: true)
                        for w in NSApp.windows {
                            w.makeKeyAndOrderFront(nil)
                        }
                    }
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1200, height: 800)
    }
}
