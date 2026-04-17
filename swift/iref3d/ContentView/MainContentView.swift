import SwiftUI
import WebKit

/// File logger shared with WebViewBridge — NSLog doesn't show in terminal on modern macOS.
func logToFile(_ msg: String) {
    let line = "\(Date()): \(msg)\n"
    // Try multiple paths
    for path in ["/tmp/iref3d_debug.log",
                 NSTemporaryDirectory() + "iref3d_debug.log",
                 NSHomeDirectory() + "/iref3d_debug.log"] {
        if let fh = FileHandle(forWritingAtPath: path) {
            fh.seekToEndOfFile()
            if let data = line.data(using: .utf8) { fh.write(data) }
            fh.closeFile()
            return
        } else if FileManager.default.createFile(atPath: path, contents: line.data(using: .utf8)) {
            return
        }
    }
    // Last resort: print to stdout (may or may not be visible)
    print("[iref3d] \(msg)")
}

struct MainContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        MathWebView()
            .environmentObject(appState)
    }
}

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "function")
                .font(.system(size: 48))
                .foregroundStyle(.tertiary)
            Text("Refined Index Calculator")
                .font(.title2)
                .foregroundStyle(.secondary)
            Text("Load a manifold from the sidebar to begin.")
                .font(.body)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - WKWebView wrapper

struct MathWebView: NSViewRepresentable {
    @EnvironmentObject var appState: AppState

    func makeCoordinator() -> WebViewBridge {
        WebViewBridge()
    }

    func makeNSView(context: Context) -> WKWebView {
        try? "makeNSView at \(Date())".write(toFile: "/tmp/iref3d_makeNSView.txt", atomically: true, encoding: .utf8)
        logToFile("makeNSView called")
        let config = WKWebViewConfiguration()
        let contentController = config.userContentController

        // Register JS -> Swift message handler
        contentController.add(context.coordinator, name: "bridge")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        context.coordinator.webView = webView

        #if DEBUG
        webView.isInspectable = true
        #endif

        loadContent(webView: webView)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        // When AppState changes (e.g. sidebar load), inject the name into the
        // WebView's input field and trigger load via JS.
        if let pending = appState.pendingWebLoad {
            appState.pendingWebLoad = nil
            let escaped = pending.replacingOccurrences(of: "'", with: "\\'")
            webView.evaluateJavaScript("""
                document.getElementById('inp-name').value = '\(escaped)';
                doLoadManifold();
            """, completionHandler: nil)
        }
    }

    private func loadContent(webView: WKWebView) {
        let bundle = Bundle.main
        logToFile("loadContent: bundle=\(bundle.bundlePath)")

        if let url = bundle.url(forResource: "content", withExtension: "html") {
            let dir = url.deletingLastPathComponent()
            logToFile("Loaded content.html from: \(url.path)")
            webView.loadFileURL(url, allowingReadAccessTo: dir)
            return
        }

        if let url = bundle.url(forResource: "content", withExtension: "html", subdirectory: "Resources") {
            let dir = url.deletingLastPathComponent()
            logToFile("Loaded content.html from subdirectory: \(url.path)")
            webView.loadFileURL(url, allowingReadAccessTo: dir)
            return
        }

        let manualURL = bundle.bundleURL.appendingPathComponent("Contents/Resources/Resources/content.html")
        if FileManager.default.fileExists(atPath: manualURL.path) {
            let dir = manualURL.deletingLastPathComponent()
            logToFile("Loaded content.html from manual path: \(manualURL.path)")
            webView.loadFileURL(manualURL, allowingReadAccessTo: dir)
            return
        }

        logToFile("ERROR: content.html not found in bundle: \(bundle.bundlePath)")
        if let items = try? FileManager.default.contentsOfDirectory(atPath: bundle.bundlePath + "/Contents/Resources") {
            logToFile("Bundle resources: \(items)")
        }
    }
}
