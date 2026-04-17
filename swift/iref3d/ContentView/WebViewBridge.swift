import WebKit
import AppKit

/// Handles JS -> Swift messages from the WKWebView.
final class WebViewBridge: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    weak var webView: WKWebView?

    /// Path to census.db — bundled or found at a fixed dev path.
    private var dbPath: String {
        if let bundled = Bundle.main.url(forResource: "census", withExtension: "db") {
            return bundled.path
        }
        // Dev fallback: relative to the repo
        let devPath = (Bundle.main.bundlePath as NSString)
            .deletingLastPathComponent  // .app dir
        let candidates = [
            (devPath as NSString).appendingPathComponent("../../data/census.db"),
            NSString(string: "~/Documents/Research/iref3d/data/census.db").expandingTildeInPath,
        ]
        for c in candidates {
            let resolved = (c as NSString).standardizingPath
            if FileManager.default.fileExists(atPath: resolved) {
                return resolved
            }
        }
        return "census.db" // will fail with a clear Rust error
    }

    /// Stored NC cycles from the last search (for filling lookups).
    private var ncCycleData: [[String: Any]] = []
    /// Stored compat results per cusp per cycle.
    private var ncCompatCache: [String: [String: Any]] = [:]
    /// Cancellation flag — checked between steps of multi-call operations.
    private var cancelled: Set<String> = []

    // MARK: - JS -> Swift

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard let body = message.body as? [String: Any],
              let action = body["action"] as? String else {
            NSLog("Bridge: invalid message format")
            return
        }

        let params = body["params"] as? [String: Any] ?? [:]
        logToFile("Bridge: \(action)(\(params))")

        switch action {
        case "loadManifold":
            handleLoadManifold(params)
        case "computeIndex":
            handleComputeIndex(params)
        case "findNCCycles":
            handleFindNCCycles(params)
        case "computeFilling":
            handleComputeFilling(params)
        case "cancel":
            if let task = params["task"] as? String {
                logToFile("Bridge: cancel(\(task))")
                cancelled.insert(task)
            }
        case "copyLatex":
            handleCopyLatex(params)
        case "export":
            handleExport(params)
        default:
            NSLog("Bridge: unknown action '%@'", action)
        }
    }

    // MARK: - Navigation delegate

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        logToFile("WebView: page loaded successfully")
        webView.evaluateJavaScript("if (typeof onReady === 'function') onReady();", completionHandler: nil)

        #if DEBUG
        webView.evaluateJavaScript("""
            (function() {
                if (typeof runTests === 'function') {
                    var r = runTests();
                    return JSON.stringify(r);
                }
                return '{"error":"runTests not found"}';
            })()
        """) { result, error in
            if let error {
                NSLog("Test error: %@", error.localizedDescription)
            } else if let json = result as? String,
                      let data = json.data(using: .utf8),
                      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let pass = obj["pass"] as? Int ?? 0
                let fail = obj["fail"] as? Int ?? 0
                let log = obj["log"] as? [String] ?? []
                NSLog("═══ JS Tests: %d passed, %d failed ═══", pass, fail)
                for entry in log {
                    NSLog("  %@", entry)
                }
                if fail > 0 {
                    NSLog("⚠️  %d test(s) FAILED — check content.js", fail)
                }
            }
        }
        #endif
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        NSLog("WebView navigation failed: %@", error.localizedDescription)
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        NSLog("WebView provisional navigation failed: %@", error.localizedDescription)
    }

    // MARK: - Swift -> JS helpers

    func callJS(_ js: String) {
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(js) { _, error in
                if let error {
                    NSLog("Bridge JS error: %@", error.localizedDescription)
                }
            }
        }
    }

    func sendJSON(_ function: String, _ jsonString: String) {
        callJS("\(function)(\(jsonString));")
    }

    // MARK: - Load manifold

    private func handleLoadManifold(_ params: [String: Any]) {
        let name = params["name"] as? String ?? "m006"
        let db = dbPath
        logToFile("Bridge: loading '\(name)' from \(db)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = RustFFI.loadManifold(dbPath: db, name: name)
            logToFile("Bridge: Rust returned keys=\(Array(result.keys))")

            if let error = result["error"] as? String {
                logToFile("Bridge: loadManifold error: \(error)")
                self?.sendJSON("updateManifold", """
                    {"name":"\(name)","n":0,"r":0,"error":"\(error)"}
                """)
                return
            }

            // Transform Rust output to JS-expected format
            let n = result["n"] as? Int ?? 0
            let r = result["r"] as? Int ?? 0
            let hard = result["hard"] as? Int ?? 0
            let easy = result["easy"] as? Int ?? 0
            let nzG = result["nz_g_x2"] as? [Any] ?? []
            let nuX = result["nu_x"] as? [Any] ?? []
            let nuP = result["nu_p_x2"] as? [Any] ?? []

            // Build NZ LaTeX: g_NZ matrix from doubled values
            let nzLatex = Self.buildNZLatex(nzG: nzG, n: n, r: r)
            let nuXStr = Self.formatVector(nuX)
            let nuPStr = Self.formatHalfVector(nuP)

            let out: [String: Any] = [
                "name": result["name"] as? String ?? name,
                "n": n, "r": r, "hard": hard, "easy": easy,
                "nzLatex": nzLatex,
                "nuX": nuXStr,
                "nuP": nuPStr,
            ]

            if let json = try? JSONSerialization.data(withJSONObject: out),
               let jsonStr = String(data: json, encoding: .utf8) {
                self?.sendJSON("updateManifold", jsonStr)
            }
        }
    }

    // MARK: - Compute index

    private func handleComputeIndex(_ params: [String: Any]) {
        // JS sends parseInt/parseFloat — NSNumber may arrive as Int or Double
        let mLo = (params["mLo"] as? Int) ?? Int(params["mLo"] as? Double ?? 0)
        let mHi = (params["mHi"] as? Int) ?? Int(params["mHi"] as? Double ?? 0)
        let eLo = (params["eLo"] as? Double) ?? Double(params["eLo"] as? Int ?? 0)
        let eHi = (params["eHi"] as? Double) ?? Double(params["eHi"] as? Int ?? 0)
        let qqOrder = (params["qqOrder"] as? Int) ?? 10

        let eLoX2 = Int(eLo * 2)
        let eHiX2 = Int(eHi * 2)
        let eStepX2 = 1

        logToFile("computeIndex: m=\(mLo)..\(mHi) e_x2=\(eLoX2)..\(eHiX2) qq=\(qqOrder)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = RustFFI.computeIndex(
                mLo: mLo, mHi: mHi,
                eLoX2: eLoX2, eHiX2: eHiX2, eStepX2: eStepX2,
                qqOrder: qqOrder
            )
            logToFile("computeIndex: Rust returned keys=\(Array(result.keys))")

            if let error = result["error"] as? String {
                logToFile("computeIndex error: \(error)")
                self?.callJS("document.getElementById('index-status').textContent = 'Error: \(error.replacingOccurrences(of: "'", with: "\\'"))';")
                return
            }

            let rawEntries = result["entries"] as? [[String: Any]] ?? []
            let r = result["r"] as? Int ?? 1
            logToFile("computeIndex: \(rawEntries.count) entries, r=\(r)")

            var jsEntries: [[String: Any]] = []
            for entry in rawEntries {
                let mExt = (entry["m_ext"] as? [Int]) ?? (entry["m_ext"] as? [Any])?.map { ($0 as? Int) ?? Int($0 as? Double ?? 0) } ?? []
                let eExtX2 = (entry["e_ext_x2"] as? [Int]) ?? (entry["e_ext_x2"] as? [Any])?.map { ($0 as? Int) ?? Int($0 as? Double ?? 0) } ?? []
                let series = entry["series"] as? [String: Any] ?? [:]

                var charges: [[Double]] = []
                for i in 0..<r {
                    let m = i < mExt.count ? Double(mExt[i]) : 0
                    let e = i < eExtX2.count ? Double(eExtX2[i]) / 2.0 : 0
                    charges.append([m, e])
                }

                // Series keys are JSON-encoded Vec<i64>, values are i64
                var intSeries: [String: Int] = [:]
                for (k, v) in series {
                    intSeries[k] = (v as? Int) ?? Int(v as? Double ?? 0)
                }
                let seriesLatex = Self.formatQSeries(intSeries, qqOrder: qqOrder)

                jsEntries.append([
                    "charges": charges,
                    "series": seriesLatex,
                    "source": "computed",
                ])
            }

            let out: [String: Any] = ["entries": jsEntries, "r": r]
            if let json = try? JSONSerialization.data(withJSONObject: out),
               let jsonStr = String(data: json, encoding: .utf8) {
                logToFile("computeIndex: sending \(jsEntries.count) entries to JS")
                self?.sendJSON("updateIndexResults", jsonStr)
            }
        }
    }

    // MARK: - Find NC cycles

    private func handleFindNCCycles(_ params: [String: Any]) {
        let pMax = params["pMax"] as? Int ?? 5
        let qMax = params["qMax"] as? Int ?? 5
        let qqOrder = params["qqOrder"] as? Int ?? 10

        cancelled.remove("find-nc")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            var allCusps: [[String: Any]] = []
            let start = DispatchTime.now()

            for cuspIdx in 0..<10 {
                if cancelled.contains("find-nc") {
                    logToFile("findNCCycles: cancelled")
                    return
                }

                let result = RustFFI.findNCCycles(
                    cuspIdx: cuspIdx,
                    pMin: -pMax, pMax: pMax + 1,
                    qMin: -qMax, qMax: qMax + 1,
                    qqOrder: qqOrder
                )

                if result["error"] != nil { break }

                let rawCycles = result["cycles"] as? [[String: Any]] ?? []
                if rawCycles.isEmpty && cuspIdx > 0 { break }

                var cycles: [[String: Any]] = []
                // Deduplicate ±(p,q): keep the canonical representative
                // where the first nonzero component is positive.
                var seen = Set<String>()
                for cyc in rawCycles {
                    if cancelled.contains("find-nc") { return }
                    let p = cyc["p"] as? Int ?? 0
                    let q = cyc["q"] as? Int ?? 0

                    // Canonical form: first nonzero component positive
                    let (cp, cq): (Int, Int)
                    if p > 0 || (p == 0 && q > 0) {
                        (cp, cq) = (p, q)
                    } else {
                        (cp, cq) = (-p, -q)
                    }
                    let key = "\(cp),\(cq)"
                    if seen.contains(key) { continue }
                    seen.insert(key)

                    let compat = RustFFI.ncCompat(
                        cuspIdx: cuspIdx, p: cp, q: cq, qqOrder: qqOrder
                    )

                    let isMarginal = compat["is_marginal"] as? Bool ?? false
                    let q1proj = compat["unrefined_q1_proj"] as? Int ?? 0
                    let abValid = compat["ab_valid"] as? Bool ?? false

                    cycles.append([
                        "gamma": Self.slopeLabel(p: cp, q: cq, cusp: cuspIdx, component: "alpha"),
                        "delta": Self.slopeLabel(p: -cq, q: cp, cusp: cuspIdx, component: "beta"),
                        "weyl": abValid ? "(0,\\, 0)" : "—",
                        "q1proj": "\(q1proj)",
                        "marginal": isMarginal,
                        "p": cp, "q": cq,
                        "collapsed_edges": compat["collapsed_edges"] ?? [],
                    ])
                }

                allCusps.append([
                    "cuspIdx": cuspIdx,
                    "cycles": cycles,
                ])
            }

            self.ncCycleData = allCusps

            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
            let data: [String: Any] = [
                "cusps": allCusps,
                "elapsed": String(format: "%.1f", elapsed),
            ]

            if let json = try? JSONSerialization.data(withJSONObject: data),
               let jsonStr = String(data: json, encoding: .utf8) {
                self.sendJSON("updateNCCycles", jsonStr)
            }
        }
    }

    // MARK: - Compute filling

    private func handleComputeFilling(_ params: [String: Any]) {
        let cusps = params["cusps"] as? [[String: Any]] ?? []
        cancelled.remove("compute-filling")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            var results: [[String: Any]] = []

            for cuspSpec in cusps {
                if cancelled.contains("compute-filling") {
                    logToFile("computeFilling: cancelled")
                    return
                }
                let cuspIdx = cuspSpec["cuspIdx"] as? Int ?? 0
                let selectedCycles = cuspSpec["selectedCycles"] as? [Int] ?? []

                // Look up stored NC cycle data for this cusp
                guard cuspIdx < ncCycleData.count,
                      let cycles = ncCycleData[cuspIdx]["cycles"] as? [[String: Any]] else {
                    continue
                }

                for cycleIdx in selectedCycles {
                    guard cycleIdx < cycles.count else { continue }
                    let cycle = cycles[cycleIdx]
                    let p = cycle["p"] as? Int ?? 0
                    let q = cycle["q"] as? Int ?? 0
                    let isMarginal = cycle["marginal"] as? Bool ?? false
                    let collapsed = cycle["collapsed_edges"] as? [Int] ?? []

                    logToFile("filling: cusp=\(cuspIdx) p=\(p) q=\(q) collapsed=\(collapsed)")
                    let fillResult = RustFFI.filledRefinedIndex(
                        cuspIdx: cuspIdx, p: p, q: q, qqOrder: 10,
                        incompatEdges: collapsed
                    )
                    logToFile("filling: Rust returned keys=\(Array(fillResult.keys))")

                    if let error = fillResult["error"] as? String {
                        logToFile("filling error for (\(p),\(q)): \(error)")
                        continue
                    }

                    let hjKs = fillResult["hj_ks"] as? [Int] ?? []
                    let _ = fillResult["has_cusp_eta"] as? Bool ?? false
                    let rawSeries = fillResult["series"] as? [String: [Int]] ?? [:]

                    // Format the result for JS
                    let gammaLabel = cycle["gamma"] as? String ?? "?"
                    let kernelLabel = isMarginal ? "$K$ unrefined (marginal)" : "$K^{\\text{ref}}$ IS-chain"
                    let kernelClass = isMarginal ? "marginal" : "refined"

                    let hjStr = hjKs.map(String.init).joined(separator: ", ")
                    let muted = "$(p, q) = (\(p), \(q))$ &nbsp; HJ: $[\(hjStr)]$"

                    let seriesLatex = Self.formatRationalQSeries(rawSeries)

                    // Build index notation: I_{(name)_{gamma}}
                    let manifoldName = self.ncCycleData.isEmpty ? "M" : "\\mathrm{\(params["name"] as? String ?? "M")}"
                    let indexNotation = "\\mathcal{I}_{(\(manifoldName))_{\\gamma_\(cuspIdx)}}"

                    results.append([
                        "title": "NC Cycle \(cycleIdx):  $\\gamma_\(cuspIdx) = \(gammaLabel)$",
                        "kernelLabel": kernelLabel,
                        "kernelClass": kernelClass,
                        "muted": muted,
                        "indexNotation": indexNotation,
                        "series": seriesLatex,
                        "edgeToggles": [],
                    ])
                }
            }

            let data: [String: Any] = [
                "manifoldName": "manifold",
                "results": results,
            ]

            if let json = try? JSONSerialization.data(withJSONObject: data),
               let jsonStr = String(data: json, encoding: .utf8) {
                self.sendJSON("updateFillingResults", jsonStr)
            }
        }
    }

    // MARK: - Copy / Export

    private func handleCopyLatex(_ params: [String: Any]) {
        if let latex = params["latex"] as? String {
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.setString(latex, forType: .string)
        }
    }

    private func handleExport(_ params: [String: Any]) {
        NSLog("Bridge: export(%@)", "\(params)")
        // TODO: file save dialog
    }

    // MARK: - Formatting helpers

    /// Build LaTeX for the NZ matrix from doubled-integer values.
    /// nzG is a flat array of (2n+2r) × n values (row-major, ×2).
    static func buildNZLatex(nzG: [Any], n: Int, r: Int) -> String {
        // g_NZ is 2n × 2n (row-major, doubled values)
        let rows = 2 * n
        let cols = 2 * n
        guard nzG.count == rows * cols else {
            return "$$g_{\\mathrm{NZ}} \\text{ (format error: expected \\(\(rows)×\(cols)=\(rows*cols)\\) values, got \\(\(nzG.count)\\))}$$"
        }

        var latex = "$$g_{\\mathrm{NZ}} = \\begin{pmatrix} "
        for i in 0..<rows {
            if i > 0 { latex += " \\\\ " }
            for j in 0..<cols {
                if j > 0 { latex += " & " }
                let val = (nzG[i * cols + j] as? Int) ?? (nzG[i * cols + j] as? Double).map { Int($0) } ?? 0
                if val % 2 == 0 {
                    latex += "\(val / 2)"
                } else if val > 0 {
                    latex += "\\tfrac{\(val)}{2}"
                } else {
                    latex += "-\\tfrac{\(-val)}{2}"
                }
            }
        }
        latex += " \\end{pmatrix}$$"
        return latex
    }

    /// Format an integer vector as (a, b, c).
    static func formatVector(_ v: [Any]) -> String {
        let vals = v.map { "\($0)" }
        return "(" + vals.joined(separator: ", ") + ")"
    }

    /// Format a doubled-integer vector as half-integers: (0, 1/2, -1).
    static func formatHalfVector(_ v: [Any]) -> String {
        let vals: [String] = v.map { raw in
            let val = (raw as? Int) ?? (raw as? Double).map { Int($0) } ?? 0
            if val % 2 == 0 {
                return "\(val / 2)"
            } else if val > 0 {
                return "\\tfrac{\(val)}{2}"
            } else {
                return "-\\tfrac{\(-val)}{2}"
            }
        }
        return "(" + vals.joined(separator: ", ") + ")"
    }

    /// Format a sparse q-series {key: coeff} as LaTeX.
    /// Keys are JSON-encoded vectors like "[0]", "[1]", etc.
    static func formatQSeries(_ series: [String: Int], qqOrder: Int) -> String {
        // Parse keys as power arrays, sort by total power
        var terms: [(power: [Int], coeff: Int)] = []
        for (key, coeff) in series {
            if let data = key.data(using: .utf8),
               let arr = try? JSONSerialization.jsonObject(with: data) as? [Int] {
                terms.append((arr, coeff))
            }
        }
        terms.sort { a, b in
            let sa = a.power.reduce(0, +)
            let sb = b.power.reduce(0, +)
            return sa < sb
        }

        if terms.isEmpty { return "0" }

        var latex = ""
        let maxTerms = 6
        for (i, term) in terms.prefix(maxTerms).enumerated() {
            let c = term.coeff
            let p = term.power.first ?? 0

            if i > 0 {
                if c > 0 { latex += " + " } else { latex += " - " }
            } else if c < 0 {
                latex += "-"
            }

            let ac = abs(c)
            if p == 0 {
                latex += "\(ac)"
            } else {
                if ac != 1 { latex += "\(ac)" }
                latex += "q"
                if p != 1 {
                    // Handle half-integer powers
                    if p % 2 == 0 {
                        latex += "^{\(p / 2)}"
                    } else {
                        latex += "^{\(p)/2}"
                    }
                }
            }
        }

        if terms.count > maxTerms {
            latex += " + \\cdots"
        }

        return latex
    }

    /// Format a rational q-series (from filled index) with (numer, denom) pairs.
    static func formatRationalQSeries(_ series: [String: [Int]]) -> String {
        var terms: [(power: [Int], numer: Int, denom: Int)] = []
        for (key, nd) in series {
            if let data = key.data(using: .utf8),
               let arr = try? JSONSerialization.jsonObject(with: data) as? [Int],
               nd.count == 2 {
                terms.append((arr, nd[0], nd[1]))
            }
        }
        terms.sort { a, b in
            let sa = a.power.reduce(0, +)
            let sb = b.power.reduce(0, +)
            return sa < sb
        }

        if terms.isEmpty { return "0" }

        var latex = ""
        let maxTerms = 6
        for (i, term) in terms.prefix(maxTerms).enumerated() {
            let n = term.numer
            let d = term.denom
            let p = term.power.first ?? 0

            if i > 0 {
                if n > 0 { latex += " + " } else { latex += " - " }
            } else if n < 0 {
                latex += "-"
            }

            let an = abs(n)
            let coeffStr: String
            if d == 1 {
                coeffStr = an == 1 && p != 0 ? "" : "\(an)"
            } else {
                coeffStr = "\\tfrac{\(an)}{\(d)}"
            }

            if p == 0 {
                latex += coeffStr.isEmpty ? "1" : coeffStr
            } else {
                latex += coeffStr
                latex += "q"
                if p != 1 {
                    if p % 2 == 0 {
                        latex += "^{\(p / 2)}"
                    } else {
                        latex += "^{\(p)/2}"
                    }
                }
            }
        }

        if terms.count > maxTerms {
            latex += " + \\cdots"
        }

        return latex
    }

    /// Build a slope label like "2\,\alpha_0 + \beta_0" from (p, q).
    static func slopeLabel(p: Int, q: Int, cusp: Int, component: String) -> String {
        let alpha = "\\alpha_\(cusp)"
        let beta = "\\beta_\(cusp)"

        if q == 0 {
            return coeffStr(p) + alpha
        }
        if p == 0 {
            return coeffStr(q) + beta
        }

        let first = coeffStr(p) + alpha
        let second: String
        if q > 0 {
            second = " + " + coeffStr(q) + beta
        } else {
            second = " - " + coeffStr(-q) + beta
        }
        return first + second
    }

    private static func coeffStr(_ c: Int) -> String {
        switch c {
        case 1: return ""
        case -1: return "-"
        default: return "\(c)\\,"
        }
    }
}
