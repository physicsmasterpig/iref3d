import Foundation

/// Swift wrapper around the Rust C-ABI FFI.
/// All Rust functions exchange JSON as C strings.
enum RustFFI {

    /// Call a Rust FFI function with JSON input, return parsed JSON.
    private static func call(
        _ fn: (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?,
        args: [String: Any]
    ) -> [String: Any] {
        guard let jsonData = try? JSONSerialization.data(withJSONObject: args),
              let jsonStr = String(data: jsonData, encoding: .utf8) else {
            return ["error": "failed to serialize args"]
        }

        guard let resultPtr = jsonStr.withCString({ fn($0) }) else {
            return ["error": "null result from Rust"]
        }

        let resultStr = String(cString: resultPtr)
        iref3d_free_string(resultPtr)

        guard let data = resultStr.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ["error": "failed to parse Rust JSON: \(resultStr.prefix(200))"]
        }

        return obj
    }

    // MARK: - Public API

    static func loadManifold(dbPath: String, name: String) -> [String: Any] {
        call(iref3d_load_manifold, args: ["db_path": dbPath, "name": name])
    }

    static func computeIndex(
        mLo: Int, mHi: Int,
        eLoX2: Int, eHiX2: Int, eStepX2: Int,
        qqOrder: Int
    ) -> [String: Any] {
        call(iref3d_compute_index, args: [
            "m_lo": mLo, "m_hi": mHi,
            "e_lo_x2": eLoX2, "e_hi_x2": eHiX2, "e_step_x2": eStepX2,
            "qq_order": qqOrder
        ])
    }

    static func findNCCycles(
        cuspIdx: Int,
        pMin: Int, pMax: Int,
        qMin: Int, qMax: Int,
        qqOrder: Int
    ) -> [String: Any] {
        call(iref3d_find_nc_cycles, args: [
            "cusp_idx": cuspIdx,
            "p_min": pMin, "p_max": pMax,
            "q_min": qMin, "q_max": qMax,
            "qq_order": qqOrder
        ])
    }

    static func ncCompat(
        cuspIdx: Int, p: Int, q: Int, qqOrder: Int
    ) -> [String: Any] {
        call(iref3d_nc_compat, args: [
            "cusp_idx": cuspIdx, "p": p, "q": q, "qq_order": qqOrder
        ])
    }

    static func filledRefinedIndex(
        cuspIdx: Int, p: Int, q: Int, qqOrder: Int,
        incompatEdges: [Int] = []
    ) -> [String: Any] {
        call(iref3d_filled_refined_index, args: [
            "cusp_idx": cuspIdx, "p": p, "q": q,
            "qq_order": qqOrder, "incompat_edges": incompatEdges
        ])
    }
}
