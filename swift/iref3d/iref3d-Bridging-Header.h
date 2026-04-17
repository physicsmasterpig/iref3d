// iref3d-Bridging-Header.h
// C-ABI interface to the Rust iref3d-core static library.
// All functions exchange JSON as C strings.

#ifndef IREF3D_BRIDGING_HEADER_H
#define IREF3D_BRIDGING_HEADER_H

#include <stdint.h>

/// Free a string returned by any iref3d_* function.
void iref3d_free_string(char *ptr);

/// Load a manifold from the census database.
/// Input:  {"db_path": "...", "name": "m006"}
/// Output: {"name", "n", "r", "hard", "easy", "nz_g_x2", "nu_x", "nu_p_x2"}
char *iref3d_load_manifold(const char *json_in);

/// Compute refined index for a grid of (m, e) values.
/// Input:  {"m_lo", "m_hi", "e_lo_x2", "e_hi_x2", "e_step_x2", "qq_order"}
/// Output: {"entries": [...], "r"}
char *iref3d_compute_index(const char *json_in);

/// Find non-closable cycles for one cusp.
/// Input:  {"cusp_idx", "p_min", "p_max", "q_min", "q_max", "qq_order"}
/// Output: {"cycles": [{"p", "q"}], "slopes_tested"}
char *iref3d_find_nc_cycles(const char *json_in);

/// NC compatibility check for a single cycle.
/// Input:  {"cusp_idx", "p", "q", "qq_order"}
/// Output: {"ab_valid", "collapsed_edges", "all_weyl_symmetric", ...}
char *iref3d_nc_compat(const char *json_in);

/// Compute refined filled index (single cusp).
/// Input:  {"cusp_idx", "p", "q", "qq_order", "incompat_edges": []}
/// Output: {"series", "hj_ks", "has_cusp_eta", "qq_order", "n_kernel_terms"}
char *iref3d_filled_refined_index(const char *json_in);

#endif
