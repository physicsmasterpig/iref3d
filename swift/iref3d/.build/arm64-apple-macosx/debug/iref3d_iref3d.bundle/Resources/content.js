// ── iref3d content.js — WebView logic + Swift bridge ──

// ── Bridge: post messages to Swift ──
function postToSwift(action, params) {
    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.bridge) {
        window.webkit.messageHandlers.bridge.postMessage({ action, params: params || {} });
    } else {
        console.log("Bridge (no Swift):", action, params);
    }
}

// ── KaTeX: render all math in an element ──
function renderMath(el) {
    if (!el) el = document.body;
    renderMathInElement(el, {
        delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false }
        ],
        throwOnError: false
    });
}

// ── Section collapse/expand ──
function toggleSection(name) {
    const sec = document.getElementById("sec-" + name);
    if (sec) sec.classList.toggle("collapsed");
}

// ── Called by Swift when page loads ──
function onReady() {
    renderMath();
    updateQueryCount();
}

// ── Manifold ──

function doLoadManifold() {
    const name = document.getElementById("inp-name").value.trim();
    const nMax = parseInt(document.getElementById("inp-nmax").value) || 10;
    if (!name) return;
    postToSwift("loadManifold", { name, nMax });
}

// Called by Swift with manifold data
function updateManifold(data) {
    // data = { name, n, r, hard, easy, nzLatex, nuX, nuP }
    document.getElementById("manifold-status").textContent =
        data.name + " · n = " + data.n + ", r = " + data.r;
    document.getElementById("inp-name").value = data.name;

    const info = document.getElementById("manifold-info");
    info.style.display = "";

    // Info bar
    const bar = document.getElementById("manifold-info-bar");
    bar.innerHTML = [
        infoItem("Tetrahedra", data.n),
        infoItem("Cusps", data.r),
        infoItem("Hard edges", data.hard || 0),
        infoItem("Easy edges", data.easy || 0),
    ].join("");

    // NZ matrix
    const nzEl = document.getElementById("nz-matrix");
    if (data.nzLatex) {
        nzEl.innerHTML = "<p>" + data.nzLatex + "</p>";
        if (data.nuX) {
            nzEl.innerHTML += '<p>$\\nu_x = ' + data.nuX + '$, &nbsp; $\\nu_p = ' + data.nuP + '$</p>';
        }
    } else {
        nzEl.innerHTML = '<p class="muted">NZ matrix not available.</p>';
    }

    // Build edge toggles for index section
    buildEdgeToggles(data.hard || 0);

    // Update filling status
    document.getElementById("index-status").textContent = "ready";
    document.getElementById("filling-status").textContent = "needs index data";

    renderMath();
}

function infoItem(label, value) {
    return '<div class="info-item"><span class="info-label">' + label +
           '</span><span class="info-value">' + value + '</span></div>';
}

function buildEdgeToggles(numHard) {
    const el = document.getElementById("edge-toggles-index");
    let html = "";
    for (let i = 0; i < numHard; i++) {
        html += '<label style="font-size:13px"><input type="checkbox" checked class="edge-toggle" data-idx="' + i + '"> $W_' + i + '$</label>';
    }
    el.innerHTML = html;
    renderMath(el);
}

// ── Index ──

function updateQueryCount() {
    const mLo = parseInt(document.getElementById("inp-m-lo").value) || 0;
    const mHi = parseInt(document.getElementById("inp-m-hi").value) || 0;
    const eLo = parseFloat(document.getElementById("inp-e-lo").value) || 0;
    const eHi = parseFloat(document.getElementById("inp-e-hi").value) || 0;
    const eStep = parseFloat(document.getElementById("inp-e-lo").step) || 0.5;
    const mCount = mHi - mLo + 1;
    const eCount = Math.round((eHi - eLo) / eStep) + 1;
    const total = Math.max(0, mCount * eCount);
    document.getElementById("query-count").textContent =
        total + " queries ($" + mCount + "\\,m \\times " + eCount + "\\,e$)";
    renderMath(document.getElementById("query-count"));
}

function doComputeIndex() {
    const params = {
        mLo: parseInt(document.getElementById("inp-m-lo").value),
        mHi: parseInt(document.getElementById("inp-m-hi").value),
        eLo: parseFloat(document.getElementById("inp-e-lo").value),
        eHi: parseFloat(document.getElementById("inp-e-hi").value),
        mode: document.querySelector('input[name="mode"]:checked')?.value || "grid",
        eta: document.getElementById("sel-eta").value,
    };
    postToSwift("computeIndex", params);
}

// Called by Swift with index results
// entries = [{ m, e, series, source, etaPrefix }]
function updateIndexResults(data) {
    const entries = data.entries || [];
    const r = data.r || 1;
    document.getElementById("index-status").textContent = entries.length + " entries computed";

    let html = '<table class="st"><thead><tr><th>#</th><th colspan="' + (r > 1 ? 8 : 5) + '"></th><th>Series</th><th>Source</th><th></th></tr></thead><tbody>';

    entries.forEach((e, i) => {
        html += '<tr>';
        html += '<td class="ic">' + i + '</td>';
        html += '<td class="i">$\\mathcal{I}($</td>';
        html += formatChargeColumns(e.charges, r);
        html += '<td class="cp">$)$</td>';
        html += '<td class="eq">$=$</td>';
        html += '<td class="sr">$' + e.series + '$</td>';
        html += '<td class="vc">' + (e.source || "computed") + '</td>';
        html += '<td class="ac"><button class="a" onclick="copyRow(' + i + ')" title="Copy LaTeX">&#x29C9;</button><button class="a r" onclick="removeRow(' + i + ')" title="Remove">&#x2715;</button></td>';
        html += '</tr>';
    });

    if (entries.length > 6) {
        html += '<tr><td colspan="' + (r > 1 ? 12 : 9) + '" style="text-align:center; color:#8b949e; font-size:11px; padding:6px;">Showing all ' + entries.length + ' entries</td></tr>';
    }
    html += '</tbody></table>';

    document.getElementById("index-results").innerHTML = html;
    document.getElementById("filling-status").textContent = "ready";
    renderMath(document.getElementById("index-results"));
}

function formatChargeColumns(charges, r) {
    // charges = [[aCoeff, bCoeff], ...] for each cusp
    if (!charges) return '';
    let html = '';
    charges.forEach((c, ci) => {
        if (ci > 0) html += '<td class="sym">$,$</td>';
        html += '<td class="al">$' + formatCoeff(c[0]) + '\\,\\alpha_' + ci + '$</td>';
        html += '<td class="bl">$' + formatSign(c[1]) + '\\; ' + formatCoeffAbs(c[1]) + '\\,\\beta_' + ci + '$</td>';
    });
    return html;
}

function formatCoeff(v) {
    if (v === 0) return '0';
    if (v === 1) return '';
    if (v === -1) return '-';
    if (Number.isInteger(v)) return String(v);
    // Fraction: try halves
    if (v === 0.5) return '\\tfrac{1}{2}';
    if (v === -0.5) return '-\\tfrac{1}{2}';
    return String(v);
}

function formatCoeffAbs(v) {
    return formatCoeff(Math.abs(v));
}

function formatSign(v) {
    return v < 0 ? '-' : '+';
}

// ── NC Cycles ──

function doFindNC() {
    const params = {
        pMax: parseInt(document.getElementById("inp-nc-p").value),
        qMax: parseInt(document.getElementById("inp-nc-q").value),
        useCache: document.getElementById("chk-nc-cache").checked,
    };
    postToSwift("findNCCycles", params);
}

// Called by Swift with NC cycle data
// data = { cusps: [{ cuspIdx, cycles: [{ gamma, delta, weyl, q1proj, kernel, marginal }] }], elapsed }
function updateNCCycles(data) {
    const cusps = data.cusps || [];
    const totalCycles = cusps.reduce((s, c) => s + c.cycles.length, 0);
    document.getElementById("nc-summary").textContent =
        cusps.map(c => c.cycles.length).join(" + ") + " cycles · " + (data.elapsed || "?") + " s";

    let html = "";
    cusps.forEach((cusp, ci) => {
        html += '<div class="cusp-block">';
        html += '<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">';
        html += '<div class="cusp-title" style="margin-bottom:0">Cusp ' + ci + '</div>';
        html += '<div class="radio-group" style="font-size:13px;">';
        html += '<label><input type="radio" name="cusp' + ci + '-mode" value="fill" checked onchange="updateCuspMode(' + ci + ')"> Fill</label>';
        html += '<label><input type="radio" name="cusp' + ci + '-mode" value="charge" onchange="updateCuspMode(' + ci + ')"> Charge $(m,\\, e)$</label>';
        html += '</div>';
        html += '<span style="flex:1"></span>';
        html += '<span class="form-label">Slope</span>';
        html += '<span class="form-label">$P$</span> <input type="number" value="1" style="width:50px" id="slope-p-' + ci + '">';
        html += '<span class="form-label">$Q$</span> <input type="number" value="0" style="width:50px" id="slope-q-' + ci + '">';
        html += '</div>';

        html += '<table class="nc" style="font-size:0.95em;">';
        html += '<tr>';
        html += '<th style="width:1px"><input type="checkbox" checked style="accent-color:#007aff" onchange="toggleAllNC(' + ci + ', this)"></th>';
        html += '<th>#</th>';
        html += '<th>$\\gamma_' + ci + '$</th><th>$\\delta_' + ci + '$</th>';
        html += '<th>Weyl $(a,\\, b)$</th>';
        html += '<th>$\\left.\\textrm{Coeff}_{q^1}\\right|_{\\textrm{adj}\\,\\mathfrak{su}(2)}$</th>';
        html += '<th>Kernel</th>';
        html += '</tr>';

        cusp.cycles.forEach((cyc, ri) => {
            const marginal = cyc.marginal;
            html += '<tr>';
            html += '<td><input type="checkbox" ' + (marginal ? '' : 'checked') + ' style="accent-color:#007aff" class="nc-check-' + ci + '" data-row="' + ri + '"></td>';
            html += '<td><b>' + ri + '</b></td>';
            html += '<td>$' + cyc.gamma + '$</td>';
            html += '<td>$' + cyc.delta + '$</td>';
            html += '<td>$' + cyc.weyl + '$</td>';
            if (marginal) {
                html += '<td><span style="white-space:nowrap">$\\color{#f85149}{' + cyc.q1proj + '}$&nbsp;<b style="color:#f85149">Marginal</b></span></td>';
                html += '<td><small style="color:#d4880a">$K$ (unref.)</small></td>';
            } else {
                html += '<td><span style="white-space:nowrap">$\\color{#3fb950}{' + cyc.q1proj + '}$</span></td>';
                html += '<td><small style="color:#2ea043">$K^{\\text{ref}}$</small></td>';
            }
            html += '</tr>';
        });

        html += '</table></div>';
    });

    document.getElementById("cusp-blocks").innerHTML = html;
    document.getElementById("compute-filling-row").style.display = "";
    updateFillingButtonCount();
    renderMath(document.getElementById("cusp-blocks"));
}

function toggleAllNC(cuspIdx, master) {
    document.querySelectorAll(".nc-check-" + cuspIdx).forEach(cb => {
        cb.checked = master.checked;
    });
    updateFillingButtonCount();
}

function updateCuspMode(cuspIdx) {
    // Could show/hide slope inputs vs charge inputs
}

function updateFillingButtonCount() {
    let total = 0;
    document.querySelectorAll("[class^='nc-check-']").forEach(cb => {
        if (cb.checked) total++;
    });
    document.getElementById("btn-compute-filling").textContent =
        "Compute filled index (" + total + " selected)";
}

// ── Filling ──

function doComputeFilling() {
    // Gather selected cycles per cusp
    const cuspBlocks = document.querySelectorAll(".cusp-block");
    const cusps = [];
    cuspBlocks.forEach((block, ci) => {
        const mode = document.querySelector('input[name="cusp' + ci + '-mode"]:checked')?.value || "fill";
        const p = parseInt(document.getElementById("slope-p-" + ci)?.value) || 0;
        const q = parseInt(document.getElementById("slope-q-" + ci)?.value) || 0;
        const selected = [];
        document.querySelectorAll(".nc-check-" + ci).forEach(cb => {
            if (cb.checked) selected.push(parseInt(cb.dataset.row));
        });
        cusps.push({ cuspIdx: ci, mode, p, q, selectedCycles: selected });
    });
    postToSwift("computeFilling", { cusps });
}

// Called by Swift with filling results
// data = { manifoldName, results: [{ title, kernelLabel, kernelClass, muted, series, edgeToggles }] }
function updateFillingResults(data) {
    const results = data.results || [];
    document.getElementById("filling-sep").style.display = "";
    document.getElementById("filling-status").textContent =
        results.length + " results · filled";

    let html = '<h3>2. Results</h3>';

    results.forEach((res, i) => {
        const cls = res.kernelClass === "marginal" ? "marginal" : "refined";
        html += '<div class="result-card ' + cls + '">';
        html += '<div style="display:flex; align-items:center; gap:8px;">';
        html += '<b>' + res.title + '</b>';
        html += '<small style="color:' + (cls === "refined" ? "#2ea043" : "#d4880a") + '">' + res.kernelLabel + '</small>';
        html += '<span style="flex:1"></span>';
        if (res.edgeToggles && res.edgeToggles.length > 0) {
            html += '<span style="font-size:11px; color:#86868b; margin-right:4px;">Edges:</span>';
            res.edgeToggles.forEach(e => {
                if (e.disabled) {
                    html += '<label style="font-size:13px; margin-left:4px"><input type="checkbox" disabled> <span style="color:#c7c7cc">$W_' + e.idx + '$</span></label>';
                } else {
                    html += '<label style="font-size:13px; margin-left:4px"><input type="checkbox" checked style="accent-color:#007aff"> $W_' + e.idx + '$</label>';
                }
            });
        } else {
            html += '<span style="font-size:11px; color:#c7c7cc;">no $\\eta$ variables</span>';
        }
        html += '</div>';
        if (res.muted) {
            html += '<p class="muted" style="margin:2px 0;">' + res.muted + '</p>';
        }
        html += '<table class="st" style="margin:4px 0;"><tbody><tr>';
        html += '<td class="i">$' + res.indexNotation + '$</td>';
        html += '<td class="eq">$=$</td>';
        html += '<td class="sr">$' + res.series + '$</td>';
        html += '</tr></tbody></table>';
        html += '</div>';
    });

    document.getElementById("filling-results").innerHTML = html;
    renderMath(document.getElementById("filling-results"));
}

// ── Export ──

function doExport() {
    postToSwift("export", { format: "all" });
}

function doCopyLatex() {
    postToSwift("copyLatex", { latex: "TODO" });
}

function doCopyMathematica() {
    postToSwift("export", { format: "mathematica" });
}

// ── Utilities ──

function copyRow(idx) {
    postToSwift("copyLatex", { rowIndex: idx });
}

function removeRow(idx) {
    // Remove from DOM for now
    const rows = document.querySelectorAll("#index-results .st tbody tr");
    if (rows[idx]) rows[idx].remove();
}

// ── Input listeners for query count ──
document.addEventListener("DOMContentLoaded", function() {
    ["inp-m-lo", "inp-m-hi", "inp-e-lo", "inp-e-hi"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("input", updateQueryCount);
    });

    // Checkbox listeners for filling button count
    document.addEventListener("change", function(e) {
        if (e.target.className && e.target.className.startsWith && e.target.className.startsWith("nc-check-")) {
            updateFillingButtonCount();
        }
    });

    renderMath();
    updateQueryCount();
});
