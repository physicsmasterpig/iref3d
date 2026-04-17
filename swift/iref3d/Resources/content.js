// ── iref3d content.js — WebView logic + Swift bridge ──
// All state is tracked here so we can reset cleanly on manifold change.

"use strict";

// ── App state ──
var appState = {
    manifoldName: null,
    n: 0,
    r: 0,
    hard: 0,
    stage: "empty",   // empty -> loaded -> indexed -> nc -> filled
};

// ── Active tasks (for cancellation) ──
var _activeTasks = {};

// ── Bridge: post messages to Swift ──
function postToSwift(action, params) {
    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.bridge) {
        window.webkit.messageHandlers.bridge.postMessage({ action: action, params: params || {} });
    } else {
        console.log("Bridge (no Swift):", action, params);
    }
}

// ── Progress helpers ──
function showProgress(id, label) {
    _activeTasks[id] = true;
    var el = document.getElementById(id + "-progress");
    if (!el) {
        // Create progress row after the button area
        el = document.createElement("div");
        el.id = id + "-progress";
        el.className = "progress-row";
        var container = document.getElementById(id + "-progress-slot");
        if (container) container.appendChild(el);
    }
    el.innerHTML =
        '<div class="progress-bar"><div class="bar"><div class="bar-fill"></div></div>' +
        '<span class="progress-label">' + (label || "Computing...") + '</span></div>' +
        '<button class="btn-stop" onclick="cancelTask(\'' + id + '\')">Stop</button>';
    el.style.display = "flex";
    // Disable the compute button
    var btn = document.getElementById("btn-" + id);
    if (btn) btn.disabled = true;
}

function hideProgress(id) {
    delete _activeTasks[id];
    var el = document.getElementById(id + "-progress");
    if (el) el.style.display = "none";
    var btn = document.getElementById("btn-" + id);
    if (btn) btn.disabled = false;
}

function cancelTask(id) {
    delete _activeTasks[id];
    hideProgress(id);
    postToSwift("cancel", { task: id });
}

function isTaskActive(id) {
    return !!_activeTasks[id];
}

// ── KaTeX: render all math in an element ──
function renderMath(el) {
    if (typeof renderMathInElement !== "function") {
        console.warn("renderMathInElement not loaded yet");
        return;
    }
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
    var sec = document.getElementById("sec-" + name);
    if (sec) sec.classList.toggle("collapsed");
}

// ── Pipeline state management ──

function resetToStage(stage) {
    // Reset everything at or above the given stage
    if (stage === "empty" || stage === "loaded") {
        // Clear index results
        document.getElementById("index-results").innerHTML = "";
        document.getElementById("index-status").textContent = "ready";
    }
    if (stage === "empty" || stage === "loaded" || stage === "indexed") {
        // Clear NC cycles + filling
        document.getElementById("cusp-blocks").innerHTML = "";
        document.getElementById("filling-results").innerHTML = "";
        document.getElementById("nc-summary").textContent = "";
        document.getElementById("compute-filling-row").style.display = "none";
        document.getElementById("filling-sep").style.display = "none";
        document.getElementById("filling-status").textContent = "needs index data";
    }
    if (stage === "nc") {
        // Clear only filling results
        document.getElementById("filling-results").innerHTML = "";
        document.getElementById("filling-sep").style.display = "none";
    }
    appState.stage = stage;
}

function updateSectionOpacity() {
    var s = appState.stage;
    // Index section: available if manifold is loaded
    var secIndex = document.getElementById("sec-index");
    var secFilling = document.getElementById("sec-filling");
    var secExport = document.getElementById("sec-export");

    if (s === "empty") {
        secIndex.style.opacity = "0.5";
        secIndex.style.pointerEvents = "none";
        secFilling.style.opacity = "0.5";
        secFilling.style.pointerEvents = "none";
        secExport.style.opacity = "0.5";
        secExport.style.pointerEvents = "none";
    } else if (s === "loaded") {
        secIndex.style.opacity = "1";
        secIndex.style.pointerEvents = "auto";
        secFilling.style.opacity = "0.5";
        secFilling.style.pointerEvents = "none";
        secExport.style.opacity = "0.5";
        secExport.style.pointerEvents = "none";
    } else if (s === "indexed") {
        secIndex.style.opacity = "1";
        secIndex.style.pointerEvents = "auto";
        secFilling.style.opacity = "1";
        secFilling.style.pointerEvents = "auto";
        secExport.style.opacity = "1";
        secExport.style.pointerEvents = "auto";
        document.getElementById("filling-status").textContent = "ready";
    } else {
        // nc, filled — everything active
        secIndex.style.opacity = "1";
        secIndex.style.pointerEvents = "auto";
        secFilling.style.opacity = "1";
        secFilling.style.pointerEvents = "auto";
        secExport.style.opacity = "1";
        secExport.style.pointerEvents = "auto";
    }
}

// ── Called by Swift when page loads ──
function onReady() {
    renderMath();
    updateQueryCount();
    updateSectionOpacity();
}

// ════════════════════════════════════════════════
// MANIFOLD
// ════════════════════════════════════════════════

function doLoadManifold() {
    var name = document.getElementById("inp-name").value.trim();
    var nMax = parseInt(document.getElementById("inp-nmax").value) || 10;
    if (!name) return;
    showProgress("load-manifold", "Loading " + name + "...");
    postToSwift("loadManifold", { name: name, nMax: nMax });
}

// Called by Swift with manifold data
function updateManifold(data) {
    hideProgress("load-manifold");
    // data = { name, n, r, hard, easy, nzLatex, nuX, nuP }

    // Reset all downstream state
    resetToStage("loaded");

    // Update app state
    appState.manifoldName = data.name;
    appState.n = data.n;
    appState.r = data.r;
    appState.hard = data.hard || 0;

    // Update manifold section
    document.getElementById("manifold-status").textContent =
        data.name + " \u00B7 n = " + data.n + ", r = " + data.r;
    document.getElementById("inp-name").value = data.name;

    var info = document.getElementById("manifold-info");
    info.style.display = "";

    // Info bar
    var bar = document.getElementById("manifold-info-bar");
    bar.innerHTML = [
        infoItem("Tetrahedra", data.n),
        infoItem("Cusps", data.r),
        infoItem("Hard edges", data.hard || 0),
        infoItem("Easy edges", data.easy || 0),
    ].join("");

    // NZ matrix
    var nzEl = document.getElementById("nz-matrix");
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

    // Update section states
    updateSectionOpacity();
    renderMath();
}

function infoItem(label, value) {
    return '<div class="info-item"><span class="info-label">' + label +
           '</span><span class="info-value">' + value + '</span></div>';
}

function buildEdgeToggles(numHard) {
    var el = document.getElementById("edge-toggles-index");
    var html = "";
    for (var i = 0; i < numHard; i++) {
        html += '<label style="font-size:13px"><input type="checkbox" checked class="edge-toggle" data-idx="' + i + '"> $W_' + i + '$</label>';
    }
    el.innerHTML = html;
    renderMath(el);
}

// ════════════════════════════════════════════════
// INDEX
// ════════════════════════════════════════════════

function updateQueryCount() {
    var mLo = parseInt(document.getElementById("inp-m-lo").value) || 0;
    var mHi = parseInt(document.getElementById("inp-m-hi").value) || 0;
    var eLo = parseFloat(document.getElementById("inp-e-lo").value) || 0;
    var eHi = parseFloat(document.getElementById("inp-e-hi").value) || 0;
    var eStep = parseFloat(document.getElementById("inp-e-lo").step) || 0.5;
    var mCount = mHi - mLo + 1;
    var eCount = Math.round((eHi - eLo) / eStep) + 1;
    var total = Math.max(0, mCount * eCount);
    var el = document.getElementById("query-count");
    el.textContent = total + " queries ($" + mCount + "\\,m \\times " + eCount + "\\,e$)";
    renderMath(el);
}

function doComputeIndex() {
    if (appState.stage === "empty") return;
    document.getElementById("index-results").innerHTML = "";

    var params = {
        name: appState.manifoldName,
        mLo: parseInt(document.getElementById("inp-m-lo").value),
        mHi: parseInt(document.getElementById("inp-m-hi").value),
        eLo: parseFloat(document.getElementById("inp-e-lo").value),
        eHi: parseFloat(document.getElementById("inp-e-hi").value),
        mode: (document.querySelector('input[name="mode"]:checked') || {}).value || "grid",
        eta: document.getElementById("sel-eta").value,
    };
    showProgress("compute-index", "Computing refined index...");
    postToSwift("computeIndex", params);
}

// Called by Swift with index results
function updateIndexResults(data) {
    hideProgress("compute-index");
    if (!isTaskActive("compute-index") && _activeTasks["compute-index"] === undefined) {
        // Not cancelled — proceed
    }
    var entries = data.entries || [];
    var r = data.r || 1;

    appState.stage = "indexed";

    document.getElementById("index-status").textContent = entries.length + " entries computed";

    var colSpan = r > 1 ? 8 : 5;
    var totalCols = r > 1 ? 12 : 9;
    var html = '<table class="st"><thead><tr><th>#</th><th colspan="' + colSpan + '"></th><th>Series</th><th>Source</th><th></th></tr></thead><tbody>';

    for (var i = 0; i < entries.length; i++) {
        var e = entries[i];
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
    }

    if (entries.length > 6) {
        html += '<tr><td colspan="' + totalCols + '" style="text-align:center; color:#8b949e; font-size:11px; padding:6px;">Showing all ' + entries.length + ' entries</td></tr>';
    }
    html += '</tbody></table>';

    document.getElementById("index-results").innerHTML = html;
    updateSectionOpacity();
    renderMath(document.getElementById("index-results"));
}

function formatChargeColumns(charges, r) {
    if (!charges) return '';
    var html = '';
    for (var ci = 0; ci < charges.length; ci++) {
        var c = charges[ci];
        if (ci > 0) html += '<td class="sym">$,$</td>';
        html += '<td class="al">$' + formatCoeff(c[0]) + '\\,\\alpha_' + ci + '$</td>';
        html += '<td class="bl">$' + formatSign(c[1]) + '\\; ' + formatCoeffAbs(c[1]) + '\\,\\beta_' + ci + '$</td>';
    }
    return html;
}

function formatCoeff(v) {
    if (v === 0) return '0';
    if (v === 1) return '';
    if (v === -1) return '-';
    if (Number.isInteger(v)) return String(v);
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

// ════════════════════════════════════════════════
// NC CYCLES
// ════════════════════════════════════════════════

function doFindNC() {
    if (appState.stage === "empty" || appState.stage === "loaded") return;
    resetToStage("nc");

    var params = {
        name: appState.manifoldName,
        pMax: parseInt(document.getElementById("inp-nc-p").value),
        qMax: parseInt(document.getElementById("inp-nc-q").value),
        useCache: document.getElementById("chk-nc-cache").checked,
    };
    showProgress("find-nc", "Searching NC cycles...");
    postToSwift("findNCCycles", params);
}

// Called by Swift with NC cycle data
function updateNCCycles(data) {
    hideProgress("find-nc");
    var cusps = data.cusps || [];
    appState.stage = "nc";

    document.getElementById("nc-summary").textContent =
        cusps.map(function(c) { return c.cycles.length; }).join(" + ") + " cycles \u00B7 " + (data.elapsed || "?") + " s";

    var html = "";
    for (var ci = 0; ci < cusps.length; ci++) {
        var cusp = cusps[ci];
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

        for (var ri = 0; ri < cusp.cycles.length; ri++) {
            var cyc = cusp.cycles[ri];
            var marginal = cyc.marginal;
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
        }

        html += '</table></div>';
    }

    document.getElementById("cusp-blocks").innerHTML = html;
    document.getElementById("compute-filling-row").style.display = "";
    updateFillingButtonCount();
    updateSectionOpacity();
    renderMath(document.getElementById("cusp-blocks"));
}

function toggleAllNC(cuspIdx, master) {
    var cbs = document.querySelectorAll(".nc-check-" + cuspIdx);
    for (var i = 0; i < cbs.length; i++) cbs[i].checked = master.checked;
    updateFillingButtonCount();
}

function updateCuspMode(cuspIdx) {
    // Future: show/hide slope vs charge inputs
}

function updateFillingButtonCount() {
    var total = 0;
    var cbs = document.querySelectorAll("[class^='nc-check-']");
    for (var i = 0; i < cbs.length; i++) {
        if (cbs[i].checked) total++;
    }
    var btn = document.getElementById("btn-compute-filling");
    if (btn) btn.textContent = "Compute filled index (" + total + " selected)";
}

// ════════════════════════════════════════════════
// FILLING
// ════════════════════════════════════════════════

function doComputeFilling() {
    if (appState.stage !== "nc" && appState.stage !== "filled") return;
    showProgress("compute-filling", "Computing filled index...");

    var cuspBlocks = document.querySelectorAll(".cusp-block");
    var cusps = [];
    for (var ci = 0; ci < cuspBlocks.length; ci++) {
        var modeEl = document.querySelector('input[name="cusp' + ci + '-mode"]:checked');
        var mode = modeEl ? modeEl.value : "fill";
        var p = parseInt((document.getElementById("slope-p-" + ci) || {}).value) || 0;
        var q = parseInt((document.getElementById("slope-q-" + ci) || {}).value) || 0;
        var selected = [];
        var checks = document.querySelectorAll(".nc-check-" + ci);
        for (var j = 0; j < checks.length; j++) {
            if (checks[j].checked) selected.push(parseInt(checks[j].dataset.row));
        }
        cusps.push({ cuspIdx: ci, mode: mode, p: p, q: q, selectedCycles: selected });
    }
    postToSwift("computeFilling", { name: appState.manifoldName, cusps: cusps });
}

// Called by Swift with filling results
function updateFillingResults(data) {
    hideProgress("compute-filling");
    var results = data.results || [];
    appState.stage = "filled";

    document.getElementById("filling-sep").style.display = "";
    document.getElementById("filling-status").textContent =
        results.length + " results \u00B7 filled";

    var html = '<h3>2. Results</h3>';

    for (var i = 0; i < results.length; i++) {
        var res = results[i];
        var cls = res.kernelClass === "marginal" ? "marginal" : "refined";
        html += '<div class="result-card ' + cls + '">';
        html += '<div style="display:flex; align-items:center; gap:8px;">';
        html += '<b>' + res.title + '</b>';
        html += '<small style="color:' + (cls === "refined" ? "#2ea043" : "#d4880a") + '">' + res.kernelLabel + '</small>';
        html += '<span style="flex:1"></span>';
        if (res.edgeToggles && res.edgeToggles.length > 0) {
            html += '<span style="font-size:11px; color:#86868b; margin-right:4px;">Edges:</span>';
            for (var j = 0; j < res.edgeToggles.length; j++) {
                var e = res.edgeToggles[j];
                if (e.disabled) {
                    html += '<label style="font-size:13px; margin-left:4px"><input type="checkbox" disabled> <span style="color:#c7c7cc">$W_' + e.idx + '$</span></label>';
                } else {
                    html += '<label style="font-size:13px; margin-left:4px"><input type="checkbox" checked style="accent-color:#007aff"> $W_' + e.idx + '$</label>';
                }
            }
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
    }

    document.getElementById("filling-results").innerHTML = html;
    updateSectionOpacity();
    renderMath(document.getElementById("filling-results"));
}

// ════════════════════════════════════════════════
// EXPORT / COPY
// ════════════════════════════════════════════════

function doExport() {
    postToSwift("export", { format: "all" });
}

function doCopyLatex() {
    postToSwift("copyLatex", { latex: "TODO" });
}

function doCopyMathematica() {
    postToSwift("export", { format: "mathematica" });
}

function copyRow(idx) {
    postToSwift("copyLatex", { rowIndex: idx });
}

function removeRow(idx) {
    var rows = document.querySelectorAll("#index-results .st tbody tr");
    if (rows[idx]) rows[idx].remove();
}

// ════════════════════════════════════════════════
// INPUT LISTENERS
// ════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", function() {
    ["inp-m-lo", "inp-m-hi", "inp-e-lo", "inp-e-hi"].forEach(function(id) {
        var el = document.getElementById(id);
        if (el) el.addEventListener("input", updateQueryCount);
    });

    // Enter key in name field triggers load
    var nameField = document.getElementById("inp-name");
    if (nameField) {
        nameField.addEventListener("keydown", function(ev) {
            if (ev.key === "Enter") doLoadManifold();
        });
    }

    // Delegate checkbox changes for NC cycle count
    document.addEventListener("change", function(ev) {
        if (ev.target.className && typeof ev.target.className === "string" && ev.target.className.indexOf("nc-check-") === 0) {
            updateFillingButtonCount();
        }
    });

    renderMath();
    updateQueryCount();
    updateSectionOpacity();
});

// ════════════════════════════════════════════════
// TEST HARNESS — callable from Swift or browser console
// ════════════════════════════════════════════════

var _testLog = [];

function _testAssert(cond, msg) {
    if (!cond) {
        _testLog.push("FAIL: " + msg);
        console.error("FAIL: " + msg);
    } else {
        _testLog.push("PASS: " + msg);
    }
}

function runTests() {
    _testLog = [];
    var pass = 0, fail = 0;

    // ── Test 1: initial state ──
    _testAssert(appState.stage === "empty" || appState.stage === "loaded", "T1: initial stage is empty or loaded");
    _testAssert(document.getElementById("index-results").innerHTML === "", "T1: index results empty initially");
    _testAssert(document.getElementById("cusp-blocks").innerHTML === "", "T1: cusp blocks empty initially");
    _testAssert(document.getElementById("filling-results").innerHTML === "", "T1: filling results empty initially");

    // ── Test 2: load manifold m006 ──
    updateManifold({ name: "m006", n: 3, r: 1, hard: 1, easy: 1,
        nzLatex: "$$g=1$$", nuX: "(0,0,-2)", nuP: "(0,0,0)" });
    _testAssert(appState.stage === "loaded", "T2: stage is loaded after updateManifold");
    _testAssert(appState.manifoldName === "m006", "T2: manifoldName is m006");
    _testAssert(appState.r === 1, "T2: r is 1");
    _testAssert(document.getElementById("manifold-info").style.display === "", "T2: manifold info visible");
    _testAssert(document.getElementById("index-results").innerHTML === "", "T2: index results still empty");
    _testAssert(document.getElementById("sec-index").style.opacity === "1", "T2: index section enabled");
    _testAssert(document.getElementById("sec-filling").style.opacity === "0.5", "T2: filling section disabled");

    // ── Test 3: compute index ──
    updateIndexResults({ r: 1, entries: [
        { charges: [[0, 0]], series: "1+q", source: "computed" },
        { charges: [[0.5, 0]], series: "\\eta^{W_0}", source: "computed" },
    ]});
    _testAssert(appState.stage === "indexed", "T3: stage is indexed after updateIndexResults");
    _testAssert(document.getElementById("index-results").innerHTML !== "", "T3: index results populated");
    _testAssert(document.getElementById("sec-filling").style.opacity === "1", "T3: filling section now enabled");
    _testAssert(document.getElementById("cusp-blocks").innerHTML === "", "T3: cusp blocks still empty");

    // ── Test 4: find NC cycles ──
    updateNCCycles({ cusps: [
        { cuspIdx: 0, cycles: [
            { gamma: "\\alpha_0", delta: "\\beta_0", weyl: "(0,0)", q1proj: "-1", marginal: false },
            { gamma: "2\\alpha_0+\\beta_0", delta: "-\\alpha_0", weyl: "(0,0)", q1proj: "0", marginal: true },
        ]}
    ], elapsed: "0.1" });
    _testAssert(appState.stage === "nc", "T4: stage is nc");
    _testAssert(document.getElementById("cusp-blocks").innerHTML !== "", "T4: cusp blocks populated");
    _testAssert(document.querySelectorAll(".nc-check-0").length === 2, "T4: 2 NC cycle checkboxes");
    _testAssert(document.getElementById("compute-filling-row").style.display === "", "T4: compute filling button visible");
    _testAssert(document.getElementById("filling-results").innerHTML === "", "T4: filling results still empty");

    // ── Test 5: compute filling ──
    updateFillingResults({ manifoldName: "m006", results: [
        { title: "NC Cycle 0", kernelLabel: "Kref", kernelClass: "refined",
          muted: "test", indexNotation: "\\mathcal{I}_{m006}", series: "1+q",
          edgeToggles: [{ idx: 0, disabled: false }] },
    ]});
    _testAssert(appState.stage === "filled", "T5: stage is filled");
    _testAssert(document.getElementById("filling-results").innerHTML !== "", "T5: filling results populated");
    _testAssert(document.querySelectorAll(".result-card").length === 1, "T5: 1 result card");

    // ── Test 6: CHANGE manifold — everything must reset ──
    updateManifold({ name: "s776", n: 5, r: 2, hard: 2, easy: 1,
        nzLatex: "$$g=2$$", nuX: "(0,0)", nuP: "(0,0)" });
    _testAssert(appState.stage === "loaded", "T6: stage reset to loaded on manifold change");
    _testAssert(appState.manifoldName === "s776", "T6: manifoldName changed to s776");
    _testAssert(appState.r === 2, "T6: r changed to 2");
    _testAssert(document.getElementById("index-results").innerHTML === "", "T6: index results cleared");
    _testAssert(document.getElementById("cusp-blocks").innerHTML === "", "T6: cusp blocks cleared");
    _testAssert(document.getElementById("filling-results").innerHTML === "", "T6: filling results cleared");
    _testAssert(document.getElementById("nc-summary").textContent === "", "T6: nc summary cleared");
    _testAssert(document.getElementById("compute-filling-row").style.display === "none", "T6: compute filling button hidden");
    _testAssert(document.getElementById("filling-sep").style.display === "none", "T6: filling separator hidden");
    _testAssert(document.getElementById("sec-filling").style.opacity === "0.5", "T6: filling section disabled again");

    // ── Test 7: new compute cycle on s776 ──
    updateIndexResults({ r: 2, entries: [
        { charges: [[0, 0], [0, 0]], series: "1+2q", source: "computed" },
    ]});
    _testAssert(appState.stage === "indexed", "T7: indexed on s776");
    _testAssert(document.getElementById("sec-filling").style.opacity === "1", "T7: filling enabled");

    updateNCCycles({ cusps: [
        { cuspIdx: 0, cycles: [{ gamma: "\\alpha_0", delta: "\\beta_0", weyl: "(0,0)", q1proj: "-1", marginal: false }] },
        { cuspIdx: 1, cycles: [{ gamma: "\\alpha_1", delta: "\\beta_1", weyl: "(0,0)", q1proj: "-1", marginal: false }] },
    ], elapsed: "0.5" });
    _testAssert(appState.stage === "nc", "T7: nc on s776");
    _testAssert(document.querySelectorAll(".cusp-block").length === 2, "T7: 2 cusp blocks for r=2");

    // ── Test 8: re-compute index must NOT clear NC + filling (they depend on NZ, not m/e range) ──
    updateNCCycles({ cusps: [
        { cuspIdx: 0, cycles: [{ gamma: "\\alpha_0", delta: "\\beta_0", weyl: "(0,0)", q1proj: "-1", marginal: false }] },
    ], elapsed: "0.1" });
    updateFillingResults({ manifoldName: "s776", results: [
        { title: "x", kernelLabel: "k", kernelClass: "refined", muted: "", indexNotation: "I", series: "1", edgeToggles: [] },
    ]});
    updateIndexResults({ r: 2, entries: [
        { charges: [[0, 0], [0, 0]], series: "1+3q", source: "computed" },
    ]});
    _testAssert(document.getElementById("cusp-blocks").innerHTML !== "", "T8: cusp blocks preserved after re-index");
    _testAssert(document.getElementById("filling-results").innerHTML !== "", "T8: filling preserved after re-index");

    // ── Summarize ──
    for (var i = 0; i < _testLog.length; i++) {
        if (_testLog[i].indexOf("FAIL") === 0) fail++;
        else pass++;
    }

    var summary = "Tests: " + pass + " passed, " + fail + " failed, " + _testLog.length + " total";
    console.log("=== " + summary + " ===");
    for (var j = 0; j < _testLog.length; j++) console.log(_testLog[j]);

    // Reset to clean state after tests
    resetToStage("empty");
    appState.manifoldName = null;
    document.getElementById("manifold-info").style.display = "none";
    document.getElementById("manifold-status").textContent = "no manifold loaded";
    updateSectionOpacity();

    return { pass: pass, fail: fail, log: _testLog };
}
