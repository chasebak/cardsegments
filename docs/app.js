/* Buy List Explorer. The page computes nothing: every number is looked up
   in docs/data/explorer.json (written by `python -m cardsegments.viz build`
   from the committed synthetic results) and only formatted here. Colors come
   from tokens.css at render time, so light and dark mode need no second
   palette; charts re-render when the scheme flips. The page refuses to show
   a bundle or signal block that is not labelled SYNTHETIC. */
(function () {
  "use strict";

  var state = { data: null, signal: null, setting: 0, index: 0, charts: {} };

  function byId(id) { return document.getElementById(id); }

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function pct(value, digits) {
    if (value === null || value === undefined) { return "–"; }
    return (value * 100).toFixed(digits) + "%";
  }

  function pp(value, digits) {
    if (value === null || value === undefined) { return "–"; }
    var text = (value * 100).toFixed(digits);
    return (value >= 0 ? "+" : "") + text + " pp";
  }

  function signedNum(value, digits) {
    if (value === null || value === undefined) { return "–"; }
    return (value >= 0 ? "+" : "") + value.toFixed(digits);
  }

  function num(value, digits) {
    if (value === null || value === undefined) { return "–"; }
    return value.toFixed(digits);
  }

  function count(value) { return Number(value).toLocaleString("en-US"); }

  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) { node.textContent = text; }
    if (className) { node.className = className; }
    return node;
  }

  function fillTable(table, headers, rows, numericColumns) {
    table.textContent = "";
    var thead = el("thead");
    var headRow = el("tr");
    headers.forEach(function (header, index) {
      var th = el("th", header);
      if (numericColumns && numericColumns.indexOf(index) !== -1) { th.className = "num"; }
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function (row) {
      var tr = el("tr");
      if (row.className) { tr.className = row.className; }
      row.cells.forEach(function (cell, index) {
        var td = el("td", cell);
        if (numericColumns && numericColumns.indexOf(index) !== -1) { td.className = "num"; }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  function signalBlock() { return state.data.signals[state.signal]; }
  function settingMeta() { return state.data.settings[state.setting]; }
  function settingBlock() { return signalBlock().settings[state.setting]; }
  function modelName(index) { return state.data.model_order[index]; }
  function threshold() { return state.data.thresholds[state.index]; }

  function isLabelled(data) {
    if (!data || typeof data.label !== "string" || data.label.indexOf("SYNTHETIC") !== 0) { return false; }
    return data.signal_order.every(function (key) {
      var block = data.signals[key];
      return Boolean(block && block.label);
    });
  }

  /* ---------- static text and controls ---------- */

  function renderStatic() {
    var data = state.data;
    byId("banner").textContent = data.label;
    document.querySelectorAll("[data-prov]").forEach(function (node) {
      node.textContent = data.provenance[node.getAttribute("data-prov")] || "";
    });
    document.querySelectorAll("[data-why]").forEach(function (node) {
      var why = data.why[node.getAttribute("data-why")];
      node.textContent = "";
      node.appendChild(el("b", "Why this metric"));
      node.appendChild(el("p", "Owner's question: " + why.question));
      node.appendChild(el("p", "Kind of number: " + why.kind));
      node.appendChild(el("p", "What would make it lie: " + why.lies));
    });
    byId("footer-text").textContent = data.footer;
    var slider = byId("threshold");
    slider.max = String(data.thresholds.length - 1);
  }

  function renderButtons() {
    var data = state.data;
    var signalButtons = byId("signal-buttons");
    signalButtons.textContent = "";
    data.signal_order.forEach(function (key) {
      var button = el("button", data.signals[key].label);
      button.type = "button";
      button.setAttribute("aria-pressed", key === state.signal ? "true" : "false");
      button.addEventListener("click", function () {
        state.signal = key;
        renderAll();
      });
      signalButtons.appendChild(button);
    });
    var costButtons = byId("cost-buttons");
    costButtons.textContent = "";
    data.settings.forEach(function (setting, index) {
      var button = el("button", setting.name);
      button.type = "button";
      button.setAttribute("aria-pressed", index === state.setting ? "true" : "false");
      button.addEventListener("click", function () {
        state.setting = index;
        state.index = setting.nearest_index;
        byId("threshold").value = String(state.index);
        renderAll();
      });
      costButtons.appendChild(button);
    });
  }

  function renderTicks() {
    var data = state.data;
    var ticks = byId("ticks");
    ticks.textContent = "";
    var last = data.thresholds.length - 1;
    var tStarIndex = settingMeta().nearest_index;
    data.thresholds.forEach(function (value, index) {
      var stop = el("span", null, "stop");
      if (index === 0) { stop.classList.add("first"); }
      if (index === last) { stop.classList.add("last"); }
      if (index === tStarIndex) { stop.classList.add("tstar"); }
      if (index === state.index) { stop.classList.add("on"); }
      stop.style.left = (index / last) * 100 + "%";
      stop.appendChild(el("span", null, "mark"));
      var labelled = index === 0 || index === last || (index + 1) % 10 === 0;
      if (labelled) {
        stop.classList.add("labelled");
        stop.appendChild(document.createTextNode(num(value, 2)));
      } else if (index === tStarIndex) {
        stop.appendChild(document.createTextNode("t*"));
      }
      ticks.appendChild(stop);
    });
  }

  /* ---------- tiles (exact t* cell) ---------- */

  function renderTiles() {
    var signal = signalBlock();
    var meta = settingMeta();
    var setting = settingBlock();
    var model = setting.models[modelName(0)];
    var velocity = setting.models[modelName(1)];
    var labels = state.data.model_labels;
    document.querySelectorAll("[data-tstar]").forEach(function (node) {
      node.textContent = meta.t_star_text + " (m " + num(meta.m, 2) + ", h " + num(meta.h, 2) + ")";
    });
    var delta = setting.delta_parked;
    byId("tile-parked").textContent = pct(model.parked, 1) + " vs " + pct(velocity.parked, 1);
    byId("tile-parked-note").textContent =
      "Δ " + pp(delta.value, 2) + ", 95% CI [" + pp(delta.ci_low, 2) + ", " + pp(delta.ci_high, 2) +
      "]; positive = the full logistic's list parks less. Baseline: " + labels[modelName(1)] +
      ", same folds, same t*. " + signal.short + ".";
    byId("tile-missed").textContent = count(model.fn) + " vs " + count(velocity.fn);
    byId("tile-missed-note").textContent =
      pct(model.missed_share, 1) + " vs " + pct(velocity.missed_share, 1) + " of the " +
      count(signal.n_sellers) + " cards that sold; lists of " + count(model.n_flagged) + " and " +
      count(velocity.n_flagged) + " cards.";
    byId("tile-regret").textContent = pct(model.regret_share, 2) + " vs " + pct(velocity.regret_share, 2);
    byId("tile-regret-note").textContent =
      "cost at t* " + num(model.cost_t_star, 4) + " vs " + num(velocity.cost_t_star, 4) +
      " per card; hindsight-best grid t = " + num(model.best_grid_threshold, 2) + " and " +
      num(velocity.best_grid_threshold, 2) + " (illustrative units).";
    var auc = signal.delta_auc;
    byId("tile-auc").textContent = signedNum(auc.value, 4);
    byId("tile-auc-note").textContent =
      "95% CI [" + signedNum(auc.ci_low, 4) + ", " + signedNum(auc.ci_high, 4) + "]; beats velocity " +
      "(CI above 0 and Δ ≥ +0.01): " + (signal.auc_beats_velocity ? "yes" : "no") + ".";
    byId("context-note").textContent =
      "Context, not a baseline: buying every card parks " + pct(signal.buy_everything_parked, 1) +
      " of them; buying none costs " + num(model.buy_none_cost, 4) + " per card, and the full " +
      "logistic at t* costs " + pct(model.saving_vs_buy_none, 1) + " less. " + signal.label + ", n = " +
      count(signal.n_rows) + " cards in " + count(signal.n_groups) + " set groups, base rate " +
      num(signal.base_rate, 3) + ".";
  }

  /* ---------- the two buy lists at the slider threshold ---------- */

  function gridCell(name) {
    var columns = signalBlock().grid[name];
    var i = state.index;
    return {
      tp: columns.tp[i], fp: columns.fp[i], fn: columns.fn[i], tn: columns.tn[i],
      n_flagged: columns.n_flagged[i], parked: columns.parked[i], missed_share: columns.missed_share[i]
    };
  }

  function matrixTable(cell) {
    var table = el("table", null, "matrix");
    var thead = el("thead");
    var head = el("tr");
    head.appendChild(el("th", ""));
    head.appendChild(el("th", "sold ≥ 5 (liquid)", "num"));
    head.appendChild(el("th", "sat (not liquid)", "num"));
    thead.appendChild(head);
    table.appendChild(thead);
    var tbody = el("tbody");
    var onList = el("tr");
    onList.appendChild(el("th", "on the list"));
    onList.appendChild(el("td", count(cell.tp) + " sold", "num sold"));
    onList.appendChild(el("td", count(cell.fp) + " parked", "num parked"));
    tbody.appendChild(onList);
    var passed = el("tr");
    passed.appendChild(el("th", "passed"));
    passed.appendChild(el("td", count(cell.fn) + " missed", "num missed"));
    passed.appendChild(el("td", count(cell.tn) + " skipped", "num"));
    tbody.appendChild(passed);
    table.appendChild(tbody);
    return table;
  }

  function renderLists() {
    var container = byId("lists");
    container.textContent = "";
    var labels = state.data.model_labels;
    var signal = signalBlock();
    state.data.model_order.forEach(function (name, position) {
      var cell = gridCell(name);
      var card = el("div", null, "list-card" + (position === 0 ? " model" : ""));
      card.appendChild(el("h3", labels[name] + (position === 0 ? "" : " (baseline)")));
      card.appendChild(el("div", "parked " + pct(cell.parked, 1), "big"));
      card.appendChild(el("p",
        count(cell.n_flagged) + " cards on the list; " + count(cell.fn) + " missed sellers (" +
        pct(cell.missed_share, 1) + " of the " + count(signal.n_sellers) + " that sold)", "sub"));
      var wrap = el("div", null, "table-wrap");
      wrap.appendChild(matrixTable(cell));
      card.appendChild(wrap);
      container.appendChild(card);
    });
    var text = num(threshold(), 2);
    byId("t-now").textContent = text;
    document.querySelectorAll("[data-t-now]").forEach(function (node) { node.textContent = text; });
    byId("threshold").setAttribute("aria-valuetext", "t = " + text);
  }

  function renderGridTable() {
    var signal = signalBlock();
    var setting = settingBlock();
    var first = modelName(0);
    var second = modelName(1);
    var rows = state.data.thresholds.map(function (value, i) {
      return {
        className: i === state.index ? "chosen" : "",
        cells: [num(value, 2),
          count(signal.grid[first].n_flagged[i]), pct(signal.grid[first].parked[i], 1),
          pct(signal.grid[first].missed_share[i], 1), num(setting.models[first].cost_curve[i], 4),
          count(signal.grid[second].n_flagged[i]), pct(signal.grid[second].parked[i], 1),
          pct(signal.grid[second].missed_share[i], 1), num(setting.models[second].cost_curve[i], 4)]
      };
    });
    fillTable(byId("grid-table"),
      ["t", "model list", "model parked", "model missed", "model cost",
        "velocity list", "velocity parked", "velocity missed", "velocity cost"],
      rows, [1, 2, 3, 4, 5, 6, 7, 8]);
  }

  function renderCostNote() {
    var setting = settingBlock();
    var meta = settingMeta();
    var model = setting.models[modelName(0)];
    var velocity = setting.models[modelName(1)];
    byId("cost-note").textContent =
      meta.name + ": cost per card = ((m + h) × parked + m × missed) / n. At t* the full logistic costs " +
      num(model.cost_t_star, 4) + " (cheapest grid point " + num(model.best_grid_cost, 4) + " at t = " +
      num(model.best_grid_threshold, 2) + ") and velocity " + num(velocity.cost_t_star, 4) +
      " (cheapest " + num(velocity.best_grid_cost, 4) + " at t = " + num(velocity.best_grid_threshold, 2) + ").";
  }

  /* ---------- charts ---------- */

  function chartsAvailable() { return typeof window.Chart === "function"; }

  function showFallbacks() {
    document.querySelectorAll("[data-fallback]").forEach(function (node) { node.hidden = false; });
    document.querySelectorAll("[data-fallback-details]").forEach(function (node) { node.open = true; });
    ["curves-box", "cost-box"].forEach(function (id) { byId(id).hidden = true; });
  }

  /* Inline plugin (no Chart.js plugin library): vertical lines at the
     slider threshold and at t*. */
  var markerPlugin = {
    id: "thresholdMarkers",
    afterDatasetsDraw: function (chart) {
      var scale = chart.scales.x;
      var area = chart.chartArea;
      var ctx = chart.ctx;
      var lines = [
        { x: settingMeta().t_star, color: token("--warn"), dash: [5, 4], width: 2 },
        { x: threshold(), color: token("--accent"), dash: [], width: 2 }
      ];
      ctx.save();
      lines.forEach(function (line) {
        var px = scale.getPixelForValue(line.x);
        ctx.beginPath();
        ctx.setLineDash(line.dash);
        ctx.lineWidth = line.width;
        ctx.strokeStyle = line.color;
        ctx.moveTo(px, area.top);
        ctx.lineTo(px, area.bottom);
        ctx.stroke();
      });
      ctx.restore();
    }
  };

  function baseOptions(yTitle) {
    var text = token("--text");
    var muted = token("--text-muted");
    var border = token("--border");
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: { legend: { labels: { color: text, boxWidth: 12 } }, tooltip: { enabled: true } },
      scales: {
        x: { type: "linear", min: 0, max: 1, ticks: { color: muted }, grid: { color: border },
          title: { display: true, text: "buy threshold t", color: muted } },
        y: { ticks: { color: muted }, grid: { color: border }, title: { display: true, text: yTitle, color: muted } }
      }
    };
  }

  function points(values) {
    return state.data.thresholds.map(function (t, i) { return { x: t, y: values[i] }; });
  }

  function curvesDatasets() {
    var signal = signalBlock();
    var first = signal.grid[modelName(0)];
    var second = signal.grid[modelName(1)];
    var labels = state.data.model_labels;
    var accent = token("--series-1");
    var base = token("--series-2");
    return [
      { label: labels[modelName(0)] + ": parked", data: points(first.parked), borderColor: accent,
        backgroundColor: accent, pointRadius: 0, borderWidth: 2 },
      { label: labels[modelName(1)] + ": parked", data: points(second.parked), borderColor: base,
        backgroundColor: base, pointRadius: 0, borderWidth: 2 },
      { label: labels[modelName(0)] + ": missed sellers", data: points(first.missed_share), borderColor: accent,
        backgroundColor: accent, pointRadius: 0, borderWidth: 2, borderDash: [6, 4] },
      { label: labels[modelName(1)] + ": missed sellers", data: points(second.missed_share), borderColor: base,
        backgroundColor: base, pointRadius: 0, borderWidth: 2, borderDash: [6, 4] }
    ];
  }

  function buildCurves() {
    var options = baseOptions("share");
    options.scales.y.min = 0;
    options.scales.y.max = 1;
    options.scales.y.ticks.callback = function (value) { return pct(value, 0); };
    options.plugins.tooltip.callbacks = {
      title: function (items) { return items.length ? "t = " + num(items[0].parsed.x, 2) : ""; },
      label: function (context) { return context.dataset.label + ": " + pct(context.parsed.y, 1); }
    };
    state.charts.curves = new window.Chart(byId("curves-chart"), {
      type: "line", data: { datasets: curvesDatasets() }, options: options, plugins: [markerPlugin]
    });
  }

  function costDatasets() {
    var setting = settingBlock();
    var first = setting.models[modelName(0)];
    var second = setting.models[modelName(1)];
    var labels = state.data.model_labels;
    var accent = token("--series-1");
    var base = token("--series-2");
    return [
      { label: labels[modelName(0)], data: points(first.cost_curve), borderColor: accent,
        backgroundColor: accent, pointRadius: 0, borderWidth: 2 },
      { label: labels[modelName(1)], data: points(second.cost_curve), borderColor: base,
        backgroundColor: base, pointRadius: 0, borderWidth: 2 },
      { label: "cheapest grid point", type: "scatter",
        data: [{ x: first.best_grid_threshold, y: first.best_grid_cost },
          { x: second.best_grid_threshold, y: second.best_grid_cost }],
        borderColor: token("--text"), backgroundColor: token("--surface"), pointRadius: 5, borderWidth: 2 }
    ];
  }

  function buildCost() {
    var options = baseOptions("cost per card (illustrative units)");
    options.scales.y.beginAtZero = true;
    options.scales.y.ticks.callback = function (value) { return num(value, 3); };
    options.plugins.tooltip.callbacks = {
      title: function (items) { return items.length ? "t = " + num(items[0].parsed.x, 2) : ""; },
      label: function (context) { return context.dataset.label + ": " + num(context.parsed.y, 4); }
    };
    state.charts.cost = new window.Chart(byId("cost-chart"), {
      type: "line", data: { datasets: costDatasets() }, options: options, plugins: [markerPlugin]
    });
  }

  function destroyCharts() {
    Object.keys(state.charts).forEach(function (key) { state.charts[key].destroy(); });
    state.charts = {};
  }

  function buildCharts() {
    if (!chartsAvailable()) { showFallbacks(); return; }
    destroyCharts();
    buildCurves();
    buildCost();
  }

  function updateCharts() {
    if (!chartsAvailable() || !state.charts.curves) { return; }
    state.charts.curves.data.datasets = curvesDatasets();
    state.charts.curves.update();
    state.charts.cost.data.datasets = costDatasets();
    state.charts.cost.update();
  }

  /* ---------- wiring ---------- */

  function renderThreshold() {
    renderTicks();
    renderLists();
    renderGridTable();
    if (chartsAvailable() && state.charts.curves) {
      state.charts.curves.draw();
      state.charts.cost.draw();
    }
  }

  function renderAll() {
    renderButtons();
    renderTiles();
    renderCostNote();
    renderThreshold();
    updateCharts();
  }

  function loadJson(path) {
    return fetch(path).then(function (response) {
      if (!response.ok) { throw new Error(path + " " + response.status); }
      return response.json();
    });
  }

  function start() {
    loadJson("data/explorer.json")
      .then(function (data) {
        if (!isLabelled(data)) {
          byId("label-error").hidden = false;
          byId("controls").hidden = true;
          document.querySelectorAll("main > section").forEach(function (node) { node.hidden = true; });
          return;
        }
        state.data = data;
        state.signal = data.default_signal;
        state.setting = data.default_setting;
        state.index = data.settings[state.setting].nearest_index;
        renderStatic();
        byId("threshold").value = String(state.index);
        buildCharts();
        renderAll();
        byId("threshold").addEventListener("input", function (event) {
          state.index = Number(event.target.value);
          renderThreshold();
        });
        if (window.matchMedia) {
          var scheme = window.matchMedia("(prefers-color-scheme: dark)");
          var rebuild = function () { buildCharts(); };
          if (scheme.addEventListener) { scheme.addEventListener("change", rebuild); }
          else if (scheme.addListener) { scheme.addListener(rebuild); }
        }
      })
      .catch(function () { byId("load-error").hidden = false; });
  }

  start();
})();
