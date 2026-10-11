// =============================================================
// app.js — Client-Side Logic for Quantum Success Prediction UI
// Only predicts when the user clicks "Predict Success Probability"
// =============================================================

document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    algorithm: "deutsch_jozsa",
    n_qubits: 3,
    circuit_depth: 6,
    gate_count_1q: 8,
    gate_count_2q: 3,
    total_gate_count: 11,
    noise_type: "none",
    noise_strength: 0.0,
    hasPredicted: false,
  };

  // DOM Elements
  const algoBtns = document.querySelectorAll(".algo-btn");
  const noiseBtns = document.querySelectorAll(".noise-btn");
  
  
  const inputN = document.getElementById("input-n-qubits");
  const btnDecQubits = document.getElementById("btn-dec-qubits");
  const btnIncQubits = document.getElementById("btn-inc-qubits");
  
  const inputDepth = document.getElementById("input-circuit-depth");
  const inputGate1q = document.getElementById("input-gate-1q");
  const inputGate2q = document.getElementById("input-gate-2q");

  const metricWarning = document.getElementById("metric-warning");
  const metricWarningText = document.getElementById("metric-warning-text");

  const inputNoiseStrength = document.getElementById("input-noise-strength");
  const noiseRangePill = document.getElementById("noise-range-pill");

  const btnPredict = document.getElementById("btn-predict");
  const btnSimulate = document.getElementById("btn-simulate");
  
  // Results Elements
  const gaugeFill = document.getElementById("gauge-fill");
  const resProbPct = document.getElementById("res-prob-pct");
  const resFidelityBadge = document.getElementById("res-fidelity-badge");
  const resFidelityDesc = document.getElementById("res-fidelity-desc");
  const resConfidenceScore = document.getElementById("res-confidence-score");
  const resConfidenceBar = document.getElementById("res-confidence-bar");
  const resConfidencePill = document.getElementById("res-confidence-pill");
  const treeHistogram = document.getElementById("tree-histogram");

  // Simulation Elements
  const simCard = document.getElementById("simulation-result-card");
  const simProbPct = document.getElementById("sim-prob-pct");
  const simMlProbPct = document.getElementById("sim-ml-prob-pct");
  const simDeltaVal = document.getElementById("sim-delta-val");
  const simBitstringsList = document.getElementById("sim-bitstrings-list");

  // Modals
  const modalBenchmarks = document.getElementById("modal-benchmarks");
  const btnModelInfo = document.getElementById("btn-model-info");
  const btnCloseBenchmarks = document.getElementById("btn-close-benchmarks");

  const CIRCUMFERENCE = 2 * Math.PI * 82; // ~515.221

  // -------------------------------------------------------------
  // INITIALIZATION
  // -------------------------------------------------------------
  function init() {
    setupEventListeners();
    // Initialize default circuit metrics without predicting
    updateCircuitEstimate();
    resetResultsToAwaiting();
  }

  function resetResultsToAwaiting() {
    gaugeFill.style.strokeDashoffset = CIRCUMFERENCE;
    resProbPct.textContent = "--.-%";
    resFidelityBadge.textContent = "Awaiting Prediction";
    resFidelityBadge.style.color = "var(--text-muted)";
    resFidelityBadge.style.borderColor = "var(--border-subtle)";
    resFidelityBadge.style.background = "var(--bg-subtle)";
    resFidelityDesc.textContent = "Configure parameters on the left, then click 'Predict Success Probability'.";
    resConfidenceScore.textContent = "--.-%";
    resConfidenceScore.style.color = "var(--text-muted)";
    resConfidenceBar.style.width = "0%";
    resConfidencePill.textContent = "Awaiting Prediction";
    resConfidencePill.style.color = "var(--text-muted)";
    resConfidencePill.style.borderColor = "var(--border-subtle)";
    resConfidencePill.style.background = "var(--bg-subtle)";
    treeHistogram.innerHTML = '<div class="hist-empty-placeholder">Click Predict to compute tree spread</div>';
  }

  function markParametersChanged() {
    if (state.hasPredicted) {
      resFidelityBadge.textContent = "Parameters Modified";
      resFidelityBadge.style.color = "var(--color-primary)";
      resFidelityBadge.style.borderColor = "var(--color-primary)";
      resFidelityBadge.style.background = "var(--color-primary-light)";
      resFidelityDesc.textContent = "Inputs changed. Click 'Predict Success Probability' to calculate updated output.";
    }
  }

  // -------------------------------------------------------------
  // EVENT LISTENERS
  // -------------------------------------------------------------
  function setupEventListeners() {
    // Algorithm Buttons
    algoBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        algoBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.algorithm = btn.dataset.algo;

        updateCircuitEstimate();
      });
    });

    // Qubits Stepper Controls
    btnDecQubits.addEventListener("click", () => {
      let val = parseInt(inputN.value, 10) || 3;
      if (val > 2) {
        val -= 1;
        inputN.value = val;
        state.n_qubits = val;
        handleQubitChange();
      }
    });

    btnIncQubits.addEventListener("click", () => {
      let val = parseInt(inputN.value, 10) || 3;
      if (val < 8) {
        val += 1;
        inputN.value = val;
        state.n_qubits = val;
        handleQubitChange();
      }
    });

    inputN.addEventListener("change", () => {
      let val = parseInt(inputN.value, 10);
      if (isNaN(val)) val = 3;
      if (val < 2) val = 2;
      if (val > 8) val = 8;
      inputN.value = val;
      state.n_qubits = val;
      handleQubitChange();
    });

    // Manual Metric Inputs (when auto-derive is disabled)
    inputDepth.addEventListener("input", () => {
      state.circuit_depth = parseInt(inputDepth.value, 10) || 6;
      validateInputs();
      markParametersChanged();
    });

    inputGate1q.addEventListener("input", () => {
      state.gate_count_1q = parseInt(inputGate1q.value, 10) || 8;
      updateTotalGates();
      validateInputs();
      markParametersChanged();
    });

    inputGate2q.addEventListener("input", () => {
      state.gate_count_2q = parseInt(inputGate2q.value, 10) || 3;
      updateTotalGates();
      validateInputs();
      markParametersChanged();
    });

    // Noise Model Selection
    noiseBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        noiseBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.noise_type = btn.dataset.noise;

        configureNoiseControls(state.noise_type);
        validateInputs();
        markParametersChanged();
      });
    });

    // Noise Strength Input
    inputNoiseStrength.addEventListener("input", () => {
      let val = parseFloat(inputNoiseStrength.value);
      if (isNaN(val)) val = 0.0;
      state.noise_strength = val;
      validateInputs();
      markParametersChanged();
    });

    // Predict Button Click (ONLY entry point for prediction)
    btnPredict.addEventListener("click", () => {
      performPredictionCall();
    });

    // Simulate Button Click
    btnSimulate.addEventListener("click", () => {
      executeLiveSimulation();
    });

    // Modal controls
    btnModelInfo.addEventListener("click", () => modalBenchmarks.classList.remove("hidden"));
    btnCloseBenchmarks.addEventListener("click", () => modalBenchmarks.classList.add("hidden"));

    window.addEventListener("click", (e) => {
      if (e.target === modalBenchmarks) modalBenchmarks.classList.add("hidden");
    });
  }

  function handleQubitChange() {
    updateCircuitEstimate();
  }

  function updateTotalGates() {
    state.total_gate_count = state.gate_count_1q + state.gate_count_2q;
  }

  // Configure Noise controls per noise type
  function configureNoiseControls(noiseType, customVal = null) {
    if (noiseType === "none") {
      inputNoiseStrength.value = "0.000";
      inputNoiseStrength.readOnly = true;
      noiseRangePill.textContent = "Strength: — (Ideal)";
      state.noise_strength = 0.0;
    } else if (noiseType === "depolarizing") {
      inputNoiseStrength.min = "0.001";
      inputNoiseStrength.max = "0.100";
      inputNoiseStrength.step = "0.001";
      inputNoiseStrength.readOnly = false;
      const val = customVal !== null ? customVal : 0.010;
      inputNoiseStrength.value = val.toFixed(3);
      state.noise_strength = val;
      noiseRangePill.textContent = "p = 0.001 – 0.100";
    } else if (noiseType === "amplitude_damping") {
      inputNoiseStrength.min = "0.001";
      inputNoiseStrength.max = "0.100";
      inputNoiseStrength.step = "0.001";
      inputNoiseStrength.readOnly = false;
      const val = customVal !== null ? customVal : 0.010;
      inputNoiseStrength.value = val.toFixed(3);
      state.noise_strength = val;
      noiseRangePill.textContent = "γ = 0.001 – 0.100";
    } else if (noiseType === "thermal_relaxation") {
      inputNoiseStrength.min = "0.010";
      inputNoiseStrength.max = "0.100";
      inputNoiseStrength.step = "0.005";
      inputNoiseStrength.readOnly = false;
      const val = customVal !== null ? customVal : 0.033;
      inputNoiseStrength.value = val.toFixed(3);
      state.noise_strength = val;
      noiseRangePill.textContent = "T = 0.010 – 0.100";
    }
  }

  // Fetch Circuit Estimates (Does NOT trigger prediction)
  async function updateCircuitEstimate() {
    try {
      const res = await fetch(`/api/circuit_estimate?algorithm=${state.algorithm}&n_qubits=${state.n_qubits}`);
      const data = await res.json();
      if (data.status === "success" && data.metrics) {
        const m = data.metrics;
        state.circuit_depth = m.circuit_depth;
        state.gate_count_1q = m.gate_count_1q;
        state.gate_count_2q = m.gate_count_2q;
        state.total_gate_count = m.total_gate_count;

        inputDepth.value = m.circuit_depth;
        inputGate1q.value = m.gate_count_1q;
        inputGate2q.value = m.gate_count_2q;

        validateInputs();
        markParametersChanged();
      }
    } catch (err) {
      console.error("Error fetching circuit metrics:", err);
    }
  }

  // Circuit Constructibility Validation
  function validateInputs() {
    const errors = [];
    const warnings = [];

    const n = state.n_qubits;
    const depth = state.circuit_depth;
    const g1 = state.gate_count_1q;
    const g2 = state.gate_count_2q;
    const gTotal = g1 + g2;
    state.total_gate_count = gTotal;

    // Total physical qubits required (DJ/BV: n+1, Simon: 2n)
    const totalQubits = (state.algorithm === "simons") ? (2 * n) : (n + 1);

    // 1. Qubit bounds
    if (n < 2) {
      errors.push("At least 2 qubits required.");
    } else if (n > 8) {
      warnings.push(`Qubit count (${n}) exceeds calibrated range (2–8).`);
    }

    // 2. Minimal gate requirements to construct algorithm oracles
    let min1q = 0;
    let min2q = 0;
    if (state.algorithm === "deutsch_jozsa" || state.algorithm === "bernstein_vazirani") {
      min1q = 2 * n + 2; // Ancilla prep (X+H) + input initialization (nH) + measurement interference (nH)
      min2q = 1;         // Oracle requires at least 1 CNOT gate
    } else if (state.algorithm === "simons") {
      min1q = 2 * n;     // Input initialization (nH) + measurement interference (nH)
      min2q = n;         // Copying input register requires at least n CNOT gates
    }

    if (g1 < min1q) {
      errors.push(`Insufficient 1Q gates (${g1}): requires ≥ ${min1q} for state prep & interference.`);
    }
    if (g2 < min2q) {
      errors.push(`Insufficient 2Q gates (${g2}): requires ≥ ${min2q} CNOT gates to construct oracle.`);
    }

    // 3. Physical circuit depth & scheduling constraints
    if (depth < 1) {
      errors.push("Circuit depth must be at least 1.");
    } else {
      if (depth > gTotal + 2) {
        errors.push(`Depth (${depth}) cannot exceed total gates (${gTotal}).`);
      }
      const maxCapacity = depth * totalQubits;
      if (gTotal > maxCapacity) {
        errors.push(`Cannot pack ${gTotal} gates into depth ${depth} on ${totalQubits} qubits (max is ${maxCapacity}).`);
      }
      const max2qPerLayer = Math.max(1, Math.floor(totalQubits / 2));
      const minDepth2q = Math.ceil(g2 / max2qPerLayer);
      if (depth < minDepth2q) {
        errors.push(`Depth too shallow: ${g2} CNOTs on ${totalQubits} qubits need depth ≥ ${minDepth2q}, but depth is ${depth}.`);
      }
    }

    // 4. Update UI indicator
    if (errors.length > 0) {
      metricWarningText.textContent = "Cannot construct circuit: " + errors[0];
      metricWarning.classList.remove("hidden");
      metricWarning.classList.add("error-alert");
      btnPredict.disabled = true;
      btnPredict.style.opacity = "0.5";
      btnPredict.style.cursor = "not-allowed";
      return false;
    } else if (warnings.length > 0) {
      metricWarningText.textContent = warnings[0];
      metricWarning.classList.remove("hidden");
      metricWarning.classList.remove("error-alert");
      btnPredict.disabled = false;
      btnPredict.style.opacity = "1";
      btnPredict.style.cursor = "pointer";
      return true;
    } else {
      metricWarning.classList.add("hidden");
      metricWarning.classList.remove("error-alert");
      btnPredict.disabled = false;
      btnPredict.style.opacity = "1";
      btnPredict.style.cursor = "pointer";
      return true;
    }
  }

  // -------------------------------------------------------------
  // PREDICTION EXECUTION (Triggered ONLY on "Predict" button click)
  // -------------------------------------------------------------
  async function performPredictionCall() {
    if (!validateInputs()) {
      showToast("Parameters cannot construct a valid quantum circuit.");
      return;
    }
    btnPredict.textContent = "Calculating...";

    const payload = {
      algorithm: state.algorithm,
      n_qubits: state.n_qubits,
      circuit_depth: state.circuit_depth,
      gate_count_1q: state.gate_count_1q,
      gate_count_2q: state.gate_count_2q,
      total_gate_count: state.total_gate_count,
      noise_type: state.noise_type,
      noise_strength: state.noise_strength,
    };

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const res = await response.json();
      if (res.status === "success") {
        state.hasPredicted = true;
        renderPredictionResults(res);
      } else {
        showToast(res.message || "Prediction error");
      }
    } catch (err) {
      console.error("Prediction API failed:", err);
      showToast("Server connection error during prediction");
    } finally {
      btnPredict.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        Predict Success Probability
      `;
    }
  }

  // Render Prediction Results (Using Best Model: Decision Tree R² = 0.9693)
  function renderPredictionResults(res) {
    const prob = res.prediction.decision_tree_probability;
    const probPct = res.prediction.decision_tree_probability_pct;
    const conf = res.confidence;
    const fidelity = res.fidelity;
    const telemetry = res.circuit_telemetry;

    // 1. Gauge Animation
    animateGauge(prob, probPct);

    // 2. Fidelity Meta
    resFidelityBadge.textContent = fidelity.status;
    resFidelityBadge.style.color = fidelity.color;
    resFidelityBadge.style.borderColor = fidelity.color;
    resFidelityBadge.style.background = `${fidelity.color}15`;
    resFidelityDesc.textContent = fidelity.description;

    // 3. Confidence Metrics
    resConfidenceScore.textContent = `${conf.score.toFixed(1)}%`;
    resConfidenceScore.style.color = conf.color;
    resConfidenceBar.style.width = `${Math.min(100, Math.max(0, conf.score))}%`;
    resConfidenceBar.style.background = conf.color;
    resConfidencePill.textContent = conf.level;
    resConfidencePill.style.color = conf.color;
    resConfidencePill.style.borderColor = `${conf.color}50`;
    resConfidencePill.style.background = `${conf.color}15`;

    // 4. Distribution Histogram
    renderTreeHistogram(conf.distribution);
  }

  // Gauge SVG Animation
  function animateGauge(prob, probPct) {
    const offset = CIRCUMFERENCE * (1 - prob);
    gaugeFill.style.strokeDashoffset = offset;

    // Solid color shift based on probability
    if (prob >= 0.95) {
      gaugeFill.style.stroke = "#059669"; // Emerald
    } else if (prob >= 0.88) {
      gaugeFill.style.stroke = "#2563eb"; // Blue
    } else if (prob >= 0.70) {
      gaugeFill.style.stroke = "#d97706"; // Amber
    } else {
      gaugeFill.style.stroke = "#dc2626"; // Crimson
    }

    animateValue(resProbPct, parseFloat(resProbPct.textContent) || 0, probPct, 400, "%");
  }

  function animateValue(elem, start, end, duration, suffix = "") {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const current = start + (end - start) * progress;
      elem.textContent = `${current.toFixed(1)}${suffix}`;
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  }

  // Render Histogram
  function renderTreeHistogram(distribution) {
    if (!distribution || !distribution.length) return;
    treeHistogram.innerHTML = "";

    const maxCount = Math.max(...distribution.map(d => d.count), 1);

    distribution.forEach((bin) => {
      const col = document.createElement("div");
      col.className = "hist-bar-col";
      col.title = `Range: [${bin.bin_start} - ${bin.bin_end}] | Trees: ${bin.count} (${bin.percentage}%)`;

      const heightPct = (bin.count / maxCount) * 100;

      const bar = document.createElement("div");
      bar.className = "hist-bar";
      bar.style.height = `${Math.max(4, heightPct)}%`;

      if (bin.count === 0) {
        bar.style.opacity = "0.2";
      }

      col.appendChild(bar);
      treeHistogram.appendChild(col);
    });
  }

  // -------------------------------------------------------------
  // LIVE QISKIT SIMULATION
  // -------------------------------------------------------------
  async function executeLiveSimulation() {
    btnSimulate.textContent = "Simulating (4096 shots)...";

    const payload = {
      algorithm: state.algorithm,
      n_qubits: state.n_qubits,
      noise_type: state.noise_type,
      noise_strength: state.noise_strength,
    };

    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (data.status === "success") {
        simCard.classList.remove("hidden");
        const simPct = data.simulated_success_probability_pct;
        const currentMlProb = parseFloat(resProbPct.textContent) || 0;
        const delta = Math.abs(simPct - currentMlProb).toFixed(2);

        simProbPct.textContent = `${simPct.toFixed(2)}%`;
        simMlProbPct.textContent = `${currentMlProb.toFixed(2)}%`;
        simDeltaVal.textContent = `|Δ| = ${delta}%`;

        // Bitstrings
        simBitstringsList.innerHTML = "";
        data.top_measurements.forEach(item => {
          const card = document.createElement("div");
          card.className = "bitstring-card";
          card.innerHTML = `
            <span class="bit-state">|${item.state}⟩</span>
            <span class="bit-shots">${item.count} (${item.pct}%)</span>
          `;
          simBitstringsList.appendChild(card);
        });

        showToast("Live Qiskit Aer simulation complete!");
      } else {
        showToast(data.message || "Simulation error");
      }
    } catch (err) {
      console.error("Simulation error:", err);
      showToast("Failed to run Qiskit simulation");
    } finally {
      btnSimulate.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
        Verify with Live Qiskit (Aer)
      `;
    }
  }

  // Toast Notifications
  function showToast(msg) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = msg;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 2800);
  }

  // Boot
  init();
});
