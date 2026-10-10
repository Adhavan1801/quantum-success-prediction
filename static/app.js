// =============================================================
// app.js — Client-Side Logic for Quantum Success Prediction UI
// Clean Light Theme, Explicit Ranges, Best Model Pipeline
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
    auto_metrics: true,
  };

  // DOM Elements
  const algoBtns = document.querySelectorAll(".algo-btn");
  const noiseBtns = document.querySelectorAll(".noise-btn");
  
  const toggleAuto = document.getElementById("toggle-auto-metrics");
  const toggleLabel = document.getElementById("toggle-label");
  
  const inputN = document.getElementById("input-n-qubits");
  const btnDecQubits = document.getElementById("btn-dec-qubits");
  const btnIncQubits = document.getElementById("btn-inc-qubits");
  
  const inputDepth = document.getElementById("input-circuit-depth");
  const inputGate1q = document.getElementById("input-gate-1q");
  const inputGate2q = document.getElementById("input-gate-2q");
  const inputGateTotal = document.getElementById("input-gate-total");

  const metricWarning = document.getElementById("metric-warning");
  const metricWarningText = document.getElementById("metric-warning-text");

  const inputNoiseStrength = document.getElementById("input-noise-strength");
  const noiseRangePill = document.getElementById("noise-range-pill");
  const noiseHelpText = document.getElementById("noise-help-text");
  const noiseUnitTag = document.getElementById("noise-unit-tag");

  const btnPredict = document.getElementById("btn-predict");
  const btnSimulate = document.getElementById("btn-simulate");
  
  // Results Elements
  const gaugeFill = document.getElementById("gauge-fill");
  const resProbPct = document.getElementById("res-prob-pct");
  const resProbRaw = document.getElementById("res-prob-raw");
  const resFidelityBadge = document.getElementById("res-fidelity-badge");
  const resFidelityDesc = document.getElementById("res-fidelity-desc");
  const resDecoherence = document.getElementById("res-decoherence-loss");
  const resDensity = document.getElementById("res-depth-density");
  const resEntanglement = document.getElementById("res-entanglement-ratio");

  const resConfidenceScore = document.getElementById("res-confidence-score");
  const resConfidenceBar = document.getElementById("res-confidence-bar");
  const resConfidencePill = document.getElementById("res-confidence-pill");
  const resCiRange = document.getElementById("res-ci-range");
  const resCiMargin = document.getElementById("res-ci-margin");
  const resStdError = document.getElementById("res-std-error");
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
    updateCircuitEstimate();
    executePrediction();
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

        if (state.auto_metrics) {
          updateCircuitEstimate();
        } else {
          executePrediction();
        }
      });
    });

    // Auto-Metrics Toggle
    toggleAuto.addEventListener("change", (e) => {
      state.auto_metrics = e.target.checked;
      updateToggleState();
      if (state.auto_metrics) {
        updateCircuitEstimate();
      }
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
      executePrediction();
    });

    inputGate1q.addEventListener("input", () => {
      state.gate_count_1q = parseInt(inputGate1q.value, 10) || 8;
      updateTotalGates();
      validateInputs();
      executePrediction();
    });

    inputGate2q.addEventListener("input", () => {
      state.gate_count_2q = parseInt(inputGate2q.value, 10) || 3;
      updateTotalGates();
      validateInputs();
      executePrediction();
    });

    // Noise Model Selection
    noiseBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        noiseBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.noise_type = btn.dataset.noise;

        configureNoiseControls(state.noise_type);
        executePrediction();
      });
    });

    // Noise Strength Input
    inputNoiseStrength.addEventListener("input", () => {
      let val = parseFloat(inputNoiseStrength.value);
      if (isNaN(val)) val = 0.0;
      state.noise_strength = val;
      validateInputs();
      executePrediction();
    });

    // Predict Button Click
    btnPredict.addEventListener("click", () => {
      executePrediction(true);
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
    if (state.auto_metrics) {
      updateCircuitEstimate();
    } else {
      validateInputs();
      executePrediction();
    }
  }

  function updateToggleState() {
    if (state.auto_metrics) {
      toggleLabel.textContent = "Auto-compute from circuit";
      inputDepth.setAttribute("readonly", true);
      inputGate1q.setAttribute("readonly", true);
      inputGate2q.setAttribute("readonly", true);
    } else {
      toggleLabel.textContent = "Manual configuration";
      inputDepth.removeAttribute("readonly");
      inputGate1q.removeAttribute("readonly");
      inputGate2q.removeAttribute("readonly");
    }
  }

  function updateTotalGates() {
    state.total_gate_count = state.gate_count_1q + state.gate_count_2q;
    inputGateTotal.value = state.total_gate_count;
  }

  // Configure Noise controls per noise type
  function configureNoiseControls(noiseType, customVal = null) {
    if (noiseType === "none") {
      inputNoiseStrength.value = "0.000";
      inputNoiseStrength.readOnly = true;
      noiseRangePill.textContent = "Range: 0.000 (Ideal)";
      noiseHelpText.textContent = "Ideal noiseless quantum simulation without environmental interaction.";
      noiseUnitTag.textContent = "rate";
      state.noise_strength = 0.0;
    } else if (noiseType === "depolarizing") {
      inputNoiseStrength.min = "0.001";
      inputNoiseStrength.max = "0.100";
      inputNoiseStrength.step = "0.001";
      inputNoiseStrength.readOnly = false;
      const val = customVal !== null ? customVal : 0.010;
      inputNoiseStrength.value = val;
      state.noise_strength = val;
      noiseRangePill.textContent = "Range: 0.001 – 0.100 (p)";
      noiseHelpText.textContent = "Depolarizing noise: Probability 'p' of replacing qubit state with maximally mixed state per gate.";
      noiseUnitTag.textContent = "error p";
    } else if (noiseType === "amplitude_damping") {
      inputNoiseStrength.min = "0.001";
      inputNoiseStrength.max = "0.100";
      inputNoiseStrength.step = "0.001";
      inputNoiseStrength.readOnly = false;
      const val = customVal !== null ? customVal : 0.010;
      inputNoiseStrength.value = val;
      state.noise_strength = val;
      noiseRangePill.textContent = "Range: 0.001 – 0.100 (γ)";
      noiseHelpText.textContent = "Amplitude damping: Energy decay probability 'γ' (|1⟩ → |0⟩), modeling spontaneous emission & T1 relaxation.";
      noiseUnitTag.textContent = "decay γ";
    } else if (noiseType === "thermal_relaxation") {
      inputNoiseStrength.min = "0.010";
      inputNoiseStrength.max = "0.100";
      inputNoiseStrength.step = "0.005";
      inputNoiseStrength.readOnly = false;
      const val = customVal !== null ? customVal : 0.033;
      inputNoiseStrength.value = val;
      state.noise_strength = val;
      noiseRangePill.textContent = "Range: 0.010 – 0.100 (1/T1)";
      noiseHelpText.textContent = "Thermal relaxation: Combined longitudinal relaxation (T1) and transverse dephasing (T2), real hardware decoherence.";
      noiseUnitTag.textContent = "strength";
    }
  }

  // Fetch Circuit Estimates
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
        inputGateTotal.value = m.total_gate_count;

        validateInputs();
        executePrediction();
      }
    } catch (err) {
      console.error("Error fetching circuit metrics:", err);
    }
  }

  // Validation
  function validateInputs() {
    let warning = null;
    if (state.n_qubits < 2 || state.n_qubits > 8) {
      warning = "Number of qubits is outside tested range (2 – 8).";
    } else if (state.circuit_depth < 3 || state.circuit_depth > 12) {
      warning = "Circuit depth is outside tested range (3 – 12).";
    } else if (state.gate_count_1q < 4 || state.gate_count_1q > 27) {
      warning = "1-Qubit gate count is outside tested range (4 – 27).";
    } else if (state.gate_count_2q < 0 || state.gate_count_2q > 11) {
      warning = "2-Qubit gate count is outside tested range (0 – 11).";
    }

    if (warning) {
      metricWarningText.textContent = warning;
      metricWarning.classList.remove("hidden");
    } else {
      metricWarning.classList.add("hidden");
    }
  }

  // -------------------------------------------------------------
  // PREDICTION EXECUTION
  // -------------------------------------------------------------
  let debounceTimer = null;

  function executePrediction(immediate = false) {
    if (debounceTimer) clearTimeout(debounceTimer);

    if (immediate) {
      performPredictionCall();
    } else {
      debounceTimer = setTimeout(performPredictionCall, 120);
    }
  }

  async function performPredictionCall() {
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
        renderPredictionResults(res);
      } else {
        showToast(res.message || "Prediction error", "error");
      }
    } catch (err) {
      console.error("Prediction API failed:", err);
      showToast("Server connection error during prediction", "error");
    } finally {
      btnPredict.innerHTML = `
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        Predict Success Probability
      `;
    }
  }

  // Render Prediction Results (Using Best Model: Decision Tree R² = 0.9693)
  function renderPredictionResults(res) {
    // Best model is Decision Tree (highest R² and lowest MAE)
    const prob = res.prediction.decision_tree_probability;
    const probPct = res.prediction.decision_tree_probability_pct;
    const conf = res.confidence;
    const fidelity = res.fidelity;
    const telemetry = res.circuit_telemetry;

    // 1. Gauge Animation
    animateGauge(prob, probPct);
    resProbRaw.textContent = `P = ${prob.toFixed(6)}`;

    // 2. Fidelity Meta
    resFidelityBadge.textContent = fidelity.status;
    resFidelityBadge.style.color = fidelity.color;
    resFidelityBadge.style.borderColor = fidelity.color;
    resFidelityBadge.style.background = `${fidelity.color}15`;
    resFidelityDesc.textContent = fidelity.description;

    resDecoherence.textContent = `${(100 - probPct).toFixed(2)}%`;
    resDensity.textContent = telemetry.depth_density;
    resEntanglement.textContent = `${telemetry.entanglement_ratio_pct}%`;

    // 3. Confidence Metrics
    resConfidenceScore.textContent = `${conf.score.toFixed(1)}%`;
    resConfidenceScore.style.color = conf.color;
    resConfidenceBar.style.width = `${Math.min(100, Math.max(0, conf.score))}%`;
    resConfidenceBar.style.background = conf.color;
    resConfidencePill.textContent = conf.level;
    resConfidencePill.style.color = conf.color;
    resConfidencePill.style.borderColor = `${conf.color}50`;
    resConfidencePill.style.background = `${conf.color}15`;

    resCiRange.textContent = `[${conf.ci_95.lower_pct}%, ${conf.ci_95.upper_pct}%]`;
    resCiMargin.textContent = `Margin: ±${conf.ci_95.margin_pct}%`;
    resStdError.textContent = conf.std_error.toFixed(4);

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

        simCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        showToast("Live Qiskit Aer simulation complete!");
      } else {
        showToast(data.message || "Simulation error", "error");
      }
    } catch (err) {
      console.error("Simulation error:", err);
      showToast("Failed to run Qiskit simulation", "error");
    } finally {
      btnSimulate.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m10 15 5-3-5-3v6z"/></svg>
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
