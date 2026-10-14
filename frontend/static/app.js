// AURA Jewellery Visual Retrieval Engine — Frontend Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropPrompt = document.getElementById("dropPrompt");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const btnChangeImage = document.getElementById("btnChangeImage");
  const qualityBox = document.getElementById("qualityBox");
  
  const modelSelect = document.getElementById("modelSelect");
  const searchModeSelect = document.getElementById("searchModeSelect");
  const categoryFilter = document.getElementById("categoryFilter");
  const topKSelect = document.getElementById("topKSelect");
  const checkAutoEnhance = document.getElementById("checkAutoEnhance");
  const checkClahe = document.getElementById("checkClahe");
  const checkDenoise = document.getElementById("checkDenoise");
  const checkCrop = document.getElementById("checkCrop");
  const checkAttentionMap = document.getElementById("checkAttentionMap");
  const attentionModeSelect = document.getElementById("attentionModeSelect");
  
  const btnSearch = document.getElementById("btnSearch");
  const btnSearchText = document.getElementById("btnSearchText");
  const resultsContainer = document.getElementById("resultsContainer");
  
  const diagBar = document.getElementById("diagBar");
  const diagModel = document.getElementById("diagModel");
  const diagTotalMs = document.getElementById("diagTotalMs");
  const diagPreprocMs = document.getElementById("diagPreprocMs");
  const diagEncMs = document.getElementById("diagEncMs");
  const diagSearchMs = document.getElementById("diagSearchMs");

  const btnOpenEval = document.getElementById("btnOpenEval");
  const evalModal = document.getElementById("evalModal");
  const btnCloseEval = document.getElementById("btnCloseEval");
  const evalContent = document.getElementById("evalContent");

  const lightboxModal = document.getElementById("lightboxModal");
  const btnCloseLightbox = document.getElementById("btnCloseLightbox");
  const lightboxImg = document.getElementById("lightboxImg");
  const lightboxTitle = document.getElementById("lightboxTitle");
  const lightboxDetails = document.getElementById("lightboxDetails");

  let currentFile = null;
  let currentQueryId = null;
  let originalPreviewSrc = null;
  let currentAttentionHeatmapSrc = null;

  // File Upload Handlers
  dropzone.addEventListener("click", () => fileInput.click());
  btnChangeImage.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.match("image.*")) {
      alert("Please upload a valid image file (JPG, PNG, or WEBP).");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      originalPreviewSrc = e.target.result;
      currentAttentionHeatmapSrc = null;
      if (checkAttentionMap) checkAttentionMap.checked = false;
      previewImg.src = e.target.result;
      dropPrompt.style.display = "none";
      previewContainer.style.display = "flex";
      btnSearch.disabled = false;
      btnSearchText.textContent = "Search Catalog";
      qualityBox.style.display = "none";
      const qBox = document.getElementById("queryBbox");
      if (qBox) qBox.style.display = "none";
    };
    reader.readAsDataURL(file);
  }

  // Attention heatmap toggle listener
  if (checkAttentionMap) {
    checkAttentionMap.addEventListener("change", () => {
      if (checkAttentionMap.checked) {
        if (currentAttentionHeatmapSrc) {
          previewImg.src = currentAttentionHeatmapSrc;
        } else {
          checkAttentionMap.checked = false;
          alert("Run a search first to generate the soft attention heatmap for this image.");
        }
      } else {
        if (originalPreviewSrc) {
          previewImg.src = originalPreviewSrc;
        }
      }
    });
  }

  // Synchronize model selection with attention mode
  if (modelSelect && attentionModeSelect) {
    modelSelect.addEventListener("change", () => {
      if (modelSelect.value === "soft_attention" && attentionModeSelect.value === "none") {
        attentionModeSelect.value = "soft_combined";
      }
    });
  }

  // Search Action
  btnSearch.addEventListener("click", async () => {
    if (!currentFile) return;

    btnSearch.disabled = true;
    btnSearchText.textContent = "Analyzing & Searching...";
    resultsContainer.innerHTML = `
      <div class="state-box">
        <div class="state-icon">⏳</div>
        <h3>Scanning Jewellery Catalog...</h3>
        <p>Extracting fine geometric patch embeddings and matching prongs, stones, and silhouettes.</p>
      </div>
    `;

    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("model", modelSelect.value);
    formData.append("search_mode", searchModeSelect.value);
    formData.append("category_filter", categoryFilter.value);
    formData.append("top_k", topKSelect.value);
    formData.append("attention_mode", attentionModeSelect ? attentionModeSelect.value : "none");
    formData.append("enable_clahe", checkClahe.checked);
    formData.append("enable_denoise", checkDenoise.checked);
    formData.append("enable_crop", checkCrop.checked);
    formData.append("auto_enhance", checkAutoEnhance.checked);

    try {
      const resp = await fetch("/api/search", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Search request failed");
      }

      const data = await resp.json();
      currentQueryId = data.query_id;
      if (data.attention_heatmap_b64) {
        currentAttentionHeatmapSrc = "data:image/jpeg;base64," + data.attention_heatmap_b64;
        if (checkAttentionMap && checkAttentionMap.checked) {
          previewImg.src = currentAttentionHeatmapSrc;
        }
      }
      renderSearchResults(data);
      renderDiagnostics(data);
      renderQualityAssessment(data.quality);
      renderLocalization(data.localization);
    } catch (err) {
      resultsContainer.innerHTML = `
        <div class="state-box" style="border-color: var(--danger);">
          <div class="state-icon" style="color: var(--danger);">⚠️</div>
          <h3>Search Error</h3>
          <p>${err.message}</p>
        </div>
      `;
    } finally {
      btnSearch.disabled = false;
      btnSearchText.textContent = "Search Catalog";
    }
  });

  function renderDiagnostics(data) {
    diagBar.style.display = "flex";
    diagModel.textContent = `Model: ${data.model_name.toUpperCase()} (${data.search_mode.toUpperCase()})`;
    const t = data.timing_ms || {};
    diagTotalMs.textContent = `${t.total || 0}ms`;
    diagPreprocMs.textContent = `${t.query_enhancement ?? t.preprocessing ?? 0}ms`;

    if (t.dinov2_inference !== undefined || t.patch_reranking !== undefined) {
      const encTotal = (t.dinov2_inference || 0) + (t.dinov3_inference || 0) + (t.patch_extraction || 0);
      diagEncMs.textContent = `${Math.round(encTotal)}ms (Encs)`;
      diagSearchMs.textContent = `${t.patch_reranking ? Math.round(t.patch_reranking) + 'ms (Rerank)' : Math.round(t.global_candidate_generation || 0) + 'ms'}`;
    } else {
      diagEncMs.textContent = `${t.encoding || 0}ms`;
      diagSearchMs.textContent = `${t.search || 0}ms`;
    }
  }

  function renderQualityAssessment(quality) {
    if (!quality) return;
    qualityBox.style.display = "block";
    
    if (quality.warnings && quality.warnings.length > 0) {
      qualityBox.className = "quality-box warning";
      qualityBox.innerHTML = `
        <strong>⚠️ Image Quality Notice:</strong>
        <ul style="margin-left: 1.2rem; margin-top: 0.25rem;">
          ${quality.warnings.map(w => `<li>${w}</li>`).join("")}
        </ul>
        <div style="margin-top: 0.35rem; font-size: 0.72rem; color: #fde047;">
          Sharpness: ${quality.sharpness_score} | Mean Luminance: ${quality.mean_luminance}
        </div>
      `;
    } else {
      qualityBox.className = "quality-box clean";
      qualityBox.innerHTML = `
        <strong>✓ High Quality Image:</strong>
        Sharpness score: ${quality.sharpness_score} | Luminance: ${quality.mean_luminance} | Ideal for fine detail matching.
      `;
    }
  }

  function renderLocalization(loc) {
    const bbox = document.getElementById("queryBbox");
    const label = document.getElementById("queryBboxLabel");
    if (!bbox || !label) return;
    if (!loc || loc.is_fallback || !loc.box) {
      bbox.style.display = "none";
      return;
    }
    const [ymin, xmin, ymax, xmax] = loc.box;
    bbox.style.top = `${(ymin * 100).toFixed(1)}%`;
    bbox.style.left = `${(xmin * 100).toFixed(1)}%`;
    bbox.style.height = `${((ymax - ymin) * 100).toFixed(1)}%`;
    bbox.style.width = `${((xmax - xmin) * 100).toFixed(1)}%`;
    const labelClean = (loc.label || "component").replace("_", " ");
    label.textContent = `${labelClean} (${Math.round((loc.confidence || 0) * 100)}%)`;
    bbox.style.display = "block";
  }

  function renderSearchResults(data) {
    const results = data.results;
    if (!results || results.length === 0) {
      resultsContainer.innerHTML = `
        <div class="state-box">
          <div class="state-icon">🔍</div>
          <h3>No Matching Jewellery Found</h3>
          <p>No candidates matched within the specified similarity threshold or category filter.</p>
        </div>
      `;
      return;
    }

    let gridHtml = `<div class="gallery-grid">`;
    results.forEach(r => {
      const simPercent = (r.similarity * 100).toFixed(1);
      const isExact = r.match_type === "exact_match";
      
      gridHtml += `
        <div class="card" data-id="${r.id}" data-category="${r.category}" data-sim="${r.similarity}">
          <div class="card-img-wrapper" onclick="openLightbox('${r.image_url}', '${r.id}', '${r.category}', '${simPercent}%', '${r.cluster_id}')">
            <img class="card-img" src="${r.image_url}" alt="${r.id}" loading="lazy">
            <div class="rank-badge">#${r.rank}</div>
            <div class="confidence-badge ${r.match_type}">
              ${r.confidence_label}
            </div>
          </div>
          <div class="card-body">
            <div class="card-title" title="${r.id}">${r.id}</div>
            <div class="card-meta">
              <span>Category: <strong>${r.category.toUpperCase()}</strong></span>
              <span class="sim-score">${simPercent}% Match</span>
            </div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">
              Cluster: <code>${r.cluster_id}</code>
            </div>
            
            <div class="card-feedback">
              <span class="feedback-label">Match accuracy:</span>
              <div class="feedback-btns">
                <button class="btn-fb" title="Confirmed exact match" onclick="submitFeedback('${r.id}', 'correct', ${r.similarity}, this)">✓</button>
                <button class="btn-fb" title="Different design / Incorrect" onclick="submitFeedback('${r.id}', 'incorrect', ${r.similarity}, this)">✗</button>
                <button class="btn-fb" title="Uncertain / Borderline" onclick="submitFeedback('${r.id}', 'uncertain', ${r.similarity}, this)">?</button>
              </div>
            </div>
          </div>
        </div>
      `;
    });
    gridHtml += `</div>`;
    resultsContainer.innerHTML = gridHtml;
  }

  // Feedback Submission
  window.submitFeedback = async function(resultId, feedbackType, simScore, btnElem) {
    try {
      const resp = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_id: currentQueryId || "manual_query",
          result_id: resultId,
          feedback_type: feedbackType,
          similarity_score: simScore,
          model_name: modelSelect.value
        })
      });

      if (resp.ok) {
        const parent = btnElem.parentElement;
        parent.querySelectorAll(".btn-fb").forEach(b => b.classList.remove("active"));
        btnElem.classList.add("active");
        btnElem.style.borderColor = "var(--badge-exact)";
      }
    } catch (e) {
      console.error("Feedback submission error:", e);
    }
  };

  // Lightbox Modal
  window.openLightbox = function(url, id, cat, sim, cluster) {
    lightboxImg.src = url;
    lightboxTitle.textContent = `${id} (${cat.toUpperCase()})`;
    lightboxDetails.textContent = `Cosine Similarity: ${sim} | Cluster Identity: ${cluster}`;
    lightboxModal.style.display = "flex";
  };

  btnCloseLightbox.addEventListener("click", () => lightboxModal.style.display = "none");
  lightboxModal.addEventListener("click", (e) => {
    if (e.target === lightboxModal) lightboxModal.style.display = "none";
  });

  // Evaluation & Benchmark Modal
  btnOpenEval.addEventListener("click", async () => {
    evalModal.style.display = "flex";
    evalContent.innerHTML = "Fetching latest empirical benchmarks...";

    try {
      const resp = await fetch("/api/evaluation");
      const data = await resp.json();
      
      if (data.status === "pending") {
        evalContent.innerHTML = `<p>${data.message}</p>`;
        return;
      }

      let html = "";
      if (data.dinov2 && data.dinov3) {
        const rows = [
          { name: "Baseline Global (CLS)", k2: "Baseline Global (CLS)", k3: "Baseline Global (CLS)" },
          { name: "Config A: Uniform Pooling", k2: "Config A: Uniform Patch Pooling", k3: "Config A: Uniform Patch Pooling" },
          { name: "Config B: Foreground Saliency", k2: "Config B: Foreground Saliency", k3: "Config B: Foreground Saliency" },
          { name: "Config C: Soft Component Region", k2: "Config C: Soft Component Region", k3: "Config C: Soft Component Region" },
          { name: "Config E: Combined Soft Saliency", k2: "Config E: Combined Soft Saliency", k3: "Config E: Combined Soft Saliency" },
          { name: "Fused: Global + Combined (α=0.3)", k2: "Fused: Global + Combined (alpha=0.3)", k3: "Fused: Global + Combined (alpha=0.3)" },
          { name: "Negative Control: Hard-Crop", k2: "Negative Control: Hard-Crop Retrieval", k3: "Negative Control: Hard-Crop Retrieval" },
          { name: "Diagnostic Oracle: GT Region", k2: "Diagnostic Oracle: GT Region Saliency", k3: "Diagnostic Oracle: GT Region Saliency" },
        ];

        html = `
          <div style="font-size: 0.85rem; color: var(--accent-gold-light); margin-bottom: 0.5rem; font-weight: bold;">
            EXP-007: Soft Spatial Attention & Patch Reweighting Benchmark (60 Held-Out Queries: 34 Real, 26 Synthetic)
          </div>
          <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem;">
            Empirical evaluation of train-free soft token-reweighting on 16×16 patch grids. Real and Synthetic subsets strictly separated.
          </p>
          <div style="overflow-x: auto;">
          <table style="width: 100%; border-collapse: collapse; font-size: 0.78rem;">
            <thead>
              <tr style="border-bottom: 2px solid var(--border-subtle); text-align: left; color: var(--accent-gold-light);">
                <th style="padding: 0.4rem;">Configuration</th>
                <th style="padding: 0.4rem;">Model</th>
                <th style="padding: 0.4rem;">Overall R@1</th>
                <th style="padding: 0.4rem;">Real R@1 (N=34)</th>
                <th style="padding: 0.4rem;">Syn R@1 (N=26)</th>
                <th style="padding: 0.4rem;">R@5</th>
                <th style="padding: 0.4rem;">R@20</th>
                <th style="padding: 0.4rem;">MRR</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(r => {
                const c2 = data.dinov2[r.k2];
                const c3 = data.dinov3[r.k3];
                const r2_all = c2 ? c2.overall_all_60 : {};
                const r2_real = c2 ? c2.real_queries_all_34 : {};
                const r2_syn = c2 ? c2.synthetic_rings_26 : {};
                const r3_all = c3 ? c3.overall_all_60 : {};
                const r3_real = c3 ? c3.real_queries_all_34 : {};
                const r3_syn = c3 ? c3.synthetic_rings_26 : {};
                return `
                  <tr style="border-top: 1px solid var(--border-subtle);">
                    <td rowspan="2" style="padding: 0.4rem; font-weight: bold; vertical-align: middle;">${r.name}</td>
                    <td style="padding: 0.4rem; color: #93c5fd;">DINOv2</td>
                    <td style="padding: 0.4rem;">${(r2_all.r1 * 100).toFixed(1)}% (${r2_all.r1_num})</td>
                    <td style="padding: 0.4rem;">${(r2_real.r1 * 100).toFixed(1)}% (${r2_real.r1_num})</td>
                    <td style="padding: 0.4rem;">${(r2_syn.r1 * 100).toFixed(1)}% (${r2_syn.r1_num})</td>
                    <td style="padding: 0.4rem;">${(r2_all.r5 * 100).toFixed(1)}%</td>
                    <td style="padding: 0.4rem;">${(r2_all.r20 * 100).toFixed(1)}%</td>
                    <td style="padding: 0.4rem;">${(r2_all.mrr || 0).toFixed(4)}</td>
                  </tr>
                  <tr style="border-bottom: 1px solid var(--border-subtle); background: rgba(255,255,255,0.02);">
                    <td style="padding: 0.4rem; color: #f59e0b;">DINOv3</td>
                    <td style="padding: 0.4rem; font-weight: bold;">${(r3_all.r1 * 100).toFixed(1)}% (${r3_all.r1_num})</td>
                    <td style="padding: 0.4rem; font-weight: bold;">${(r3_real.r1 * 100).toFixed(1)}% (${r3_real.r1_num})</td>
                    <td style="padding: 0.4rem; font-weight: bold;">${(r3_syn.r1 * 100).toFixed(1)}% (${r3_syn.r1_num})</td>
                    <td style="padding: 0.4rem; font-weight: bold;">${(r3_all.r5 * 100).toFixed(1)}%</td>
                    <td style="padding: 0.4rem; font-weight: bold;">${(r3_all.r20 * 100).toFixed(1)}%</td>
                    <td style="padding: 0.4rem; font-weight: bold;">${(r3_all.mrr || 0).toFixed(4)}</td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
          </div>
        `;
      } else if (cfgs && cfgs.dinov2_global) {
        const cV2 = cfgs.dinov2_global;
        const cV3 = cfgs.dinov3_global;
        const cHyb = cfgs.hybrid_global || cfgs.hybrid_rrf_no_rerank;
        const cRerank = cfgs.hybrid_rerank_sym_pw02 || cfgs.hybrid_rrf_patch_rerank;

        // Use benchmark_v2_corrected if present, else fallback
        const mKey = cV2.benchmark_v2_corrected ? "benchmark_v2_corrected" : null;
        const getM = (cfg) => mKey ? cfg[mKey] : cfg;

        const mV2 = getM(cV2);
        const mV3 = getM(cV3);
        const mHyb = getM(cHyb);
        const mRerank = getM(cRerank);

        html = `
          <div style="font-size: 0.8rem; color: var(--accent-gold-light); margin-bottom: 0.5rem; font-weight: bold;">
            Primary Benchmark (Corrected Ground-Truth, N=${mV2.num_queries || 29} Twin Queries)
          </div>
          <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.5rem;">
            <thead>
              <tr style="border-bottom: 2px solid var(--border-subtle); text-align: left; color: var(--accent-gold-light);">
                <th style="padding: 0.5rem;">Metric</th>
                <th style="padding: 0.5rem;">DINOv2</th>
                <th style="padding: 0.5rem;">DINOv3</th>
                <th style="padding: 0.5rem;">Hybrid Global</th>
                <th style="padding: 0.5rem;">Hybrid + Rerank (pw=0.2)</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.5rem;"><strong>Recall@1 (Exact Match)</strong></td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mV2.recall_at_1}%</td>
                <td style="padding: 0.5rem;">${mV3.recall_at_1}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mHyb.recall_at_1}%</td>
                <td style="padding: 0.5rem;">${mRerank.recall_at_1}%</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.5rem;"><strong>Recall@5</strong></td>
                <td style="padding: 0.5rem;">${mV2.recall_at_5}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mV3.recall_at_5}%</td>
                <td style="padding: 0.5rem;">${mHyb.recall_at_5}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mRerank.recall_at_5}%</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.5rem;"><strong>Recall@10</strong></td>
                <td style="padding: 0.5rem;">${mV2.recall_at_10}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mV3.recall_at_10}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mHyb.recall_at_10}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mRerank.recall_at_10}%</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.5rem;"><strong>Candidate Recall@20</strong></td>
                <td style="padding: 0.5rem;">${mV2.candidate_recall_at_20}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mV3.candidate_recall_at_20}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mHyb.candidate_recall_at_20}%</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mRerank.candidate_recall_at_20}%</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.5rem;"><strong>MRR</strong></td>
                <td style="padding: 0.5rem;">${mV2.mrr}</td>
                <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mV3.mrr}</td>
                <td style="padding: 0.5rem;">${mHyb.mrr}</td>
                <td style="padding: 0.5rem;">${mRerank.mrr}</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 0.5rem;"><strong>False Exact (&ge;&tau;<sub>99%</sub>)</strong></td>
                <td style="padding: 0.5rem;">${cV2.false_exact_matches}</td>
                <td style="padding: 0.5rem;">${cV3.false_exact_matches}</td>
                <td style="padding: 0.5rem;">${cHyb.false_exact_matches}</td>
                <td style="padding: 0.5rem;">${cRerank.false_exact_matches}</td>
              </tr>
              <tr>
                <td style="padding: 0.5rem;"><strong>Mean Latency (CPU)</strong></td>
                <td style="padding: 0.5rem;">${cV2.mean_latency_ms} ms</td>
                <td style="padding: 0.5rem;">${cV3.mean_latency_ms} ms</td>
                <td style="padding: 0.5rem;">${cHyb.mean_latency_ms} ms</td>
                <td style="padding: 0.5rem;">${cRerank.mean_latency_ms} ms</td>
              </tr>
            </tbody>
          </table>
        `;

        if (data.patch_weight_sweep) {
          html += `
            <div style="font-size: 0.8rem; color: var(--accent-gold-light); margin-top: 1.25rem; margin-bottom: 0.5rem; font-weight: bold;">
              Fusion Parameter Sweep (patch_weight: 0.0 = Global Only, 1.0 = Chamfer Only)
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
              <thead>
                <tr style="border-bottom: 1px solid var(--border-subtle); color: var(--text-muted); text-align: left;">
                  <th style="padding: 0.4rem;">patch_weight</th>
                  <th style="padding: 0.4rem;">Recall@1</th>
                  <th style="padding: 0.4rem;">Recall@5</th>
                  <th style="padding: 0.4rem;">Recall@10</th>
                  <th style="padding: 0.4rem;">MRR</th>
                </tr>
              </thead>
              <tbody>
                ${data.patch_weight_sweep.map(sw => `
                  <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.4rem; font-weight: bold;">${sw.patch_weight}</td>
                    <td style="padding: 0.4rem;">${sw.recall_at_1}%</td>
                    <td style="padding: 0.4rem; color: ${sw.recall_at_5 >= 93 ? 'var(--badge-exact)' : 'inherit'}; font-weight: bold;">${sw.recall_at_5}%</td>
                    <td style="padding: 0.4rem;">${sw.recall_at_10}%</td>
                    <td style="padding: 0.4rem;">${sw.mrr}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          `;
        }
      } else {
        html = `
          <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 1rem;">
            <thead>
              <tr style="border-bottom: 2px solid var(--border-subtle); text-align: left; color: var(--accent-gold-light);">
                <th style="padding: 0.5rem;">Metric</th>
                <th style="padding: 0.5rem;">DINOv2 (ViT-S/14)</th>
                <th style="padding: 0.5rem;">DINOv3 (ViT-S/16)</th>
                <th style="padding: 0.5rem;">OpenCLIP (ViT-B/32)</th>
              </tr>
            </thead>
            <tbody>
        `;
        const mDino2 = data.dinov2 ? data.dinov2.metrics : null;
        const mDino3 = data.dinov3 ? data.dinov3.metrics : null;
        const mClip = data.clip ? data.clip.metrics : null;
        if (mDino2) {
          html += `
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 0.5rem;"><strong>Recall@1 (Exact Twin)</strong></td>
              <td style="padding: 0.5rem; color: var(--badge-exact); font-weight: bold;">${mDino2.recall_at_1}%</td>
              <td style="padding: 0.5rem;">${mDino3 ? mDino3.recall_at_1 + '%' : '—'}</td>
              <td style="padding: 0.5rem;">${mClip ? mClip.recall_at_1 + '%' : '—'}</td>
            </tr>
          `;
        }
        html += `</tbody></table>`;
      }
      evalContent.innerHTML = html;
    } catch (e) {
      evalContent.innerHTML = `<p style="color: var(--danger);">Failed to load evaluation report: ${e.message}</p>`;
    }
  });

  btnCloseEval.addEventListener("click", () => evalModal.style.display = "none");
  evalModal.addEventListener("click", (e) => {
    if (e.target === evalModal) evalModal.style.display = "none";
  });
});
