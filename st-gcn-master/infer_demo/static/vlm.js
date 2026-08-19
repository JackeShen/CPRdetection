/**
 * VLM 图片分类前端逻辑
 * 流程：选图 → 预览 → POST /api/vlm_classify → 渲染结果
 */
(function () {
  "use strict";

  var els = {
    dropzone:   document.getElementById("vlm-dropzone"),
    fileInput:  document.getElementById("vlm-file"),
    submitBtn:  document.getElementById("vlm-submit"),
    previewWrap:document.getElementById("vlm-preview-wrap"),
    imgMeta:    document.getElementById("img-meta"),
    resultBox:  document.getElementById("vlm-result"),
    resultHero: document.getElementById("result-hero"),
    resultIcon: document.getElementById("result-icon"),
    resultName: document.getElementById("result-class-name"),
    resultMeta: document.getElementById("result-class-meta"),
    resultConf: document.getElementById("result-conf"),
    reasoningCard: document.getElementById("result-reasoning-card"),
    reasoningText: document.getElementById("result-reasoning-text"),
    featuresCard:  document.getElementById("result-features-card"),
    featureList:   document.getElementById("result-feature-list"),
    rawOutput:  document.getElementById("result-raw"),
    classGrid:  document.getElementById("class-grid"),
  };

  var selectedFile = null;
  var CLASS_ICONS = ["✅", "✋", "✊", "🖐️", "💪", "📐", "🧎", "🧍", "📍"];

  // ── 文件选择 ──────────────────────────────────────────────
  els.dropzone.addEventListener("click", function () {
    els.fileInput.click();
  });

  els.dropzone.addEventListener("dragover", function (e) {
    e.preventDefault();
    els.dropzone.classList.add("drag");
  });

  els.dropzone.addEventListener("dragleave", function () {
    els.dropzone.classList.remove("drag");
  });

  els.dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    els.dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  els.fileInput.addEventListener("change", function () {
    if (this.files.length > 0) {
      handleFile(this.files[0]);
    }
  });

  function handleFile(file) {
    console.log("[vlm] file:", file.name, "type=" + file.type, "size=" + file.size);
    if (!file.type.startsWith("image/")) {
      alert("请选择图片文件（jpg/png/bmp）");
      return;
    }
    if (file.size === 0) {
      alert("文件为空");
      return;
    }
    selectedFile = file;

    // 预览
    var reader = new FileReader();
    reader.onload = function (e) {
      els.previewWrap.innerHTML =
        '<img src="' + e.target.result + '" alt="预览">';
      els.imgMeta.textContent =
        file.name + " · " + (file.size / 1024).toFixed(0) + " KB";
    };
    reader.readAsDataURL(file);

    // 启用按钮
    els.submitBtn.disabled = false;
    els.dropzone.classList.add("has-file");
    els.dropzone.querySelector(".dropzone-title").textContent =
      "已选择: " + file.name;

    // 隐藏旧结果
    els.resultBox.style.display = "none";
    clearClassChips();
  }

  // ── 提交分析 ──────────────────────────────────────────────
  els.submitBtn.addEventListener("click", function () {
    if (!selectedFile) return;
    analyze();
  });

  function analyze() {
    els.submitBtn.disabled = true;
    els.submitBtn.querySelector(".btn-text").textContent = "分析中...";

    // 显示 loading 状态
    els.resultBox.style.display = "flex";
    els.resultHero.className = "result-hero loading";
    els.resultIcon.textContent = "⏳";
    els.resultName.innerHTML =
      '<span class="spinner"></span> VLM 正在分析图片...';
    els.resultMeta.textContent = "";
    els.resultConf.textContent = "";
    els.resultConf.className = "conf-badge";
    els.reasoningCard.style.display = "none";
    els.featuresCard.style.display = "none";
    els.rawOutput.textContent = "";
    clearClassChips();

    var formData = new FormData();
    formData.append("image", selectedFile);

    fetch("/api/vlm_classify", { method: "POST", body: formData })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderResult(data);
      })
      .catch(function (err) {
        renderError("网络错误: " + err.message);
      })
      .finally(function () {
        els.submitBtn.disabled = false;
        els.submitBtn.querySelector(".btn-text").textContent =
          "🔍 开始 VLM 分析";
      });
  }

  // ── 渲染结果 ──────────────────────────────────────────────
  function renderResult(data) {
    if (data.error) {
      renderError(data.error);
      return;
    }

    var idx = data.class_index;
    var className = data.predicted_class || "未知";
    var conf = data.confidence || "low";
    var origIdx = data.original_class_index;

    els.resultHero.className = "result-hero";
    els.resultIcon.textContent =
      idx !== null && idx >= 0 && idx < CLASS_ICONS.length
        ? CLASS_ICONS[idx]
        : "❓";
    els.resultName.textContent = className;

    var metaParts = [];
    if (idx !== null) metaParts.push("VLM 索引: " + idx);
    if (origIdx !== null) metaParts.push("原始 14 类索引: " + origIdx);
    metaParts.push("模型: " + (data.model || "qwen3-vl:2b"));
    els.resultMeta.textContent = metaParts.join(" · ");

    els.resultConf.textContent = conf.toUpperCase();
    els.resultConf.className = "conf-badge conf-" + conf;

    // 高亮分类芯片
    if (idx !== null) {
      var chip = els.classGrid.querySelector(
        '.class-chip[data-idx="' + idx + '"]'
      );
      if (chip) chip.classList.add("active");
    }

    // 推理理由
    if (data.reasoning) {
      els.reasoningCard.style.display = "block";
      els.reasoningText.textContent = data.reasoning;
    } else {
      els.reasoningCard.style.display = "none";
    }

    // 可观察特征
    if (data.observable_features && data.observable_features.length > 0) {
      els.featuresCard.style.display = "block";
      els.featureList.innerHTML = data.observable_features
        .map(function (f) { return "<li>" + escapeHtml(f) + "</li>"; })
        .join("");
    } else {
      els.featuresCard.style.display = "none";
    }

    // 原始输出
    els.rawOutput.textContent = data.raw_response || JSON.stringify(data, null, 2);
  }

  function renderError(msg) {
    els.resultHero.className = "result-hero error";
    els.resultIcon.textContent = "⚠️";
    els.resultName.textContent = "分析失败";
    els.resultMeta.textContent = msg;
    els.resultConf.textContent = "ERROR";
    els.resultConf.className = "conf-badge conf-low";
    els.reasoningCard.style.display = "none";
    els.featuresCard.style.display = "none";
    els.rawOutput.textContent = msg;
  }

  function clearClassChips() {
    var chips = els.classGrid.querySelectorAll(".class-chip");
    for (var i = 0; i < chips.length; i++) {
      chips[i].classList.remove("active");
    }
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }
})();
