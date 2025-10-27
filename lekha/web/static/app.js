const state = {
  view: "line",
  segmentId: null,
  projectId: null,
  dirty: false,
};

const imageFrame = document.querySelector(".image-frame");
const imageEl = document.getElementById("segment-image");
const textbox = document.getElementById("transcription-box");
const viewRadios = document.querySelectorAll('input[name="view"]');
const buttons = document.querySelectorAll(".actions button");
const prevButton = document.querySelector('[data-action="prev"]');
const nextButton = document.querySelector('[data-action="next"]');
const nextIssueButton = document.querySelector('[data-action="next_issue"]');
const projectSelect = document.getElementById("project-select");
const exportButton = document.getElementById("export-button");
const zoomPreview = document.getElementById("zoom-preview");

const ZOOM_SCALE = 2.5;
const ZOOM_BOX_SIZE = 180;

textbox.disabled = true;

function setLoading(isLoading) {
  buttons.forEach((btn) => {
    if (btn.dataset.action !== "save") {
      if (isLoading) {
        btn.dataset.prevDisabled = btn.disabled ? "1" : "0";
        btn.disabled = true;
      } else if (btn.dataset.prevDisabled !== undefined) {
        btn.disabled = btn.dataset.prevDisabled === "1";
        delete btn.dataset.prevDisabled;
      }
    }
  });
  if (projectSelect) {
    if (isLoading) {
      projectSelect.dataset.prevDisabled = projectSelect.disabled ? "1" : "0";
      projectSelect.disabled = true;
    } else if (projectSelect.dataset.prevDisabled !== undefined) {
      projectSelect.disabled = projectSelect.dataset.prevDisabled === "1";
      delete projectSelect.dataset.prevDisabled;
    }
  }
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || res.statusText);
  }
  return res.json();
}

function updateTextbox(text) {
  textbox.value = text;
  state.dirty = false;
}

function updateImage(src, hasConflict) {
  resetZoom();
  if (src) {
    imageEl.src = src;
    if (zoomPreview) {
      zoomPreview.style.backgroundImage = `url('${src}')`;
    }
  } else {
    imageEl.removeAttribute("src");
    if (zoomPreview) {
      zoomPreview.style.backgroundImage = "none";
    }
  }
  imageFrame.classList.toggle("conflict", Boolean(hasConflict));
}

function applySegmentPayload(payload) {
  if (!payload) {
    state.segmentId = null;
    updateImage(null, false);
    updateTextbox("");
    updateNavigation(null);
    textbox.disabled = true;
    return;
  }
  state.segmentId = payload.segment_id;
  updateImage(payload.image_url, payload.has_conflict);
  updateTextbox(payload.text);
  updateNavigation(payload.navigation ?? null);
  textbox.disabled = false;
}

function updateNavigation(nav) {
  if (!prevButton || !nextButton || !nextIssueButton) {
    return;
  }
  if (!nav) {
    prevButton.disabled = true;
    nextButton.disabled = true;
    nextIssueButton.disabled = true;
    return;
  }
  prevButton.disabled = !nav.can_prev;
  nextButton.disabled = !nav.can_next;
  nextIssueButton.disabled = !nav.has_next_issue;
}

async function loadInitialState() {
  updateNavigation(null);
  const data = await fetchJSON("/api/state");
  state.view = data.view;
  state.segmentId = data.segment_id;
  state.projectId = data.project_id;
  for (const radio of viewRadios) {
    radio.checked = radio.value === state.view;
  }
  await loadProjects();
  await loadSegment();
}

async function loadProjects() {
  if (!projectSelect) {
    return;
  }
  const data = await fetchJSON("/api/projects");
  if (!Array.isArray(data.projects)) {
    projectSelect.innerHTML = "";
    return;
  }
  projectSelect.innerHTML = data.projects
    .map(
      (proj) =>
        `<option value="${proj.project_id ?? ""}">${proj.label ?? proj.project_id ?? "Untitled project"}</option>`
    )
    .join("");
  if (state.projectId) {
    projectSelect.value = state.projectId;
  }
}

async function loadSegment() {
  if (!state.segmentId) {
    applySegmentPayload(null);
    return;
  }
  setLoading(true);
  let payload = null;
  try {
    payload = await fetchJSON(`/api/segment/${state.segmentId}?view=${state.view}`);
  } catch (error) {
    console.error(error);
    alert(error.message);
  } finally {
    setLoading(false);
  }
  if (payload) {
    applySegmentPayload(payload);
  }
}

async function save(action) {
  if (!state.segmentId) {
    return;
  }
  const payload = {
    segment_id: state.segmentId,
    view: state.view,
    text: textbox.value,
    action,
  };
  try {
    const data = await fetchJSON("/api/save", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (data.view) {
      state.view = data.view;
      for (const radio of viewRadios) {
        radio.checked = radio.value === state.view;
      }
    }
    applySegmentPayload(data);
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

viewRadios.forEach((radio) => {
  radio.addEventListener("change", async (event) => {
    if (!state.segmentId) {
      return;
    }
    state.view = event.target.value;
    try {
      const data = await fetchJSON("/api/view", {
        method: "POST",
        body: JSON.stringify({view: state.view, segment_id: state.segmentId}),
      });
      state.view = data.view ?? state.view;
      applySegmentPayload(data);
      for (const option of viewRadios) {
        option.checked = option.value === state.view;
      }
    } catch (error) {
      console.error(error);
      alert(error.message);
    }
  });
});

textbox.addEventListener("input", () => {
  state.dirty = true;
});

buttons.forEach((button) => {
  button.addEventListener("click", async () => {
    const action = button.dataset.action;
    if (!state.segmentId || button.disabled) {
      return;
    }
    if (action === "save" && !state.dirty) {
      alert("No changes to save.");
      return;
    }
    await save(action);
  });
});

if (projectSelect) {
  projectSelect.addEventListener("change", async (event) => {
    const projectId = event.target.value;
    if (!projectId || projectId === state.projectId) {
      return;
    }
    const previousProject = state.projectId;
    let response = null;
    try {
      setLoading(true);
      response = await fetchJSON("/api/project", {
        method: "POST",
        body: JSON.stringify({project_id: projectId}),
      });
    } catch (error) {
      console.error(error);
      alert(error.message);
      state.projectId = previousProject;
      projectSelect.value = previousProject ?? "";
    } finally {
      setLoading(false);
    }
    if (response) {
      state.projectId = response.project_id;
      state.view = response.view;
      state.segmentId = response.segment_id;
      for (const radio of viewRadios) {
        radio.checked = radio.value === state.view;
      }
      projectSelect.value = state.projectId;
      applySegmentPayload(response.segment);
    }
  });
}

if (exportButton) {
  exportButton.addEventListener("click", () => {
    window.open("/api/export/master", "_blank");
  });
}

if (imageEl) {
  imageEl.addEventListener("load", () => {
    if (zoomPreview) {
      zoomPreview.style.backgroundImage = `url('${imageEl.src}')`;
      zoomPreview.classList.remove("active");
    }
  });
  imageEl.addEventListener("error", resetZoom);
}

if (imageFrame && zoomPreview) {
  imageFrame.addEventListener("mousemove", handleZoomMove);
  imageFrame.addEventListener("mouseleave", resetZoom);
}

function handleZoomMove(event) {
  if (
    !imageEl.complete ||
    !zoomPreview.style.backgroundImage ||
    zoomPreview.style.backgroundImage === "none"
  ) {
    return;
  }
  const imageRect = imageEl.getBoundingClientRect();
  if (!imageRect.width || !imageRect.height) {
    resetZoom();
    return;
  }
  const pointerX = event.clientX - imageRect.left;
  const pointerY = event.clientY - imageRect.top;
  if (pointerX < 0 || pointerY < 0 || pointerX > imageRect.width || pointerY > imageRect.height) {
    resetZoom();
    return;
  }

  const frameRect = imageFrame.getBoundingClientRect();
  const frameX = event.clientX - frameRect.left;
  const frameY = event.clientY - frameRect.top;

  const bgWidth = imageRect.width * ZOOM_SCALE;
  const bgHeight = imageRect.height * ZOOM_SCALE;
  zoomPreview.style.backgroundSize = `${bgWidth}px ${bgHeight}px`;

  const zoomCenterX = pointerX * ZOOM_SCALE;
  const zoomCenterY = pointerY * ZOOM_SCALE;
  const bgPosX = -(zoomCenterX - ZOOM_BOX_SIZE / 2);
  const bgPosY = -(zoomCenterY - ZOOM_BOX_SIZE / 2);
  zoomPreview.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;

  zoomPreview.style.left = `${frameX}px`;
  zoomPreview.style.top = `${frameY}px`;
  zoomPreview.style.transform = "translate(-50%, -50%)";
  zoomPreview.classList.add("active");
}

function resetZoom() {
  if (!zoomPreview) {
    return;
  }
  zoomPreview.classList.remove("active");
  zoomPreview.style.transform = "";
}

loadInitialState().catch((error) => {
  console.error(error);
  alert("Failed to load project state.");
});
