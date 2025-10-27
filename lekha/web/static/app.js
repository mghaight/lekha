const state = {
  view: "line",
  segmentId: null,
  dirty: false,
};

const imageFrame = document.querySelector(".image-frame");
const imageEl = document.getElementById("segment-image");
const textbox = document.getElementById("transcription-box");
const viewRadios = document.querySelectorAll('input[name="view"]');
const buttons = document.querySelectorAll(".actions button");
const zoomPreview = document.getElementById("zoom-preview");

const ZOOM_SCALE = 2.5;
const ZOOM_BOX_SIZE = 180;

function setLoading(isLoading) {
  buttons.forEach((btn) => {
    btn.disabled = isLoading && btn.dataset.action !== "save";
  });
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
  imageEl.src = src;
  imageFrame.classList.toggle("conflict", Boolean(hasConflict));
}

async function loadInitialState() {
  const data = await fetchJSON("/api/state");
  state.view = data.view;
  state.segmentId = data.segment_id;
  for (const radio of viewRadios) {
    radio.checked = radio.value === state.view;
  }
  await loadSegment();
}

async function loadSegment() {
  if (!state.segmentId) {
    return;
  }
  setLoading(true);
  try {
    const data = await fetchJSON(`/api/segment/${state.segmentId}?view=${state.view}`);
    updateImage(data.image_url, data.has_conflict);
    updateTextbox(data.text);
    state.segmentId = data.segment_id;
  } catch (error) {
    console.error(error);
    alert(error.message);
  } finally {
    setLoading(false);
  }
}

async function save(action) {
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
    state.segmentId = data.segment_id;
    if (data.view) {
      state.view = data.view;
      for (const radio of viewRadios) {
        radio.checked = radio.value === state.view;
      }
    }
    updateImage(data.image_url, data.has_conflict);
    updateTextbox(data.text);
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

viewRadios.forEach((radio) => {
  radio.addEventListener("change", async (event) => {
    state.view = event.target.value;
    const data = await fetchJSON("/api/view", {
      method: "POST",
      body: JSON.stringify({view: state.view, segment_id: state.segmentId}),
    });
    state.segmentId = data.segment_id;
    updateImage(data.image_url, data.has_conflict);
    updateTextbox(data.text);
  });
});

textbox.addEventListener("input", () => {
  state.dirty = true;
});

buttons.forEach((button) => {
  button.addEventListener("click", async () => {
    const action = button.dataset.action;
    if (!state.segmentId) {
      return;
    }
    if (!state.dirty && action === "save") {
      alert("No changes to save.");
      return;
    }
    await save(action);
  });
});

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
  if (!imageEl.complete) {
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
