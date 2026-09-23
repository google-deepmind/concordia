"use strict";
const $ = (id) => document.getElementById(id);
let current = null,
  busy = false,
  online = false,
  rendered = 0,
  revision = -1;
let lastRequest = null;
const storage = {
  get(k) {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(k, v);
    } catch {}
  },
  remove(k) {
    try {
      localStorage.removeItem(k);
    } catch {}
  },
};
const draftKey = "astral-canticle-draft:" + location.pathname;
function saveDraft() {
  storage.set(
    draftKey,
    JSON.stringify({ id: current?.id, value: $("command").value }),
  );
}
function setDraft(value) {
  $("command").value = value;
  saveDraft();
  enable();
}
function enable() {
  $("send").disabled =
    busy || !online || !current || !$("command").value.trim();
}
function showError(text) {
  $("feedback").textContent = text;
}
function render(state) {
  $("status").textContent = state.status;
  if (state.revision === revision) {
    enable();
    return;
  }
  revision = state.revision;
  const nearEnd =
    window.innerHeight + window.scrollY >= document.body.scrollHeight - 240;
  if (state.entries.length < rendered) {
    $("story").replaceChildren();
    rendered = 0;
  }
  for (const entry of state.entries.slice(rendered)) {
    const p = document.createElement("div");
    p.className = entry.kind === "action" ? "player-action" : "passage";
    if (entry.kind === "action") {
      const label = document.createElement("small");
      label.textContent = "YOUR ACTION";
      p.append(label);
    }
    p.append(document.createTextNode(entry.text));
    $("story").append(p);
  }
  const added = rendered !== state.entries.length;
  rendered = state.entries.length;
  $("status").textContent = state.status;
  $("ending").hidden = !state.finished;
  current = state.pending;
  $("prompt").textContent =
    current?.prompt ||
    (state.finished
      ? "Chapter complete"
      : "The world is responding. You can draft your next action.");
  $("command").placeholder =
    current?.type === "float"
      ? "Enter a finite number…"
      : "What do you try next?";
  $("command").inputMode = current?.type === "float" ? "decimal" : "text";
  if (state.role === "gm") {
    $("role-label").textContent = "YOU ARE THE";
    $("character-name").textContent = "Game master";
    $("character-detail").textContent =
      "Shape observations, choose actors, resolve their attempts";
  }
  $("entity-context").hidden = !current?.context;
  $("context").textContent = current?.context || "";
  if (current?.id !== lastRequest) {
    lastRequest = current?.id;
    $("choices").replaceChildren();
    $("choices").hidden = !current?.options?.length;
    for (const option of current?.options || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = option;
      button.onclick = () => {
        setDraft(option);
        $("command").focus();
      };
      $("choices").append(button);
    }
    $("spec-builder").hidden = current?.type !== "next_action_spec";
    if (current?.error) showError(current.error);
  }
  enable();
  if (added && rendered > 1) {
    if (nearEnd)
      $("waiting").scrollIntoView({ block: "end", behavior: "smooth" });
    else $("new-entry").hidden = false;
  }
}
async function poll() {
  try {
    const response = await fetch("api/state", {
      cache: "no-store",
      signal: AbortSignal.timeout(12000),
    });
    if (!response.ok) throw new Error("unavailable");
    const state = await response.json();
    online = true;
    $("connection").textContent = "CONNECTED";
    render(state);
  } catch {
    online = false;
    $("connection").textContent = "RECONNECTING";
    $("status").textContent =
      "Connection interrupted. Your draft is safe; reconnecting…";
    enable();
  } finally {
    setTimeout(poll, document.hidden ? 4000 : 1200);
  }
}
$("action-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!current || busy || !online || !$("command").value.trim()) return;
  busy = true;
  enable();
  showError("");
  const id = current.id,
    value = $("command").value;
  try {
    const response = await fetch("api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Astral-Client": "1" },
      body: JSON.stringify({ request_id: id, response: value }),
      signal: AbortSignal.timeout(15000),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "Please check your response and try again.",
      );
    // A successful POST, including an idempotent retry, is the only point at
    // which a draft is discarded. Network failures never erase human input.
    if ($("command").value === value) {
      $("command").value = "";
      storage.remove(draftKey);
    } else {
      saveDraft();
    }
    current = null;
    revision = -1;
    $("status").textContent = "Your action is unfolding…";
  } catch (error) {
    showError(
      error.name === "TimeoutError" || error instanceof TypeError
        ? "Could not confirm submission. Your draft is kept. Retry when connected."
        : error.message,
    );
  } finally {
    busy = false;
    enable();
  }
});
$("command").addEventListener("input", () => {
  saveDraft();
  enable();
});
$("command").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("action-form").requestSubmit();
  }
});
document.querySelectorAll("[data-command]").forEach(
  (button) =>
    (button.onclick = () => {
      setDraft(button.dataset.command);
      $("command").focus();
    }),
);
$("help-button").onclick = () => {
  $("help").hidden = !$("help").hidden;
  $("help-button").setAttribute("aria-expanded", String(!$("help").hidden));
  if (!$("help").hidden) $("help").scrollIntoView({ block: "center" });
};
$("font").onclick = () => {
  document.body.classList.toggle("large");
  storage.set(
    "astral-large-text",
    document.body.classList.contains("large") ? "1" : "0",
  );
};
$("jump").onclick = () => {
  $("waiting").scrollIntoView({ block: "end", behavior: "smooth" });
  $("new-entry").hidden = true;
};
$("build-spec").onclick = () => {
  const type = $("spec-type").value;
  setDraft(
    JSON.stringify({
      call_to_action: $("spec-prompt").value,
      output_type: type,
      options:
        type === "choice"
          ? $("spec-options")
              .value.split("\n")
              .filter((v) => v.trim())
          : [],
    }),
  );
};
try {
  const draft = JSON.parse(storage.get(draftKey));
  if (draft?.value) $("command").value = draft.value;
} catch {}
if (storage.get("astral-large-text") === "1")
  document.body.classList.add("large");
poll();
