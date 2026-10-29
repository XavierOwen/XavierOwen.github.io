const { newState, validateState, createVault, openVault, sealVault, parseEnvelope, todayChicago, addDays, daysBetween, pregnancyStart, rsvSeasonWindow, stageStatus } = await import(`./nest-vault.mjs${new URL(import.meta.url).search}`);

const STORAGE_KEY = "marchen-nest-v1";
const plan = JSON.parse(document.querySelector("#nest-plan-data").textContent);
const byId = id => document.getElementById(id);
const gate = byId("nest-gate"), workspace = byId("nest-workspace");
const gateStatus = byId("nest-gate-status"), saveStatus = byId("nest-save-status"), logStatus = byId("nest-log-status");
let state = null, key = null, salt = null, lastEnvelope = null, pending = Promise.resolve(), saving = 0, dirty = false, conflicted = false, createMode = false;
let storageUnavailable = false;

function text(value) { return String(value ?? ""); }
function escapeHtml(value) { return text(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]); }
function status(node, message, isError = false) { node.textContent = message; node.dataset.error = String(isError); }
function setWorkspaceBusy(busy) { workspace.querySelectorAll("button,input,select,textarea").forEach(control => { control.disabled = busy; }); }
function clearPrivateDom() {
  byId("nest-tasks").replaceChildren();
  byId("nest-log-list").replaceChildren();
  for (const id of ["nest-anchor-date", "nest-birth-date", "log-date", "log-weight", "log-sys", "log-dia", "log-temp", "log-feeds", "log-wet", "log-note"]) byId(id).value = "";
  byId("nest-stage").value = newState().profile.stage;
  document.querySelectorAll('input[name="nest-date-type"]').forEach(input => { input.checked = input.value === "due"; });
  for (const id of ["nest-stage-status", "nest-current-hint", "nest-save-status", "nest-log-status"]) byId(id).textContent = "";
  byId("nest-password").value = "";
  byId("nest-confirm").value = "";
}
function askBackupPassphrase() {
  const dialog = byId("nest-password-dialog"), input = byId("nest-restore-password");
  return new Promise(resolve => {
    const form = byId("nest-password-form"), cancelButton = byId("nest-password-cancel");
    const finish = value => {
      form.removeEventListener("submit", submit);
      cancelButton.removeEventListener("click", cancel);
      dialog.removeEventListener("cancel", dismiss);
      input.value = "";
      dialog.close();
      resolve(value);
    };
    const submit = event => { event.preventDefault(); if (input.reportValidity()) finish(input.value); };
    const cancel = () => finish(null);
    const dismiss = event => { event.preventDefault(); cancel(); };
    form.addEventListener("submit", submit);
    cancelButton.addEventListener("click", cancel);
    dialog.addEventListener("cancel", dismiss);
    dialog.showModal(); input.focus();
  });
}
function storageRead() {
  try { return localStorage.getItem(STORAGE_KEY); }
  catch { storageUnavailable = true; status(gateStatus, "此浏览器的本地存储不可用，无法建立或解锁记录。请启用网站本地存储后重试。", true); return null; }
}
function configureGate() {
  const stored = storageRead();
  const create = byId("nest-create");
  create.hidden = Boolean(stored) || storageUnavailable;
  byId("nest-restore-gate").parentElement.hidden = storageUnavailable;
  byId("nest-unlock-form").hidden = !stored && !createMode;
  byId("nest-confirm-wrap").hidden = !createMode;
  byId("nest-unlock").textContent = createMode ? "建立并保存" : "解锁记录";
  byId("gate-title").textContent = stored ? "解锁本机记录" : createMode ? "设置家庭口令" : "从你们的时间点开始";
  if (!stored && !createMode && !storageUnavailable) status(gateStatus, "没有本机记录。首次使用请建立家庭口令，也可以恢复加密备份。");
}
function taskRecord(id) {
  return state.tasks[id] ||= { done: false, progress: 0, note: "" };
}
function pregnancyDay(today = todayChicago()) {
  const start = pregnancyStart(state.profile);
  return start ? daysBetween(start, today) : null;
}
function inWindow(day, start, end) { return Number.isFinite(day) && start <= day && day <= end; }
function isCurrentSection(section, today) {
  if (section.anchor === "preconception") return state.profile.stage === "preconception";
  if (section.anchor === "pregnancy") {
    if (state.profile.stage !== "pregnancy") return false;
    const day = pregnancyDay(today);
    return inWindow(day, section.start, section.end) || (section.tasks || []).some(task => inWindow(day, Number.isInteger(task.start) ? task.start : section.start, Number.isInteger(task.end) ? task.end : section.end));
  }
  if (section.anchor === "birth") {
    if (state.profile.stage !== "newborn" || !state.profile.birthDate) return false;
    const day = daysBetween(state.profile.birthDate, today);
    return inWindow(day, section.start, section.end) || (section.tasks || []).some(task => inWindow(day, Number.isInteger(task.start) ? task.start : section.start, Number.isInteger(task.end) ? task.end : section.end));
  }
  if (section.anchor === "now") return inWindow(0, section.start, section.end);
  return false;
}
function resolvedDate(section, task, today) {
  const start = Number.isInteger(task.start) ? task.start : section.start;
  const end = Number.isInteger(task.end) ? task.end : section.end;
  let origin = "";
  if (section.anchor === "now") origin = today;
  if (section.anchor === "pregnancy" || section.anchor === "preconception") origin = pregnancyStart(state.profile);
  if (section.anchor === "birth") origin = state.profile.birthDate;
  if (!origin && section.anchor === "preconception" && state.profile.stage === "preconception") return "当前阶段";
  if (!origin || !Number.isFinite(start) || !Number.isFinite(end)) return "日期待填写";
  const from = addDays(origin, start);
  const to = addDays(origin, end);
  if (from === to) return from;
  return `${from} — ${to}`;
}
function renderTasks() {
  const selectedStage = byId("nest-phase").value, stage = selectedStage === "current" ? state.profile.stage : selectedStage, category = byId("nest-category").value, hideDone = byId("nest-hide-done").checked;
  const today = todayChicago(), output = [], hasPregnancyDate = Boolean(pregnancyStart(state.profile));
  byId("nest-current-hint").textContent = selectedStage === "current" && stage === "pregnancy" && !hasPregnancyDate ? "当前阶段：孕期。填写经医疗团队核对的预产期或末次月经第一天后，显示孕周窗口；日期未填时仍列出全部孕期任务。" : selectedStage === "current" && stage === "newborn" && !state.profile.birthDate ? "当前阶段：新生儿/产后。填写出生日期后显示宝宝日龄窗口。" : "";
  for (const section of plan.sections || []) {
    const currentAnchors = stage === "newborn" ? ["birth"] : [stage];
    if (selectedStage !== "all" && !(selectedStage === "current" ? currentAnchors.includes(section.anchor) : section.anchor === stage)) continue;
    const tasks = [];
    for (const task of section.tasks || []) {
      if (category !== "all" && task.category !== category) continue;
      const saved = taskRecord(task.id);
      if (hideDone && saved.done) continue;
      const due = resolvedDate(section, task, today);
      const dateEnd = due.split(" — ").at(-1);
      const past = /^\d{4}-\d{2}-\d{2}$/.test(dateEnd) && dateEnd < today && !saved.done;
      const pastLabel = task.category === "就医" ? "与医生核对是否仍需处理" : "尚未完成 · 按需补做";
      const detail = (task.detail || []).map(item => `<li>${escapeHtml(item)}</li>`).join("");
      const rsvWindow = task.id === "rsv-choice" ? rsvSeasonWindow(state.profile) : null;
      const seasonalNote = rsvWindow ? `<p class="nest__season-note">${rsvWindow.from ? `按填写日期估算，孕周与9–1月重叠的咨询窗口：${rsvWindow.from} — ${rsvWindow.to}。` : "按填写日期估算，孕32–36周6天不在9–1月季节窗口内。请与团队讨论宝宝出生后的预防方案，不自行安排母体RSV接种。"}</p>` : "";
      const sources = (task.sources || []).map(id => plan.sources?.[id] ? `<a href="${escapeHtml(plan.sources[id].url)}" rel="noreferrer">${escapeHtml(plan.sources[id].title)}</a>` : "").filter(Boolean).join(" · ");
      tasks.push(`<article class="nest__task${saved.done ? " nest__task--done" : ""}" data-task="${escapeHtml(task.id)}">
        <div class="nest__task-top"><input class="nest__task-check" type="checkbox" data-task-done="${escapeHtml(task.id)}" aria-label="完成：${escapeHtml(task.title)}"${saved.done ? " checked" : ""}><div><h4 class="nest__task-title">${escapeHtml(task.title)}</h4><p class="nest__task-meta"><span class="nest__task-date" data-past="${past}">${escapeHtml(due)}${past ? ` · ${pastLabel}` : ""}</span><span>${escapeHtml(task.when)}</span>${sources ? `<span>${sources}</span>` : ""}</p></div><span class="nest__tag">${escapeHtml(task.category)}</span></div>
        ${detail ? `<ul class="nest__task-detail">${detail}</ul>` : ""}${seasonalNote}
        ${task.control === "progress" ? `<label class="nest__progress"><span>准备进度（非医学数值）</span><input data-task-progress="${escapeHtml(task.id)}" type="range" min="0" max="100" step="1" value="${saved.progress}" aria-label="${escapeHtml(task.title)}准备进度"><output>${saved.progress}%</output></label>` : ""}
        <details class="nest__task-note"><summary>添加家庭备注</summary><textarea data-task-note="${escapeHtml(task.id)}" maxlength="2000" aria-label="${escapeHtml(task.title)}家庭备注">${escapeHtml(saved.note)}</textarea></details>
      </article>`);
    }
    if (tasks.length) output.push(`<details class="nest__period" data-current="${isCurrentSection(section, today)}"${isCurrentSection(section, today) ? " open" : ""}><summary><span class="nest__period-number">${escapeHtml(section.number)}</span><h3>${escapeHtml(section.title)}</h3><span class="nest__period-subtitle">${escapeHtml(section.subtitle)}</span></summary><div class="nest__period-body">${tasks.join("")}</div></details>`);
  }
  byId("nest-tasks").innerHTML = output.join("") || `<p class="nest__empty">当前筛选下没有清单项目。</p>`;
}
function renderProfile() {
  byId("nest-stage").value = state.profile.stage;
  byId("nest-anchor-date").value = state.profile.anchorDate;
  byId("nest-birth-date").value = state.profile.birthDate;
  document.querySelectorAll('input[name="nest-date-type"]').forEach(input => { input.checked = input.value === state.profile.dateType; });
  const pregnant = state.profile.stage === "pregnancy";
  byId("nest-anchor-wrap").hidden = !pregnant;
  byId("nest-birth-wrap").hidden = state.profile.stage !== "newborn";
  document.querySelector(".nest__date-type").hidden = !pregnant;
  byId("nest-anchor-wrap").firstChild.textContent = state.profile.dateType === "lmp" ? "末次月经第一天" : "预产期";
  const today = todayChicago();
  const phaseName = { preconception: "从零准备／备孕", pregnancy: "孕期", newborn: "新生儿/产后" }[state.profile.stage];
  byId("nest-stage-status").textContent = `当前阶段：${phaseName} · ${stageStatus(state.profile, today).label} · 芝加哥日期 ${today}`;
  renderTasks();
}
function logForDate(date) { return state.logs.find(log => log.date === date); }
function fillLogForm() {
  const log = logForDate(byId("log-date").value);
  byId("log-weight").value = log?.weight || "";
  byId("log-sys").value = log?.bpSys || "";
  byId("log-dia").value = log?.bpDia || "";
  byId("log-temp").value = log?.temp || "";
  byId("log-feeds").value = log?.feeds || "";
  byId("log-wet").value = log?.wet || "";
  byId("log-note").value = log?.note || "";
}
function renderLogs() {
  const logs = [...state.logs].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 15);
  byId("nest-log-list").innerHTML = logs.map(log => `<li><strong>${escapeHtml(log.date)}</strong> · 孕产妇体重 ${escapeHtml(log.weight || "—")} kg · 孕产妇血压 ${escapeHtml(log.bpSys || "—")}/${escapeHtml(log.bpDia || "—")} mmHg · 宝宝体温 ${escapeHtml(log.temp || "—")}℃ · 喂养 ${escapeHtml(log.feeds || "—")} 次 · 湿尿布 ${escapeHtml(log.wet || "—")} 片</li>`).join("");
}
function showWorkspace() {
  setWorkspaceBusy(false);
  gate.hidden = true;
  workspace.hidden = false;
  byId("nest-phase").value = "current";
  byId("log-date").value = todayChicago();
  renderProfile();
  fillLogForm();
  renderLogs();
  window.scrollTo(0, 0);
}
function snapshot() { return validateState(JSON.parse(JSON.stringify(state))); }
function queueSave(node = saveStatus) {
  if (!state || !key || conflicted) return Promise.reject(new Error("记录未能保存：请重新载入最新版本。"));
  dirty = true;
  status(node, "正在加密并保存…");
  const copy = snapshot();
  saving += 1;
  pending = pending.catch(() => {}).then(async () => {
    if (conflicted || localStorage.getItem(STORAGE_KEY) !== lastEnvelope) throw new Error("另一标签页已更改记录，请重新载入以避免覆盖。");
    const previousRaw = lastEnvelope, currentKey = key;
    const envelope = await sealVault(copy, currentKey, salt);
    if (conflicted || key !== currentKey || localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更改记录，本次保存已取消；请重新载入。");
    const serialized = JSON.stringify(envelope);
    localStorage.setItem(STORAGE_KEY, serialized);
    lastEnvelope = serialized;
    dirty = false;
    status(node, "已加密保存到此浏览器。");
  }).catch(error => { dirty = true; status(node, error.message || "保存失败，请导出加密备份后再锁定。", true); throw error; }).finally(() => { saving -= 1; });
  return pending;
}
function saveQuietly(node = saveStatus) { queueSave(node).catch(() => {}); }
function downloadEnvelope(envelope, filename = "nest-backup-encrypted.json") {
  const link = document.createElement("a"), url = URL.createObjectURL(new Blob([JSON.stringify(envelope, null, 2)], { type: "application/json" }));
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function restoreFile(file, passphrase, previousRaw = storageRead()) {
  if (storageUnavailable) throw new Error("此浏览器本地存储不可用，无法恢复记录。");
  if (!file || file.size > 2 * 1024 * 1024) throw new Error("备份文件超过 2 MB 或为空。");
  const envelope = parseEnvelope(await file.text());
  const opened = await openVault(passphrase, envelope);
  if (previousRaw) {
    const old = parseEnvelope(previousRaw);
    downloadEnvelope(old, "nest-previous-encrypted-backup.json");
    if (!confirm("已发起下载旧的加密备份。确定要用所选备份覆盖此浏览器现有记录吗？")) return;
  }
  if (localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更改记录；本次恢复已取消，请刷新后重试。");
  localStorage.setItem(STORAGE_KEY, JSON.stringify(envelope));
  state = opened.state; key = opened.key; salt = opened.salt; lastEnvelope = JSON.stringify(envelope); dirty = false; conflicted = false;
  showWorkspace();
  status(saveStatus, "加密备份已恢复。");
}
function lock() {
  key = null; salt = null; state = null; conflicted = false;
  clearPrivateDom();
  workspace.hidden = true; gate.hidden = false; createMode = false;
  configureGate();
  status(gateStatus, "记录已锁定。个人资料只以密文保存在本浏览器。");
}
function inputNumber(id) { return byId(id).value; }

byId("nest-create").addEventListener("click", () => {
  createMode = true; configureGate(); byId("nest-password").focus();
});
byId("nest-unlock-form").addEventListener("submit", async event => {
  event.preventDefault();
  const password = byId("nest-password").value, previousRaw = storageRead();
  status(gateStatus, "正在打开本机加密记录…");
  byId("nest-unlock").disabled = true;
  try {
    if (createMode) {
      if (previousRaw) throw new Error("另一标签页已建立记录；请刷新后解锁现有记录。");
      if (password !== byId("nest-confirm").value) throw new Error("两次输入的口令不一致。");
      const created = await createVault(password, newState());
      const serialized = JSON.stringify(created.envelope);
      if (localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已建立记录；本次创建已取消。");
      localStorage.setItem(STORAGE_KEY, serialized);
      state = newState(); key = created.key; salt = created.salt; lastEnvelope = serialized;
    } else {
      if (!previousRaw) throw new Error("没有找到本机加密记录。请恢复备份或建立新记录。");
      const opened = await openVault(password, previousRaw);
      if (localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更新记录；请刷新后重新解锁。");
      state = opened.state; key = opened.key; salt = opened.salt; lastEnvelope = previousRaw;
    }
    dirty = false; conflicted = false; createMode = false; showWorkspace();
  } catch (error) { status(gateStatus, error.message || "无法打开记录，请核对口令或备份文件。", true); }
  finally { byId("nest-password").value = ""; byId("nest-confirm").value = ""; byId("nest-unlock").disabled = false; }
});
byId("nest-restore-gate").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 2 * 1024 * 1024) throw new Error("备份文件超过 2 MB。");
    const previousRaw = storageRead();
    if (storageUnavailable) throw new Error("此浏览器本地存储不可用，无法恢复记录。");
    const password = await askBackupPassphrase();
    if (password === null) return;
    await restoreFile(file, password, previousRaw);
  } catch (error) { status(gateStatus, error.message || "恢复失败；原有记录未更改。", true); }
  finally { event.target.value = ""; }
});
byId("nest-lock").addEventListener("click", async () => {
  setWorkspaceBusy(true);
  try { if (dirty) await queueSave(); await pending; lock(); }
  catch { setWorkspaceBusy(false); status(saveStatus, "保存失败，记录仍保持解锁。请导出备份后重试。", true); }
});
function updateProfile(field, value) {
  if (!state || !key || conflicted) return;
  const previous = state.profile[field];
  try {
    state.profile[field] = value;
    snapshot();
    renderProfile();
    saveQuietly();
  } catch (error) {
    state.profile[field] = previous;
    renderProfile();
    status(saveStatus, error.message || "日期无效，请检查后重试。", true);
  }
}
byId("nest-stage").addEventListener("change", event => updateProfile("stage", event.target.value));
byId("nest-anchor-date").addEventListener("change", event => updateProfile("anchorDate", event.target.value));
byId("nest-birth-date").addEventListener("change", event => updateProfile("birthDate", event.target.value));
document.querySelectorAll('input[name="nest-date-type"]').forEach(input => input.addEventListener("change", () => {
  const selected = document.querySelector('input[name="nest-date-type"]:checked');
  if (selected) updateProfile("dateType", selected.value);
}));
for (const id of ["nest-phase", "nest-category", "nest-hide-done"]) byId(id).addEventListener("change", renderTasks);

byId("nest-tasks").addEventListener("change", event => {
  if (!state || !key || conflicted) return;
  const id = event.target.dataset.taskDone || event.target.dataset.taskProgress;
  if (!id) return;
  const record = taskRecord(id);
  if (event.target.matches("[data-task-done]")) { record.done = event.target.checked; event.target.closest(".nest__task").classList.toggle("nest__task--done", record.done); }
  if (event.target.matches("[data-task-progress]")) { record.progress = Number(event.target.value); event.target.nextElementSibling.value = `${record.progress}%`; event.target.nextElementSibling.textContent = `${record.progress}%`; }
  saveQuietly();
  if (event.target.matches("[data-task-done]") && byId("nest-hide-done").checked) renderTasks();
});
byId("nest-tasks").addEventListener("input", event => {
  if (!state || !key || conflicted) return;
  const id = event.target.dataset.taskNote;
  if (!id) return;
  taskRecord(id).note = event.target.value;
  saveQuietly();
});
byId("nest-export").addEventListener("click", async () => {
  if (!state || !key || conflicted) return;
  setWorkspaceBusy(true);
  const copy = snapshot(), currentKey = key, currentSalt = salt;
  try {
    await pending.catch(() => {});
    if (conflicted || key !== currentKey) throw new Error("记录在导出时被另一标签页更改；请刷新后重新解锁。");
    downloadEnvelope(await sealVault(copy, currentKey, currentSalt));
    status(saveStatus, dirty ? "已从当前记录生成加密备份；本机存储尚未保存最新更改。" : "已导出加密备份；文件不包含明文记录。");
  } catch (error) { status(saveStatus, error.message || "无法导出加密备份。", true); }
  finally { if (state && !conflicted) setWorkspaceBusy(false); }
});
byId("nest-restore").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  if (!state || !key || conflicted) { event.target.value = ""; return; }
  setWorkspaceBusy(true);
  try {
    if (file.size > 2 * 1024 * 1024) throw new Error("备份文件超过 2 MB。");
    await pending;
    if (dirty) throw new Error("当前记录仍有未保存更改，请先导出备份并确认保存状态。");
    const previousRaw = lastEnvelope, currentKey = key, currentSalt = salt;
    if (localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更改记录，请刷新后恢复。");
    const password = await askBackupPassphrase();
    if (password === null) return;
    const opened = await openVault(password, parseEnvelope(await file.text()));
    if (conflicted || key !== currentKey || localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更改记录；本次恢复已取消，请刷新后重试。");
    downloadEnvelope(parseEnvelope(previousRaw), "nest-previous-encrypted-backup.json");
    if (!confirm("已发起下载旧的加密备份。确定要覆盖当前记录吗？")) return;
    if (conflicted || key !== currentKey || localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更改记录；本次恢复已取消，请刷新后重试。");
    const replacement = await sealVault(opened.state, currentKey, currentSalt);
    if (conflicted || key !== currentKey || localStorage.getItem(STORAGE_KEY) !== previousRaw) throw new Error("另一标签页已更改记录；本次恢复已取消，请刷新后重试。");
    const serialized = JSON.stringify(replacement);
    localStorage.setItem(STORAGE_KEY, serialized);
    state = opened.state; lastEnvelope = serialized; dirty = false; renderProfile(); fillLogForm(); renderLogs();
    status(saveStatus, "新备份已恢复并加密保存；旧备份已导出。");
  } catch (error) { status(saveStatus, error.message || "恢复失败；原有记录未更改。", true); }
  finally { event.target.value = ""; if (state && !conflicted) setWorkspaceBusy(false); }
});
byId("nest-log-form").addEventListener("submit", event => {
  event.preventDefault();
  if (!state || !key || conflicted) return;
  const date = byId("log-date").value;
  const record = { id: date, date, weight: inputNumber("log-weight"), bpSys: inputNumber("log-sys"), bpDia: inputNumber("log-dia"), temp: inputNumber("log-temp"), feeds: inputNumber("log-feeds"), wet: inputNumber("log-wet"), note: byId("log-note").value };
  try {
    const logs = state.logs.filter(log => log.date !== date);
    logs.push(record);
    state = validateState({ ...state, logs });
    renderLogs();
    saveQuietly(logStatus);
  } catch (error) {
    status(logStatus, `${error.message}。请核对日期并使用普通数字填写；原有记录未改动。`, true);
  }
});
byId("log-date").addEventListener("change", () => { if (state && key && !conflicted) fillLogForm(); });

window.addEventListener("storage", event => {
  if (event.key !== STORAGE_KEY || !key) return;
  conflicted = true; key = null; salt = null; state = null; clearPrivateDom(); workspace.hidden = true; gate.hidden = false; createMode = false; configureGate();
  status(gateStatus, "另一标签页更改了记录。本页已锁定以避免覆盖；请刷新后解锁最新版本。", true);
});
window.addEventListener("beforeunload", event => { if (dirty || conflicted || saving > 0) { event.preventDefault(); event.returnValue = ""; } });

try { storageRead(); configureGate(); }
catch { configureGate(); }
