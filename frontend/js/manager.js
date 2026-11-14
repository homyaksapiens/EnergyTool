/* Панель менеджера: серверный поиск и страницы. Детали загружаются отдельно. */
UI.hydrateIcons();
const $ = id => document.getElementById(id);
const STATUSES = { new: "Новая", in_progress: "В работе", done: "Готово", rejected: "Отклонена" };
const QUEUES = { all: "Все заявки", active: "Активные", new: "Новые", in_progress: "В работе", done: "Готово", rejected: "Отклонённые" };
const view = { tab: "leads", status: "all", q: "", days: "0", sort: "newest", has_lead: "all", limit: 50, offset: 0, total: 0, items: [], counts: {}, selected: null };
let listRequest = 0, detailRequest = 0, searchTimer, returnFocus;

function params() {
  const p = { q: view.q, days: view.days, sort: view.sort, limit: view.limit, offset: view.offset };
  p[view.tab === "leads" ? "status" : "has_lead"] = view.tab === "leads" ? view.status : view.has_lead;
  return p;
}

function renderQueues() {
  const focused = document.activeElement?.dataset.filter;
  $("sidebar").hidden = view.tab !== "leads";
  $("layout").classList.toggle("is-sessions", view.tab === "sessions");
  $("filters").innerHTML = Object.entries(QUEUES).map(([key, label]) => `<button type="button" class="queue" data-filter="${key}" aria-pressed="${view.status === key}"><span>${label}</span><span class="queue-count">${view.counts[key] ?? "—"}</span></button>`).join("");
  if (focused) $("filters").querySelector(`[data-filter="${focused}"]`)?.focus({ preventScroll: true });
}

function renderHead() {
  const leads = view.tab === "leads";
  $("table-title").textContent = leads ? QUEUES[view.status] : "Все диалоги";
  $("table-caption").textContent = leads ? "Заявки клиентов" : "История диалогов, включая разговоры без заявки";
  $("table-scroll").setAttribute("aria-label", `${leads ? "Таблица заявок" : "Таблица диалогов"}, прокрутка по горизонтали`);
  $("lead-filter-field").hidden = leads;
  $("sort").querySelector('[value="activity"]').hidden = !leads;
  $("table-head").innerHTML = `<tr>${(leads ? ["№", "Клиент / компания", "Задача", "Статус", "Создана"] : ["Диалог", "Клиент / компания", "Сообщения", "Заявка", "Активность"]).map(x => `<th scope="col">${x}</th>`).join("")}</tr>`;
  $("reset-btn").hidden = !view.q && view.days === "0" && view.status === "all" && view.has_lead === "all" && view.sort === "newest";
}

function notice(text, error = false) {
  $("items").innerHTML = `<tr><td colspan="5"><div class="table-message" ${error ? 'role="alert"' : ''}><strong>${UI.esc(text)}</strong>${error ? '<button type="button" class="btn btn-secondary" data-retry>Повторить</button>' : '<p>Попробуйте изменить поиск или сбросить фильтры.</p>'}</div></td></tr>`;
}

function renderRows() {
  if (!view.items.length) { notice(view.q || view.days !== "0" ? "По этим условиям ничего не найдено" : view.tab === "leads" ? "В этой очереди пока нет заявок" : "Диалогов пока нет"); return; }
  $("items").innerHTML = view.items.map(item => {
    const leads = view.tab === "leads", id = leads ? item.lead_id : item.session_id;
    const client = `<strong>${UI.esc(item.full_name || "Без контактов")}</strong><span class="cell-secondary">${UI.esc(item.company || "Компания не указана")}</span>`;
    return `<tr data-id="${UI.esc(id)}" class="${view.selected === id ? "is-selected" : ""}">${leads
      ? `<td><button class="row-link" type="button" data-open="${id}" aria-label="Открыть заявку №${id}">№${id}</button></td><td class="client-cell">${client}</td><td class="task-cell"><span>${UI.esc(item.task)}</span></td><td><span class="status status--${UI.esc(item.status)}">${UI.esc(STATUSES[item.status] || item.status)}</span></td><td class="date-cell">${UI.dateTime(item.created_at)}</td>`
      : `<td class="task-cell"><button class="row-link dialog-link" type="button" data-open="${UI.esc(id)}">${UI.esc(item.preview || "Без текста")}</button><span class="cell-secondary">${UI.esc(id.slice(0, 8))}</span></td><td class="client-cell">${client}</td><td>${item.messages}</td><td>${item.lead_id ? `№${item.lead_id}` : '<span class="muted">Без заявки</span>'}</td><td class="date-cell">${UI.dateTime(item.updated_at)}</td>`}</tr>`;
  }).join("");
}

function pagination() {
  $("page-range").textContent = view.total ? `${view.offset + 1}–${Math.min(view.offset + view.limit, view.total)} из ${view.total}` : "0 записей";
  $("prev-page").disabled = view.offset === 0;
  $("next-page").disabled = view.offset + view.limit >= view.total;
}

async function loadPage(reset = false) {
  clearTimeout(searchTimer);
  const scrollTop = reset ? 0 : $("table-scroll").scrollTop;
  const scrollLeft = $("table-scroll").scrollLeft;
  if (reset) view.offset = 0;
  const request = ++listRequest;
  renderHead();
  $("table-scroll").setAttribute("aria-busy", "true");
  $("refresh-btn").disabled = true;
  $("prev-page").disabled = $("next-page").disabled = true;
  $("result-count").textContent = "Загружаем…";
  $("items").innerHTML = Array.from({ length: 6 }, () => '<tr aria-hidden="true"><td colspan="5"><div class="skeleton-row"></div></td></tr>').join("");
  try {
    const data = await (view.tab === "leads" ? Api.managerLeads(params()) : Api.managerSessions(params()));
    if (request !== listRequest) return;
    view.total = data.total;
    if (view.offset && view.offset >= data.total) {
      view.offset = Math.max(0, Math.ceil(data.total / view.limit) - 1) * view.limit;
      return loadPage();
    }
    view.items = data.items;
    view.counts = data.counts || {};
    $("result-count").textContent = `Найдено: ${data.total}`;
    $(view.tab === "leads" ? "count-leads" : "count-sessions").textContent = view.tab === "leads" ? data.counts.all : data.total;
    renderQueues(); renderRows(); pagination();
    $("table-scroll").scrollTop = scrollTop;
    $("table-scroll").scrollLeft = scrollLeft;
  } catch (err) {
    if (request !== listRequest) return;
    $("result-count").textContent = "Данные не загружены";
    $("page-range").textContent = "";
    notice(`Не удалось загрузить: ${err.message}`, true);
  } finally {
    if (request === listRequest) {
      $("table-scroll").removeAttribute("aria-busy");
      $("refresh-btn").disabled = false;
    }
  }
}

function showDetail(trigger, animate = false) {
  returnFocus = trigger || document.activeElement;
  $("detail").hidden = false;
  $("workspace").inert = true;
  $("sidebar").inert = true;
  $("layout").classList.add("has-detail");
  // Частая навигация мгновенная; короткое появление только при открытии мышью.
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    $("detail").animate([{ opacity: 0, transform: "translateX(16px)" }, { opacity: 1, transform: "translateX(0)" }], { duration: 180, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
  }
  $("close-detail").focus({ preventScroll: true });
}

function closeDetail(restoreFocus = true) {
  ++detailRequest;
  $("detail").hidden = true;
  $("workspace").inert = false;
  $("sidebar").inert = false;
  $("layout").classList.remove("has-detail");
  view.selected = null;
  document.querySelectorAll("tr.is-selected").forEach(row => row.classList.remove("is-selected"));
  if (restoreFocus) {
    if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
    else $("table-scroll").focus({ preventScroll: true });
  }
}

async function openItem(id, trigger, animate = false, forceLead = false) {
  const leads = forceLead || view.tab === "leads";
  view.selected = leads ? Number(id) : id;
  document.querySelectorAll("#items tr[data-id]").forEach(row => row.classList.toggle("is-selected", row.dataset.id === String(id)));
  const request = ++detailRequest;
  showDetail(trigger, animate);
  $("detail-label").textContent = leads ? `Заявка №${id}` : "История диалога";
  $("detail-content").innerHTML = '<h2 id="detail-title">Загрузка…</h2><div class="skeleton-row" aria-hidden="true"></div>';
  try {
    const item = await (leads ? Api.managerLead(id) : Api.getSession(id));
    if (request !== detailRequest) return;
    const box = $("detail-content");
    box.innerHTML = leads ? `<header class="detail__head"><h2 id="detail-title">${UI.esc(item.full_name)}</h2><p class="muted">${UI.esc(item.company)} · ${UI.dateTime(item.created_at)}</p></header>
      <a class="phone" href="tel:${UI.esc(item.phone.replace(/[^\d+]/g, ""))}">${UI.icon("phone")} ${UI.esc(item.phone)}</a>
      <label class="status-select">Статус заявки<select id="status-select">${Object.entries(STATUSES).map(([value, text]) => `<option value="${value}" ${value === item.status ? "selected" : ""}>${text}</option>`).join("")}</select></label>
      <p class="saved" id="saved" role="status"></p><h3>Резюме диалога</h3><div class="summary">${UI.miniMarkdown(item.summary)}</div><h3>Переписка</h3><div class="transcript" id="transcript">Загружаем переписку…</div>`
      : `<header class="detail__head"><h2 id="detail-title">Диалог от ${UI.dateTime(item.created_at)}</h2><p class="muted">${item.lead_id ? `<button type="button" class="link-btn" data-lead="${item.lead_id}">Открыть заявку №${item.lead_id}</button>` : "Без заявки"}</p></header><div class="transcript" id="transcript"></div>`;
    box.scrollTop = 0;
    if (leads) {
      $("status-select").addEventListener("change", e => changeStatus(item, e.target));
      await loadTranscript(item.session_id, $("transcript"));
    } else renderTranscript(item, $("transcript"));
  } catch (err) {
    if (request !== detailRequest) return;
    $("detail-content").innerHTML = `<h2 id="detail-title">Не удалось открыть карточку</h2><p role="alert">${UI.esc(err.message)}</p><button class="btn btn-secondary" type="button" data-detail-retry>Повторить</button>`;
    $("detail-content").querySelector('[data-detail-retry]').onclick = () => openItem(id, trigger, false, forceLead);
  }
}

function renderTranscript(s, box) {
  if (!s.messages.length) { box.innerHTML = '<p class="muted">В диалоге нет сообщений.</p>'; return; }
  box.innerHTML = s.messages.map(m => {
    const bot = m.role !== "user", d = m.debug;
    return `<div class="msg msg--${bot ? "bot" : "user"}"><div class="msg__bubble">${UI.esc(m.text).replace(/\n/g, "<br>")}</div>
      <p class="t-meta">${bot ? "Консультант" : "Клиент"} · ${UI.time(m.created_at)}${d ? ` · ${UI.seconds(d.elapsed_ms)}${d.fallback ? " · запасной ответ" : ""}` : ""}</p>
      ${d?.sources?.length ? `<details class="t-sources"><summary>Источники: ${d.sources.length}</summary><ul>${d.sources.map(x => `<li title="${UI.esc(x.source)}">${UI.esc(x.title)}</li>`).join("")}</ul></details>` : ""}</div>`;
  }).join("");
}

async function loadTranscript(id, box) {
  try { renderTranscript(await Api.getSession(id), box); }
  catch (err) { box.innerHTML = `<p role="alert">Не удалось загрузить переписку: ${UI.esc(err.message)}</p><button type="button" class="btn btn-secondary">Повторить</button>`; box.querySelector("button").onclick = () => loadTranscript(id, box); }
}

async function changeStatus(lead, select) {
  const prev = lead.status, saved = $("saved");
  select.disabled = true;
  try {
    await Api.setLeadStatus(lead.lead_id, select.value);
    lead.status = select.value;
    saved.textContent = "Статус сохранён"; saved.className = "saved";
    await loadPage();
  } catch (err) {
    select.value = prev;
    saved.textContent = `Не сохранилось: ${err.message}`; saved.className = "saved is-err";
  } finally { select.disabled = false; }
}

document.querySelectorAll("[data-tab]").forEach(tab => tab.addEventListener("click", () => {
  closeDetail(false);
  view.tab = tab.dataset.tab; view.status = "all"; view.sort = "newest"; $("sort").value = "newest";
  document.querySelectorAll("[data-tab]").forEach(t => t.setAttribute("aria-pressed", String(t === tab)));
  loadPage(true);
}));
$("filters").addEventListener("click", e => { const b = e.target.closest("[data-filter]"); if (b) { view.status = b.dataset.filter; renderQueues(); loadPage(true); } });
$("items").addEventListener("click", e => {
  if (e.target.closest("[data-retry]")) { loadPage(); return; }
  const row = e.target.closest("tr[data-id]");
  if (row && !getSelection().toString()) openItem(row.dataset.id, row.querySelector("button"), e.detail > 0);
});
$("detail-content").addEventListener("click", e => { const link = e.target.closest("[data-lead]"); if (link) openItem(link.dataset.lead, returnFocus, false, true); });
$("close-detail").addEventListener("click", () => closeDetail());
document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("detail").hidden && !document.querySelector("dialog[open]")) { e.preventDefault(); closeDetail(); } });
$("search").addEventListener("input", () => { clearTimeout(searchTimer); ++listRequest; view.q = $("search").value.trim(); searchTimer = setTimeout(() => loadPage(true), 300); });
$("search-form").addEventListener("submit", e => { e.preventDefault(); view.q = $("search").value.trim(); loadPage(true); });
for (const [id, key] of [["period", "days"], ["sort", "sort"], ["has-lead", "has_lead"], ["page-size", "limit"]]) {
  $(id).addEventListener("change", () => { view[key] = id === "page-size" ? Number($(id).value) : $(id).value; loadPage(true); });
}
$("reset-btn").addEventListener("click", () => {
  Object.assign(view, { q: "", days: "0", status: "all", has_lead: "all", sort: "newest" });
  $("search").value = ""; $("period").value = "0"; $("sort").value = "newest"; $("has-lead").value = "all";
  loadPage(true);
});
$("prev-page").addEventListener("click", () => { view.offset = Math.max(0, view.offset - view.limit); $("table-scroll").scrollTop = 0; loadPage(); });
$("next-page").addEventListener("click", () => { view.offset += view.limit; $("table-scroll").scrollTop = 0; loadPage(); });
$("refresh-btn").addEventListener("click", () => loadPage());
Auth.ready.then(() => { if (!Auth.enabled || Auth.user?.role === "admin") loadPage(); });
