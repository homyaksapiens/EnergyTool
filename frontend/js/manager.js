/* manager.js — панель менеджера.

   Две вкладки:
   - «Заявки»: контакты клиента, резюме диалога, статус (меняется сразу), сам диалог;
   - «Все диалоги»: история всех сессий, включая те, где заявку не оставили (нужно жюри для проверки). */

UI.hydrateIcons();

const STATUSES = {
  new: "Новая",
  in_progress: "В работе",
  done: "Готово",
  rejected: "Отклонена",
};

const $ = (id) => document.getElementById(id);
const view = {
  tab: "leads",      // открытая вкладка
  filter: "all",     // фильтр заявок по статусу
  leads: [],
  sessions: [],
  selected: null,    // id выбранной заявки или сессии
};

// ---------- Загрузка ----------

async function loadAll() {
  setListMessage("Загружаем…");
  try {
    [view.leads, view.sessions] = await Promise.all([Api.listLeads(), Api.listSessions()]);
    $("count-leads").textContent = view.leads.length;
    $("count-sessions").textContent = view.sessions.length;
    renderList();
  } catch (err) {
    setListMessage(`Не удалось загрузить данные: ${err.message}`, true);
  }
}

function setListMessage(text, withRetry = false) {
  $("items").innerHTML = `<li class="items__msg">${UI.esc(text)}
    ${withRetry ? '<button class="btn btn-secondary" type="button" onclick="loadAll()">Повторить</button>' : ""}</li>`;
}

// ---------- Список слева ----------

function renderList() {
  renderFilters();
  const items = $("items");

  if (view.tab === "leads") {
    const list = view.filter === "all" ? view.leads : view.leads.filter((l) => l.status === view.filter);
    if (!list.length) {
      setListMessage(view.leads.length ? "Заявок с таким статусом нет." : "Заявок пока нет. Они появятся, когда клиент оставит контакты в чате.");
      return;
    }
    items.innerHTML = list.map((l) => `
      <li><button type="button" class="item ${view.selected === l.lead_id ? "is-active" : ""}" data-id="${l.lead_id}">
        <span class="item__top">
          <strong>${UI.esc(l.full_name)}</strong>
          <span class="status status--${l.status}">${STATUSES[l.status] || l.status}</span>
        </span>
        <span class="item__sub">${UI.esc(l.company)} · ${UI.esc(l.phone)}</span>
        <span class="item__meta">№${l.lead_id} · ${UI.dateTime(l.created_at)}</span>
      </button></li>`).join("");
  } else {
    if (!view.sessions.length) {
      setListMessage("Диалогов пока нет.");
      return;
    }
    items.innerHTML = view.sessions.map((s) => `
      <li><button type="button" class="item ${view.selected === s.session_id ? "is-active" : ""}" data-id="${s.session_id}">
        <span class="item__top">
          <strong class="item__preview">${UI.esc(s.preview || "Без сообщений")}</strong>
        </span>
        <span class="item__meta">
          ${UI.dateTime(s.updated_at)} · ${s.messages} ${UI.plural(s.messages, ["сообщение", "сообщения", "сообщений"])}
          · ${s.lead_id ? `<span class="tag tag--lead">заявка №${s.lead_id}</span>` : '<span class="tag">без заявки</span>'}
        </span>
      </button></li>`).join("");
  }
}

function renderFilters() {
  const box = $("filters");
  if (view.tab !== "leads") { box.hidden = true; return; }
  box.hidden = false;
  const opts = [["all", "Все"], ...Object.entries(STATUSES)];
  box.innerHTML = opts.map(([v, t]) => {
    const n = v === "all" ? view.leads.length : view.leads.filter((l) => l.status === v).length;
    return `<button type="button" class="filter ${view.filter === v ? "is-active" : ""}" data-filter="${v}" aria-pressed="${view.filter === v}">${t} <span>${n}</span></button>`;
  }).join("");
}

// ---------- Карточка справа ----------

async function showLead(id) {
  const lead = view.leads.find((l) => l.lead_id === id);
  if (!lead) return;
  const detail = $("detail");
  detail.innerHTML = `
    <button type="button" class="btn btn-ghost back">${UI.icon("back")} К списку</button>
    <header class="detail__head">
      <div>
        <h1>${UI.esc(lead.full_name)}</h1>
        <p class="detail__sub">${UI.esc(lead.company)} · заявка №${lead.lead_id} от ${UI.dateTime(lead.created_at)}</p>
      </div>
      <label class="status-select">
        <span class="sr-only">Статус заявки</span>
        <select id="status-select">
          ${Object.entries(STATUSES).map(([v, t]) => `<option value="${v}" ${v === lead.status ? "selected" : ""}>${t}</option>`).join("")}
        </select>
      </label>
    </header>
    <a class="phone" href="tel:${UI.esc(lead.phone.replace(/[^\d+]/g, ""))}">${UI.icon("phone")} ${UI.esc(lead.phone)}</a>
    <p class="saved" id="saved" role="status"></p>
    <h2>Резюме диалога</h2>
    <div class="summary">${UI.miniMarkdown(lead.summary)}</div>
    <h2>Диалог</h2>
    <div class="transcript" id="transcript"><p class="muted">Загружаем диалог…</p></div>`;

  $("status-select").addEventListener("change", (e) => changeStatus(lead, e.target));
  openDetail();
  loadTranscript(lead.session_id, $("transcript"));
}

async function showSession(id) {
  const s = view.sessions.find((x) => x.session_id === id);
  if (!s) return;
  const detail = $("detail");
  detail.innerHTML = `
    <button type="button" class="btn btn-ghost back">${UI.icon("back")} К списку</button>
    <header class="detail__head">
      <div>
        <h1>Диалог от ${UI.dateTime(s.created_at)}</h1>
        <p class="detail__sub">Сессия <code>${UI.esc(s.session_id.slice(0, 8))}</code> ·
          ${s.lead_id ? `<a href="#" data-lead="${s.lead_id}">заявка №${s.lead_id}</a>` : "клиент не оставил контакты"}</p>
      </div>
    </header>
    <div class="transcript" id="transcript"><p class="muted">Загружаем диалог…</p></div>`;
  openDetail();
  loadTranscript(id, $("transcript"));
}

// Переписка целиком; у ответов бота — время и документы базы знаний
async function loadTranscript(sessionId, box) {
  try {
    const s = await Api.getSession(sessionId);
    if (!s.messages.length) { box.innerHTML = '<p class="muted">В диалоге нет сообщений.</p>'; return; }
    box.innerHTML = s.messages.map((m) => {
      const who = m.role === "user" ? "user" : "bot";
      const d = m.debug;
      const meta = d
        ? ` · ${UI.seconds(d.elapsed_ms)} · ${(d.sources || []).length} ${UI.plural((d.sources || []).length, ["документ", "документа", "документов"])}${d.fallback ? " · запасной ответ" : ""}`
        : "";
      const sources = d && d.sources && d.sources.length
        ? `<ul class="t-sources">${d.sources.map((x) => `<li title="${UI.esc(x.source)}">${UI.esc(x.title)}</li>`).join("")}</ul>`
        : "";
      return `<div class="msg msg--${who}">
          <div class="msg__bubble">${UI.esc(m.text).replace(/\n/g, "<br>")}</div>
          <p class="t-meta">${who === "user" ? "Клиент" : "Консультант"} · ${UI.time(m.created_at)}${meta}</p>
          ${sources}
        </div>`;
    }).join("");
  } catch (err) {
    box.innerHTML = `<p class="muted">Не удалось загрузить диалог: ${UI.esc(err.message)}</p>`;
  }
}

async function changeStatus(lead, select) {
  const prev = lead.status;
  const saved = $("saved");
  select.disabled = true;
  try {
    await Api.setLeadStatus(lead.lead_id, select.value);
    lead.status = select.value;
    saved.textContent = "Статус сохранён";
    saved.className = "saved is-ok";
    renderList();
  } catch (err) {
    select.value = prev; // вернуть как было
    saved.textContent = `Не сохранилось: ${err.message}`;
    saved.className = "saved is-err";
  } finally {
    select.disabled = false;
  }
}

// На телефоне карточка закрывает список; «К списку» возвращает назад
function openDetail() {
  $("layout").classList.add("has-detail");
  $("detail").scrollTop = 0;
}

// ---------- Обработчики ----------

document.querySelectorAll("[data-tab]").forEach((tab) => {
  tab.addEventListener("click", () => {
    view.tab = tab.dataset.tab;
    view.selected = null;
    document.querySelectorAll("[data-tab]").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    $("detail").innerHTML = '<div class="empty"><p>Выберите заявку или диалог слева.</p></div>';
    $("layout").classList.remove("has-detail");
    renderList();
  });
});

$("filters").addEventListener("click", (e) => {
  const b = e.target.closest("[data-filter]");
  if (!b) return;
  view.filter = b.dataset.filter;
  renderList();
});

$("items").addEventListener("click", (e) => {
  const b = e.target.closest(".item");
  if (!b) return;
  view.selected = view.tab === "leads" ? Number(b.dataset.id) : b.dataset.id;
  renderList();
  if (view.tab === "leads") showLead(view.selected);
  else showSession(view.selected);
});

$("detail").addEventListener("click", (e) => {
  if (e.target.closest(".back")) $("layout").classList.remove("has-detail");
  const link = e.target.closest("[data-lead]");
  if (link) {
    // Из диалога — перейти к его заявке
    e.preventDefault();
    $("tab-leads").click();
    view.filter = "all"; // чтобы заявка точно была видна в списке
    view.selected = Number(link.dataset.lead);
    renderList();
    showLead(view.selected);
  }
});

$("refresh-btn").addEventListener("click", loadAll);

loadAll();
