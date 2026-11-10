/* chat.js — чат с консультантом на весь экран.

   Что умеет:
   - приветствие и быстрые кнопки типовых задач;
   - отправка реплики, индикатор ожидания, ответ бота;
   - режим проверки (для жюри, страница открыта с ?debug=1): под ответом время и документы базы знаний;
   - ошибки: «Повторить» без дубля вопроса, потерянная сессия → новый диалог без потери вопроса;
   - заявка прямо в чате: предложение по offer_lead или по кнопке, форма, подтверждение;
   - «Новый диалог» с возможностью вернуть прошлый разговор.

   Снаружи вызываются только Chat.open(...) и Chat.close(). */

const Chat = (() => {
  // ---------- Элементы страницы ----------
  const $ = (id) => document.getElementById(id);
  const el = {
    root: $("chat"),
    scroll: $("chat-scroll"),
    feed: $("chat-feed"),
    status: $("chat-status"),
    announcer: $("chat-announcer"),
    form: $("composer"),
    input: $("composer-input"),
    count: $("composer-count"),
    send: document.querySelector(".composer__send"),
    debugSwitch: $("debug-switch"),
    debug: $("debug-toggle"),
    moreBtn: $("more-btn"),
    menu: $("chat-menu"),
    leadBtn: $("lead-btn"),
    newBtn: $("new-chat-btn"),
    closeBtn: $("close-chat-btn"),
  };

  // ---------- Состояние ----------
  const SESSION_KEY = "et-session-id"; // id сессии живёт до закрытия вкладки

  // Режим проверки виден только по ссылке с ?debug=1 — клиентам он не нужен
  const DEBUG_ALLOWED = new URLSearchParams(location.search).has("debug");

  const state = {
    sessionId: sessionStorage.getItem(SESSION_KEY),
    busy: false,        // ждём ответ — вторую реплику не отправляем
    epoch: 0,           // растёт при «Новом диалоге»: ответы старого диалога выбрасываем
    started: false,     // приветствие уже показано
    leadId: null,       // заявка отправлена — её номер
    leadDeclined: false,// клиент отказался от заявки — больше не предлагаем сами
    opener: null,       // кнопка, которой открыли чат: вернём на неё фокус при закрытии
    undo: null,         // прошлый разговор после «Нового диалога» — можно вернуть
    profile: null,      // вошедший на сайт клиент: { phone } из GET /api/me; гость — null
  };

  const STATUS_IDLE = "ИИ-помощник · отвечает по материалам компании";
  const SLOW_AFTER_MS = 8000;  // после этого показываем «ответ готовится дольше обычного»
  const MAX_LEN = 2000;        // как maxlength у поля ввода
  const COUNT_FROM = 1600;     // с этой длины показываем счётчик символов

  const GREETING =
    "Здравствуйте! Я ИИ-консультант EnergyTool. Помогу разобраться со счетами за электроэнергию, " +
    "учётом и подобрать подходящее решение. Опишите задачу своими словами или выберите вариант ниже.";
  const GREETING_LEAD =
    "Здравствуйте! Чтобы менеджер подготовился к звонку, опишите задачу в паре фраз — " +
    "я приложу к заявке резюме разговора. Если удобнее, можно сразу оставить заявку.";
  const LEAD_NOW = "Сразу оставить заявку";

  // Быстрые кнопки: формулировки клиентов из целевой аудитории, своими словами
  const QUICK = [
    "Вырос счёт за электроэнергию",
    "Хочу видеть расход всех объектов в одном месте",
    "Ищем, где теряется электроэнергия в сети",
    "Думаем о солнечной станции",
  ];

  // ---------- Открыть и закрыть ----------

  // intent: "lead" — пришли с кнопки «Оставить заявку»; ask — сразу задать этот вопрос
  async function open({ intent, ask, opener } = {}) {
    state.opener = opener || document.activeElement;

    // Анимация: чат «вырастает» из кнопки, которой его открыли
    const r = state.opener && state.opener.getBoundingClientRect ? state.opener.getBoundingClientRect() : null;
    el.root.style.setProperty("--ox", r ? `${r.left + r.width / 2}px` : "50%");
    el.root.style.setProperty("--oy", r ? `${r.top + r.height / 2}px` : "50%");
    el.root.hidden = false;
    document.body.classList.add("chat-open");
    setPageInert(true);
    requestAnimationFrame(() => requestAnimationFrame(() => el.root.classList.add("is-open")));

    if (!state.started) await start(intent);
    else if (intent === "lead") showLeadForm();

    if (ask) send(ask);
    else if (!el.feed.querySelector(".lead-card")) el.input.focus({ preventScroll: true });
  }

  function close() {
    closeMenu();
    el.root.classList.remove("is-open");
    document.body.classList.remove("chat-open");
    setPageInert(false);
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) el.root.hidden = true;
    // Иначе спрячем, когда закончится анимация самого чата (обработчик transitionend ниже)
    if (state.opener && state.opener.focus) state.opener.focus();
  }

  // Пока чат открыт, сайт под ним недоступен для Tab и экранной читалки.
  // Панель «Набросок» (.sketch) не трогаем — она нужна поверх чата
  function setPageInert(on) {
    document.querySelectorAll("body > :not(#chat):not(.sketch):not(script)").forEach((node) => {
      node.inert = on;
    });
  }

  // Первое открытие: продолжить сессию из этой вкладки или поздороваться
  async function start(intent) {
    state.started = true;
    state.profile = await loadProfile();
    if (state.sessionId && (await restore(state.sessionId))) {
      if (intent === "lead") showLeadForm();
      return;
    }
    setSession(null);
    greet(intent);
  }

  // Кто открыл чат: вошедший клиент (номер из профиля сайта) или гость.
  // Нет ответа или старый бэкенд без /api/me — считаем гостем, форма спросит телефон
  async function loadProfile() {
    try {
      const me = await Api.me();
      return me && me.authenticated && me.phone ? me : null;
    } catch {
      return null;
    }
  }

  function greet(intent) {
    addBot(intent === "lead" ? GREETING_LEAD : GREETING);
    const chips = document.createElement("div");
    chips.className = "chips";
    chips.setAttribute("role", "group");
    chips.setAttribute("aria-label", "Типовые задачи");
    const items = intent === "lead" ? [LEAD_NOW, ...QUICK] : QUICK;
    for (const text of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (text === LEAD_NOW ? " chip--accent" : "");
      b.textContent = text;
      b.addEventListener("click", () => (text === LEAD_NOW ? showLeadForm() : send(text)));
      chips.append(b);
    }
    el.feed.append(chips);
  }

  // Подтянуть историю сессии с сервера (после перезагрузки страницы)
  async function restore(id) {
    try {
      const s = await Api.getSession(id);
      if (!s.messages.length) return false;
      for (const m of s.messages) {
        if (m.role === "user") addUser(m.text);
        else addBot(m.text, m.debug);
      }
      state.leadId = s.lead_id;
      // Страницу перезагрузили, пока ждали ответ: вопрос есть, ответа нет
      const last = s.messages[s.messages.length - 1];
      if (last.role === "user") {
        addError("Ответ на последний вопрос не успел прийти — страницу обновили, пока он готовился.", last.text, null);
      }
      return true;
    } catch {
      return false; // сессии нет (сервер перезапустили) — просто начнём заново
    }
  }

  function setSession(id) {
    state.sessionId = id;
    if (id) sessionStorage.setItem(SESSION_KEY, id);
    else sessionStorage.removeItem(SESSION_KEY);
  }

  // ---------- «Новый диалог» и возврат прошлого ----------

  function newDialog() {
    closeMenu();
    const hadTalk = Boolean(el.feed.querySelector(".msg--user"));
    // Диалог и так пустой — сбрасывать нечего (и не теряем «Вернуть прошлый»)
    if (!hadTalk && !state.busy) { el.input.focus(); return; }

    // Запоминаем прошлый разговор целиком: узлы ленты и состояние — чтобы вернуть одним нажатием
    state.undo = hadTalk ? {
      nodes: [...el.feed.childNodes],
      sessionId: state.sessionId,
      leadId: state.leadId,
      leadDeclined: state.leadDeclined,
      epoch: state.epoch,
      busy: state.busy,
    } : null;

    state.epoch++; // ответ старого диалога, если он ещё в пути, в новый не попадёт
    setBusy(false);
    el.feed.replaceChildren();
    setSession(null); // новую сессию сервер создаст на первой реплике (POST /api/chat без session_id)
    state.leadId = null;
    state.leadDeclined = false;
    el.input.value = "";
    autosize();
    greet();

    if (hadTalk) {
      const note = document.createElement("p");
      note.className = "note note--undo";
      note.innerHTML = `Начали новый диалог. Прошлый разговор сохранён у менеджера.
        <button type="button" class="link-btn">${UI.icon("undo")} Вернуть прошлый</button>`;
      note.querySelector("button").addEventListener("click", undoNewDialog);
      el.feed.prepend(note);
    }
    announce("Начат новый диалог");
    el.input.focus();
  }

  function undoNewDialog() {
    const u = state.undo;
    if (!u) return;
    state.undo = null;
    el.feed.replaceChildren(...u.nodes);
    setSession(u.sessionId);
    state.leadId = u.leadId;
    state.leadDeclined = u.leadDeclined;
    state.epoch = u.epoch; // ответ, который ещё готовится, снова попадёт в эту ленту
    setBusy(u.busy);
    announce("Прошлый разговор возвращён");
    scrollDown();
    el.input.focus();
  }

  // Как только в новом диалоге написали — вернуть прошлый уже нельзя
  function dropUndo() {
    state.undo = null;
    el.feed.querySelectorAll(".note--undo").forEach((n) => n.remove());
  }

  // ---------- Отправка реплики ----------

  // clientId — метка реплики: повтор после ошибки отправляет ту же метку,
  // и сервер не запишет вопрос второй раз
  async function send(text, { retry = false, clientId = null } = {}) {
    text = text.trim();
    if (!text || state.busy) return;

    // Быстрые кнопки и предложение заявки после новой реплики больше не нужны
    el.feed.querySelectorAll(".chips, .lead-offer").forEach((n) => n.remove());
    dropUndo();
    if (!retry) addUser(text);
    el.input.value = "";
    autosize();

    const id = clientId || newClientId();
    const epoch = state.epoch;
    setBusy(true);
    const typing = addTyping();

    try {
      const res = await Api.chat(state.sessionId, text, id);
      if (epoch !== state.epoch) return; // пока ждали, начали новый диалог
      setSession(res.session_id);
      typing.remove();
      addBot(res.answer, res.debug);
      announce(`Консультант: ${res.answer}`);
      if (res.offer_lead && !state.leadId && !state.leadDeclined) showLeadOffer();
    } catch (err) {
      if (epoch !== state.epoch) return;
      typing.remove();
      if (err.status === 404 && !retry) {
        // Сервер не знает нашу сессию (например, перезапустили) — начинаем новую и повторяем вопрос
        setSession(null);
        addNote("Прошлый диалог на сервере не найден — начали новый и повторили ваш вопрос.");
        setBusy(false);
        return await send(text, { retry: true }); // await: иначе finally ниже сбросит «занят» раньше времени
      }
      addError(err.message, text, id);
    } finally {
      if (epoch === state.epoch) setBusy(false);
    }
  }

  function newClientId() {
    return crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
  }

  function setBusy(busy) {
    state.busy = busy;
    el.status.textContent = busy ? "Подбирает ответ…" : STATUS_IDLE;
    updateComposer();
  }

  // ---------- Сообщения в ленте ----------

  function addUser(text) {
    const div = document.createElement("div");
    div.className = "msg msg--user";
    div.innerHTML = `<div class="msg__bubble">${textToHtml(text)}</div>`;
    el.feed.append(div);
    scrollDown();
  }

  // debug — объект из ответа API; без него (приветствие) блок отладки не рисуем
  function addBot(text, debug) {
    const div = document.createElement("div");
    div.className = "msg msg--bot" + (debug && debug.fallback ? " msg--fallback" : "");
    div.innerHTML = `<div class="msg__bubble">${textToHtml(text)}</div>` + (debug ? debugHtml(debug) : "");
    el.feed.append(div);
    scrollDown();
  }

  // Блок отладки: строка «4,2 с · 3 документа», по клику — подробности
  function debugHtml(d) {
    const n = (d.sources || []).length;
    const docs = n ? `${n} ${UI.plural(n, ["документ", "документа", "документов"])}` : "без документов";
    const t = d.timings || {};
    const stages = [
      t.retrieval != null ? `поиск ${UI.seconds(t.retrieval * 1000)}` : "",
      t.llm != null ? `модель ${UI.seconds(t.llm * 1000)}` : "",
    ].filter(Boolean).join(" · ");
    const list = n
      ? `<ol class="debug__sources">${d.sources.map((s) =>
          `<li><span>${UI.esc(s.title)}</span><code>${UI.esc(s.source)}</code></li>`).join("")}</ol>`
      : `<p class="debug__empty">Документы базы знаний в ответ не попали.</p>`;
    return `
      <div class="debug">
        <button type="button" class="debug__summary" aria-expanded="false">
          ${UI.icon("clock")}<span>${UI.seconds(d.elapsed_ms)}</span>
          ${UI.icon("doc")}<span>${docs}</span>
          ${d.fallback ? '<span class="debug__badge">запасной ответ</span>' : ""}
        </button>
        <div class="debug__details" hidden>
          ${stages ? `<p class="debug__stages">${stages}</p>` : ""}
          ${list}
        </div>
      </div>`;
  }

  // Индикатор «печатает»: через 8 с подсказывает, что ответ готовится дольше обычного
  function addTyping() {
    const div = document.createElement("div");
    div.className = "msg msg--bot msg--typing";
    div.innerHTML = `<div class="msg__bubble"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <span class="typing__text">Подбираю ответ</span></div>`;
    el.feed.append(div);
    scrollDown();
    const timer = setTimeout(() => {
      div.querySelector(".typing__text").textContent = "Ответ готовится дольше обычного, ещё немного";
    }, SLOW_AFTER_MS);
    return { remove: () => { clearTimeout(timer); div.remove(); } };
  }

  // Ошибка: что случилось + «Повторить» тот же вопрос с той же меткой
  function addError(detail, text, clientId) {
    const div = document.createElement("div");
    div.className = "msg msg--bot msg--error";
    div.innerHTML = `<div class="msg__bubble">
        <p class="error__title">${UI.icon("alert")} Не удалось получить ответ</p>
        <p class="error__detail">${UI.esc(detail)}</p>
        <button type="button" class="btn btn-secondary">${UI.icon("retry")} Повторить вопрос</button>
      </div>`;
    div.querySelector("button").addEventListener("click", () => {
      div.remove();
      send(text, { retry: true, clientId });
    });
    el.feed.append(div);
    announce(`Не удалось получить ответ. ${detail}`);
    scrollDown();
  }

  // Служебная заметка по центру ленты (не реплика бота)
  function addNote(text) {
    const p = document.createElement("p");
    p.className = "note";
    p.textContent = text;
    el.feed.append(p);
    announce(text);
    scrollDown();
  }

  // Сказать экранной читалке (текст на экране уже есть, это только озвучка)
  function announce(text) {
    el.announcer.textContent = "";
    requestAnimationFrame(() => { el.announcer.textContent = text; });
  }

  // ---------- Заявка ----------

  // Кнопки под ответом бота, когда сервер прислал offer_lead
  function showLeadOffer() {
    const div = document.createElement("div");
    div.className = "lead-offer";
    div.innerHTML = `
      <button type="button" class="btn btn-primary" data-act="yes">${UI.icon("user")} Оставить заявку</button>
      <button type="button" class="btn btn-ghost" data-act="no">Продолжить без заявки</button>`;
    div.querySelector('[data-act="yes"]').addEventListener("click", () => { div.remove(); showLeadForm(); });
    div.querySelector('[data-act="no"]').addEventListener("click", () => { div.remove(); declineLead(); });
    el.feed.append(div);
    scrollDown();
  }

  // Клиент не хочет оставлять контакты — не настаиваем (демо-вопрос №18)
  function declineLead() {
    state.leadDeclined = true;
    addNote("Хорошо, продолжаем без заявки. Оставить её можно в любой момент — кнопка вверху.");
    el.input.focus();
  }

  function showLeadForm() {
    closeMenu();
    if (state.leadId) {
      addNote(`Заявка №${state.leadId} уже у менеджера. Можно продолжать разговор — он увидит и новые сообщения.`);
      return;
    }
    const existing = el.feed.querySelector(".lead-card");
    if (existing) { existing.querySelector("input").focus(); return; }
    el.feed.querySelectorAll(".chips, .lead-offer").forEach((n) => n.remove());

    const form = document.createElement("form");
    form.className = "lead-card";
    form.noValidate = true; // проверяем сами, чтобы тексты ошибок были на русском и в нашем стиле
    form.setAttribute("aria-labelledby", "lead-title");
    form.innerHTML = `
      <h3 id="lead-title">Заявка менеджеру</h3>
      <p class="lead-card__hint">Менеджер позвонит и продолжит с того места, где мы остановились: резюме разговора придёт ему вместе с заявкой.</p>
      ${field("full_name", "ФИО", "text", "name", "Иванов Иван Иванович")}
      ${state.profile ? phoneFromProfile(state.profile.phone) : field("phone", "Телефон", "tel", "tel", "+7 900 000-00-00")}
      ${field("company", "Компания", "text", "organization", "ООО «Пример»")}
      <div class="check-wrap">
        <label class="check">
          <input type="checkbox" name="consent" aria-describedby="lead-consent-error">
          <span>Согласен, чтобы ФИО, телефон и название компании передали менеджеру для связи по этой заявке.
            <a href="privacy.html" target="_blank" rel="noopener">Как мы храним данные</a></span>
        </label>
        <p class="field__error" id="lead-consent-error" hidden></p>
      </div>
      <p class="form-error" role="alert" hidden></p>
      <div class="lead-card__actions">
        <button type="submit" class="btn btn-primary">Отправить заявку</button>
        <button type="button" class="btn btn-ghost" data-act="cancel">Продолжить без заявки</button>
      </div>`;

    if (state.profile) setupProfilePhone(form);
    const phone = form.elements.phone;
    phone.addEventListener("blur", () => {
      phone.value = formatPhone(phone.value);
      if (phone.value.trim()) checkField(form, "phone"); // подсказываем сразу, а не только при отправке
    });
    form.querySelector('[data-act="cancel"]').addEventListener("click", () => { form.remove(); declineLead(); });
    form.addEventListener("submit", (e) => { e.preventDefault(); submitLead(form); });
    // Ошибка поля исчезает, как только его начали исправлять
    form.addEventListener("input", (e) => {
      if (e.target.getAttribute("aria-invalid") === "true") setFieldError(form, e.target.name, "");
    });

    el.feed.append(form);
    scrollDown();
    form.elements.full_name.focus({ preventScroll: true });
  }

  // Поле: подпись, ввод и место для ошибки, связанное с полем через aria-describedby
  function field(name, label, type, autocomplete, placeholder) {
    const id = `lead-${name}`;
    return `
      <div class="field">
        <label class="field__label" for="${id}">${label}</label>
        <input id="${id}" name="${name}" type="${type}" autocomplete="${autocomplete}" placeholder="${placeholder}"
               ${type === "tel" ? 'inputmode="tel"' : ""} maxlength="200" aria-describedby="${id}-error">
        <p class="field__error" id="${id}-error" hidden></p>
      </div>`;
  }

  // Телефон вошедшего клиента: номер из профиля текстом и кнопка «Другой номер».
  // Поле ввода есть всегда, но спрятано, пока клиента устраивает номер из профиля
  function phoneFromProfile(profilePhone) {
    return `
      <div class="field" data-phone-mode="profile">
        <span class="field__label" id="lead-phone-title">Телефон</span>
        <div class="phone-profile">
          ${UI.icon("phone")}
          <strong>${UI.esc(profilePhone)}</strong>
          <span class="phone-profile__note">из вашего профиля</span>
          <button type="button" class="link-btn" data-act="phone-manual">Другой номер</button>
        </div>
        <div class="phone-manual" hidden>
          <input id="lead-phone" name="phone" type="tel" autocomplete="tel" inputmode="tel" placeholder="+7 900 000-00-00"
                 maxlength="200" aria-labelledby="lead-phone-title" aria-describedby="lead-phone-error">
          <p class="field__error" id="lead-phone-error" hidden></p>
          <button type="button" class="link-btn" data-act="phone-profile">Использовать номер из профиля</button>
        </div>
      </div>`;
  }

  function setupProfilePhone(form) {
    const box = form.querySelector("[data-phone-mode]");
    const title = box.querySelector("#lead-phone-title");
    const input = form.elements.phone;
    const show = (mode) => {
      box.dataset.phoneMode = mode;
      box.querySelector(".phone-profile").hidden = mode !== "profile";
      box.querySelector(".phone-manual").hidden = mode !== "manual";
      title.textContent = mode === "manual" ? "Телефон для связи" : "Телефон";
      setFieldError(form, "phone", "");
    };
    box.querySelector('[data-act="phone-manual"]').addEventListener("click", () => { show("manual"); input.focus(); });
    box.querySelector('[data-act="phone-profile"]').addEventListener("click", () => {
      input.value = "";
      show("profile");
      box.querySelector('[data-act="phone-manual"]').focus();
    });
  }

  // Номер из профиля используется, пока клиент не нажал «Другой номер»
  function usesProfilePhone(form) {
    const box = form.querySelector("[data-phone-mode]");
    return Boolean(box && box.dataset.phoneMode === "profile");
  }

  function leadPhone(form) {
    return usesProfilePhone(form) ? state.profile.phone : formatPhone(form.elements.phone.value);
  }

  // Правила полей. Сервер проверит ещё раз и ответит 422, если что-то не так
  const RULES = {
    full_name: (v) => (v.trim().length < 2 ? "Укажите, как к вам обращаться" : ""),
    phone: (v) => {
      const d = v.replace(/\D/g, "").length;
      return d < 10 || d > 11 ? "Нужен номер из 10–11 цифр, например +7 900 000-00-00" : "";
    },
    company: (v) => (!v.trim() ? "Укажите название компании" : ""),
    consent: (_, input) => (!input.checked ? "Без согласия мы не сможем передать заявку менеджеру" : ""),
  };

  function checkField(form, name) {
    if (name === "phone" && usesProfilePhone(form)) return ""; // номер из профиля уже подтверждён сайтом
    const input = form.elements[name];
    const msg = RULES[name](input.value, input);
    setFieldError(form, name, msg);
    return msg;
  }

  function setFieldError(form, name, msg) {
    const input = form.elements[name];
    const box = form.querySelector(`#${input.getAttribute("aria-describedby")}`);
    input.setAttribute("aria-invalid", msg ? "true" : "false");
    box.hidden = !msg;
    box.textContent = msg;
  }

  function validateLead(form) {
    const bad = Object.keys(RULES).filter((name) => checkField(form, name));
    if (bad.length) {
      form.elements[bad[0]].focus();
      const n = bad.length;
      announce(n === 1 ? "Исправьте одно поле в заявке" : `Исправьте поля в заявке: ${n}`);
    }
    return !bad.length;
  }

  async function submitLead(form) {
    if (!validateLead(form)) return;
    const submit = form.querySelector('[type="submit"]');
    const formError = form.querySelector(".form-error");
    formError.hidden = true;
    submit.disabled = true;
    submit.textContent = "Отправляем…";
    try {
      // Заявке нужна сессия: если клиент ещё ничего не писал — создаём её
      if (!state.sessionId) setSession((await Api.newSession()).session_id);
      const phone = leadPhone(form);
      const res = await Api.createLead({
        session_id: state.sessionId,
        full_name: form.elements.full_name.value.trim(),
        phone,
        company: form.elements.company.value.trim(),
        consent: form.elements.consent.checked,
      });
      state.leadId = res.lead_id;
      form.replaceWith(leadDone(res, phone));
      announce(`Заявка номер ${res.lead_id} отправлена. Менеджер позвонит на ${phone}.`);
      scrollDown();
    } catch (err) {
      formError.hidden = false;
      formError.textContent = err.message; // для 422 — пояснение сервера, что исправить
      submit.disabled = false;
      submit.textContent = "Отправить заявку";
    }
  }

  // Подтверждение: номер заявки и что увидит менеджер
  function leadDone(res, phone) {
    const div = document.createElement("div");
    div.className = "lead-done";
    div.innerHTML = `
      <div class="lead-done__icon">${UI.icon("check")}</div>
      <div>
        <h3>Заявка №${res.lead_id} отправлена</h3>
        <p>Менеджер позвонит на ${UI.esc(phone)}. А пока можно продолжить разговор здесь.</p>
        ${res.summary ? `<details><summary>Что увидит менеджер</summary>
          <div class="lead-done__summary">${UI.miniMarkdown(res.summary)}</div></details>` : ""}
      </div>`;
    // Раскрыли резюме — докручиваем ленту, чтобы его было видно целиком
    const details = div.querySelector("details");
    if (details) details.addEventListener("toggle", () => details.open && scrollDown());
    return div;
  }

  // 89000000000 → +7 900 000-00-00; непонятный номер оставляем как есть
  function formatPhone(raw) {
    let d = raw.replace(/\D/g, "");
    if (d.length === 11 && (d[0] === "8" || d[0] === "7")) d = d.slice(1);
    if (d.length !== 10) return raw.trim();
    return `+7 ${d.slice(0, 3)} ${d.slice(3, 6)}-${d.slice(6, 8)}-${d.slice(8)}`;
  }

  // ---------- Меню «⋯» (видно только на телефоне) ----------

  function toggleMenu() {
    const open = el.moreBtn.getAttribute("aria-expanded") !== "true";
    el.moreBtn.setAttribute("aria-expanded", String(open));
    el.menu.classList.toggle("is-open", open);
    if (open) (DEBUG_ALLOWED ? el.debug : el.newBtn).focus();
  }

  function closeMenu() {
    if (el.moreBtn.getAttribute("aria-expanded") !== "true") return false;
    el.moreBtn.setAttribute("aria-expanded", "false");
    el.menu.classList.remove("is-open");
    return true;
  }

  // ---------- Мелочи ----------

  function textToHtml(text) {
    return UI.esc(text).replace(/\n/g, "<br>");
  }

  function scrollDown() {
    const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    requestAnimationFrame(() => el.scroll.scrollTo({ top: el.scroll.scrollHeight, behavior: smooth ? "smooth" : "auto" }));
  }

  // Поле ввода растёт вместе с текстом, но не выше 160 px
  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = Math.min(el.input.scrollHeight, 160) + "px";
    updateComposer();
  }

  // Кнопка «Отправить» и счётчик символов (появляется ближе к лимиту)
  function updateComposer() {
    el.send.disabled = state.busy || !el.input.value.trim();
    const len = el.input.value.length;
    el.count.hidden = len < COUNT_FROM;
    el.count.textContent = `${len.toLocaleString("ru-RU")} из ${MAX_LEN.toLocaleString("ru-RU")}`;
    el.count.classList.toggle("is-full", len >= MAX_LEN);
  }

  // ---------- Обработчики ----------

  el.form.addEventListener("submit", (e) => { e.preventDefault(); send(el.input.value); });
  el.input.addEventListener("input", autosize);
  el.input.addEventListener("keydown", (e) => {
    // Enter — отправить, Shift+Enter — новая строка
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(el.input.value); }
  });

  el.newBtn.addEventListener("click", newDialog);
  el.leadBtn.addEventListener("click", showLeadForm);
  el.closeBtn.addEventListener("click", close);
  el.moreBtn.addEventListener("click", toggleMenu);

  // Esc: сначала закрывает меню, потом сам чат
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || el.root.hidden) return;
    if (closeMenu()) { el.moreBtn.focus(); return; }
    close();
  });
  // Клик мимо меню — закрыть его
  document.addEventListener("click", (e) => {
    if (!el.menu.contains(e.target) && !el.moreBtn.contains(e.target)) closeMenu();
  });

  el.root.addEventListener("transitionend", (e) => {
    if (e.target === el.root && !el.root.classList.contains("is-open")) el.root.hidden = true;
  });

  // Режим проверки: по ссылке с ?debug=1 переключатель виден и сразу включён
  el.debugSwitch.hidden = !DEBUG_ALLOWED;
  el.debug.checked = DEBUG_ALLOWED;
  el.root.classList.toggle("is-debug", DEBUG_ALLOWED);
  el.debug.addEventListener("change", () => el.root.classList.toggle("is-debug", el.debug.checked));

  // Раскрыть подробности отладки по клику (один обработчик на всю ленту)
  el.feed.addEventListener("click", (e) => {
    const btn = e.target.closest(".debug__summary");
    if (!btn) return;
    const details = btn.nextElementSibling;
    details.hidden = !details.hidden;
    btn.setAttribute("aria-expanded", String(!details.hidden));
  });

  return { open, close };
})();
