/* chat.js — чат с консультантом на весь экран.

   Что умеет:
   - приветствие и быстрые кнопки типовых задач;
   - отправка реплики, индикатор ожидания, ответ бота;
   - режим проверки (для жюри, страница открыта с ?debug=1): под ответом время и документы базы знаний;
   - ошибки: «Повторить» без дубля вопроса, потерянная сессия → новый диалог без потери вопроса;
   - заявка прямо в чате: предложение по offer_lead или по кнопке, форма, подтверждение;
   - «Новый диалог» с возможностью вернуть прошлый разговор;
   - слева «Ваши диалоги»: прошлые диалоги вошедшего клиента (GET /api/me/sessions), любой можно открыть и продолжить;
   - справа оглавление: вопросы клиента в этом разговоре, по клику — прокрутка к вопросу.

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
    dialogs: $("dialogs"),             // левая колонка
    dialogsList: $("dialogs-list"),
    dialogsHint: $("dialogs-hint"),
    dialogsBusy: $("dialogs-busy"),    // «Другие диалоги откроются, когда консультант ответит»
    dialogsBtn: $("dialogs-btn"),      // открывает левую колонку на узком экране
    backdrop: $("dialogs-backdrop"),
    outline: $("outline"),             // правая колонка — оглавление
    outlineList: $("outline-list"),
    main: document.querySelector(".chat__main"), // центр: лента и поле ввода
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
    dialogs: [],        // прошлые диалоги вошедшего клиента из GET /api/me/sessions
    dialogsError: false,// список не загрузился
    pending: 0,         // сколько вопросов ещё ждут ответа сервера — в этом или в оставленном диалоге
  };
  let askSeq = 0;       // счётчик для id вопросов клиента: на них ссылается оглавление
  let dialogsSeq = 0;   // номер последнего запроса списка диалогов
  let outlinePinned = null, pinTimer = 0; // пункт оглавления, к которому сейчас прокручиваем

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
    closeDrawer();
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
    } else {
      setSession(null);
      greet(intent);
    }
    loadDialogs(); // без await: список слева подгрузится, пока клиент читает приветствие
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

  // Подтянуть историю сессии с сервера (после перезагрузки страницы или из списка слева).
  // Пока ждали ответ, клиент мог открыть другой диалог — тогда эту историю в ленту не выводим
  async function restore(id) {
    const epoch = state.epoch;
    try {
      const s = await Api.getSession(id);
      if (epoch !== state.epoch) return true; // уже открыт другой диалог — он сам отвечает за ленту
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
    closeDrawer();
    const hadTalk = Boolean(el.feed.querySelector(".msg--user"));
    // Диалог и так пустой — сбрасывать нечего (и не теряем «Вернуть прошлый»).
    // Но если прошлый диалог ещё грузится из списка слева — сбрасываем, чтобы он не появился
    if (!hadTalk && !state.busy && !state.sessionId) { el.input.focus(); return; }

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
    rebuildOutline();
    renderDialogs();
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
    rebuildOutline();
    renderDialogs();
    announce("Прошлый разговор возвращён");
    scrollDown();
    el.input.focus();
  }

  // Как только в новом диалоге написали — вернуть прошлый уже нельзя
  function dropUndo() {
    state.undo = null;
    el.feed.querySelectorAll(".note--undo").forEach((n) => n.remove());
    updateDialogsLock();
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

    state.pending++;
    updateDialogsLock();
    try {
      const res = await Api.chat(state.sessionId, text, id);
      // Пока ждали, начали новый диалог: ответ кладём в сохранённый прошлый разговор («Вернуть прошлый»)
      if (epoch !== state.epoch) { parkAnswer(epoch, typing, botNode(res.answer, res.debug), res.session_id); return; }
      setSession(res.session_id);
      typing.remove();
      addBot(res.answer, res.debug);
      announce(`Консультант: ${res.answer}`);
      if (res.offer_lead && !state.leadId && !state.leadDeclined) showLeadOffer();
      loadDialogs(); // новый диалог появится слева, у старого обновится время
    } catch (err) {
      if (epoch !== state.epoch) {
        const note = document.createElement("p");
        note.className = "note";
        note.textContent = "Ответ на последний вопрос не пришёл — задайте его ещё раз.";
        parkAnswer(epoch, typing, note);
        return;
      }
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
      state.pending--;
      if (epoch === state.epoch) setBusy(false);
      updateDialogsLock();
    }
  }

  // Ответ пришёл, когда клиент уже начал новый диалог. Если прошлый разговор ещё можно вернуть —
  // убираем из него «печатает» и кладём ответ, иначе после «Вернуть прошлый» чат навсегда остался бы «занят»
  // sessionId — id сессии из ответа: у первого вопроса нового диалога его ещё не было
  function parkAnswer(epoch, typing, node, sessionId) {
    typing.remove();
    const u = state.undo;
    if (!u || u.epoch !== epoch) return;
    u.nodes = u.nodes.filter((n) => n !== typing.el);
    if (node) u.nodes.push(node);
    if (sessionId) u.sessionId = sessionId;
    u.busy = false;
    updateDialogsLock();
  }

  function newClientId() {
    return crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
  }

  function setBusy(busy) {
    state.busy = busy;
    el.status.textContent = busy ? "Подбирает ответ…" : STATUS_IDLE;
    updateComposer();
    updateDialogsLock();
  }

  // Хоть один вопрос ещё ждёт ответа — здесь или в диалоге, из которого ушли
  function answerPending() {
    return state.pending > 0;
  }

  // Пока ответ готовится, другие диалоги в списке слева приглушены и не открываются.
  // Пояснение под заголовком показываем только после попытки открыть такой диалог (openDialog),
  // чтобы список не прыгал на каждом вопросе
  function updateDialogsLock() {
    const locked = answerPending();
    el.dialogs.classList.toggle("is-locked", locked);
    if (!locked) el.dialogsBusy.hidden = true;
    el.dialogsList.querySelectorAll(".dialog").forEach((b) => {
      if (locked && b.getAttribute("aria-current") !== "true") b.setAttribute("aria-disabled", "true");
      else b.removeAttribute("aria-disabled");
    });
  }

  // ---------- Сообщения в ленте ----------

  function addUser(text) {
    const div = document.createElement("div");
    div.className = "msg msg--user";
    div.id = `ask-${++askSeq}`; // на этот id ссылается пункт оглавления справа
    div.innerHTML = `<div class="msg__bubble">${textToHtml(text)}</div>`;
    el.feed.append(div);
    rebuildOutline();
    scrollDown();
  }

  // debug — объект из ответа API; без него (приветствие) блок отладки не рисуем
  function addBot(text, debug) {
    el.feed.append(botNode(text, debug));
    scrollDown();
  }

  // Реплика бота без вставки в ленту (нужна и для ответа, отложенного в «Вернуть прошлый»)
  function botNode(text, debug) {
    const div = document.createElement("div");
    div.className = "msg msg--bot" + (debug && debug.fallback ? " msg--fallback" : "");
    div.innerHTML = `<div class="msg__bubble">${textToHtml(text)}</div>` + (debug ? debugHtml(debug) : "");
    return div;
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
    return { el: div, remove: () => { clearTimeout(timer); div.remove(); } };
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
    closeDrawer();
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
    const phone = leadPhone(form);
    let sessionId = null; // заявка относится к этому диалогу, даже если клиент откроет другой
    try {
      // Заявке нужна сессия: если клиент ещё ничего не писал — создаём её
      if (!state.sessionId) setSession((await Api.newSession()).session_id);
      sessionId = state.sessionId;
      // Прошлая попытка оборвалась без ответа, а сервер мог дописать заявку позже —
      // сначала сверяемся с сессией и отправляем, только если заявки там ещё нет
      if (form.dataset.unsure === "1") {
        const s = await Api.getSession(sessionId);
        if (s.lead_id) { leadSent(form, { lead_id: s.lead_id, summary: null }, phone, sessionId); return; }
      }
      const res = await Api.createLead({
        session_id: sessionId,
        full_name: form.elements.full_name.value.trim(),
        phone,
        company: form.elements.company.value.trim(),
        consent: form.elements.consent.checked,
      });
      leadSent(form, res, phone, sessionId);
    } catch (err) {
      // Связь оборвалась или сервер не ответил вовремя — заявка могла всё-таки записаться.
      // Сверяемся с сессией: если заявка есть, показываем подтверждение, а не «Попробуйте ещё раз»,
      // иначе повторное нажатие создало бы у менеджера вторую заявку
      if (err.status === 0 && sessionId) {
        try {
          const s = await Api.getSession(sessionId);
          if (s.lead_id) { leadSent(form, { lead_id: s.lead_id, summary: null }, phone, sessionId); return; }
        } catch { /* сверить не удалось — показываем исходную ошибку */ }
        form.dataset.unsure = "1"; // при следующем нажатии сначала проверим, не дошла ли заявка
      }
      formError.hidden = false;
      // Для 422 — пояснение сервера, что исправить. Для обрыва связи — что повтор не создаст вторую заявку
      formError.textContent = err.status === 0 && sessionId
        ? "Не удалось подтвердить отправку. Нажмите «Отправить заявку» ещё раз — мы проверим, дошла ли она, и не отправим её дважды."
        : err.message;
      submit.disabled = false;
      submit.textContent = "Отправить заявку";
    }
  }

  // Заявка записана: подтверждение вместо формы
  function leadSent(form, res, phone, sessionId) {
    loadDialogs(); // у диалога слева появится «заявка №…»
    if (state.sessionId !== sessionId) {
      // Пока отправляли, клиент ушёл в другой диалог. Если этот разговор можно вернуть («Вернуть прошлый») —
      // кладём в него подтверждение вместо формы, иначе форма зависла бы на «Отправляем…»
      const u = state.undo;
      if (u && u.sessionId === sessionId) {
        u.leadId = res.lead_id;
        u.nodes = u.nodes.map((n) => (n === form ? leadDone(res, phone) : n));
      }
      return;
    }
    state.leadId = res.lead_id;
    form.replaceWith(leadDone(res, phone));
    announce(`Заявка номер ${res.lead_id} отправлена. Менеджер позвонит на ${phone}.`);
    scrollDown();
  }

  // Подтверждение: номер заявки и что увидит менеджер
  function leadDone(res, phone) {
    const div = document.createElement("div");
    div.className = "lead-done";
    div.innerHTML = `
      <div class="lead-done__icon">${UI.icon("check")}</div>
      <div>
        <h3>Заявка №${UI.esc(res.lead_id)} отправлена</h3>
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

  // ---------- Слева: «Ваши диалоги» ----------

  // Список прошлых диалогов вошедшего клиента. Общий GET /api/sessions здесь не используем:
  // в нём разговоры всех клиентов. Гость видит подсказку, как получить свой список
  async function loadDialogs() {
    const seq = ++dialogsSeq; // запросы могут вернуться не по порядку — показываем только последний
    if (!state.profile) {
      state.dialogs = [];
      state.dialogsError = false;
    } else {
      try {
        const list = await Api.mySessions();
        if (seq !== dialogsSeq) return;
        state.dialogs = list;
        state.dialogsError = false;
      } catch {
        if (seq !== dialogsSeq) return;
        state.dialogsError = true; // старый список оставляем, под ним — «Повторить»
      }
    }
    renderDialogs();
  }

  function renderDialogs() {
    // Список перерисовывается после каждого ответа: возвращаем фокус клавиатуры на тот же пункт
    const focusedId = el.dialogsList.contains(document.activeElement) ? document.activeElement.dataset.id : null;
    const items = [...state.dialogs];
    // Текущий разговор уже идёт, а в списке с сервера его ещё нет (первый ответ в пути, гость) — показываем сверху
    const firstAsk = el.feed.querySelector(".msg--user .msg__bubble");
    if (state.sessionId && firstAsk && !items.some((d) => d.session_id === state.sessionId)) {
      items.unshift({ session_id: state.sessionId, updated_at: new Date().toISOString(), lead_id: state.leadId, preview: firstAsk.textContent });
    }
    el.dialogsList.replaceChildren(...items.map(dialogItem));
    if (focusedId) {
      const again = [...el.dialogsList.querySelectorAll(".dialog")].find((b) => b.dataset.id === focusedId);
      if (again) again.focus({ preventScroll: true });
    }

    // Подсказка под списком: ошибка, гость или пока пусто
    let hint = "";
    if (state.dialogsError) {
      hint = `Не удалось загрузить прошлые диалоги.<br><button type="button" class="link-btn" data-act="reload">${UI.icon("retry")} Повторить</button>`;
    } else if (!state.profile) {
      hint = "Войдите на сайт по номеру телефона — здесь появятся ваши прошлые диалоги.";
    } else if (!items.length) {
      hint = "Здесь появятся ваши диалоги с консультантом.";
    }
    el.dialogsHint.hidden = !hint;
    el.dialogsHint.innerHTML = hint;
    updateDialogsLock();
    updateListEdge();
  }

  // Нижний край списка растворяется, пока ниже есть ещё диалоги (класс .is-overflow в chat.css)
  function updateListEdge() {
    const l = el.dialogsList;
    l.classList.toggle("is-overflow", l.scrollTop + l.clientHeight < l.scrollHeight - 2);
  }

  // Пункт списка: первый вопрос диалога, дата и номер заявки, если она есть
  function dialogItem(d) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.className = "dialog";
    b.dataset.id = d.session_id;
    if (d.session_id === state.sessionId) b.setAttribute("aria-current", "true");
    const lead = d.lead_id ? ` · <span class="dialog__lead">заявка №${UI.esc(d.lead_id)}</span>` : "";
    b.innerHTML = `<span class="dialog__title">${UI.esc(d.preview || "Диалог без вопросов")}</span>
      <span class="dialog__meta">${UI.dateTime(d.updated_at || d.created_at)}${lead}</span>`;
    b.addEventListener("click", () => openDialog(d.session_id));
    li.append(b);
    return li;
  }

  // Открыть прошлый диалог: лента заменяется его историей, разговор можно продолжить
  async function openDialog(id) {
    // Пока консультант отвечает (здесь или в прошлом разговоре после «Нового диалога»), диалог не переключаем:
    // ответ потерялся бы на экране, а «Повторить» записал бы вопрос на сервере второй раз.
    // Пункты списка в это время приглушены; после попытки под заголовком появляется пояснение.
    // Проверка — до закрытия выехавшего списка, иначе на узком экране пояснение спряталось бы вместе с ним
    if (id !== state.sessionId && answerPending()) {
      el.dialogsBusy.hidden = false;
      announce("Другие диалоги откроются, когда консультант ответит");
      return;
    }
    closeDrawer();
    if (id === state.sessionId) { el.input.focus({ preventScroll: true }); return; }
    const epoch = ++state.epoch; // ответы и история прошлых нажатий в эту ленту уже не попадут
    dropUndo();
    state.leadId = null;
    state.leadDeclined = false;
    setSession(id);
    const loading = document.createElement("p");
    loading.className = "note";
    loading.textContent = "Загружаем диалог…";
    el.feed.replaceChildren(loading);
    rebuildOutline();
    renderDialogs();

    const ok = await restore(id);
    if (epoch !== state.epoch) return; // пока грузили, клиент открыл ещё один диалог
    loading.remove();
    if (ok) {
      announce("Открыт прошлый диалог");
    } else {
      setSession(null);
      greet();
      addNote("Не получилось открыть этот диалог. Начали новый.");
    }
    rebuildOutline();
    renderDialogs();
    el.input.focus({ preventScroll: true });
  }

  // ---------- Справа: оглавление разговора ----------

  // Пункты — вопросы клиента в ленте. Перестраиваем целиком: вопросов немного
  function rebuildOutline() {
    const asks = [...el.feed.querySelectorAll(".msg--user")];
    el.outlineList.replaceChildren(...asks.map((msg) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "outline__item";
      b.dataset.target = msg.id;
      b.innerHTML = `<span>${UI.esc(msg.textContent.trim())}</span>`;
      li.append(b);
      return li;
    }));
    el.outline.hidden = !asks.length;
    markOutline();
  }

  // Подсветить вопрос, до которого дочитали: последний, чей верх уже выше 40 % высоты ленты
  function markOutline() {
    const items = [...el.outlineList.querySelectorAll(".outline__item")];
    if (!items.length || outlinePinned) return; // идёт прокрутка к нажатому вопросу — его подсветку не трогаем
    const box = el.scroll.getBoundingClientRect();
    const line = box.top + box.height * 0.4;
    let current = items[0];
    for (const b of items) {
      const msg = document.getElementById(b.dataset.target);
      if (msg && msg.getBoundingClientRect().top <= line) current = b;
    }
    // Долистали до конца — подсвечиваем последний вопрос, даже если он ниже линии
    if (el.scroll.scrollTop + el.scroll.clientHeight >= el.scroll.scrollHeight - 4) current = items[items.length - 1];
    items.forEach((b) => (b === current ? b.setAttribute("aria-current", "true") : b.removeAttribute("aria-current")));
  }

  // ---------- Левая колонка на узком экране: выезжает по кнопке в шапке ----------

  function toggleDrawer(open) {
    el.dialogs.classList.toggle("is-open", open);
    el.backdrop.hidden = !open;
    el.dialogsBtn.setAttribute("aria-expanded", String(open));
    el.main.inert = open; // пока список выехал, Tab не уходит в разговор под затемнением
    if (open) (el.dialogs.querySelector('[aria-current="true"]') || el.newBtn).focus();
  }

  function closeDrawer() {
    if (!el.dialogs.classList.contains("is-open")) return false;
    toggleDrawer(false);
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
  el.dialogsBtn.addEventListener("click", () => toggleDrawer(!el.dialogs.classList.contains("is-open")));
  el.backdrop.addEventListener("click", closeDrawer);
  // Экран стал шире (повернули планшет) — левая колонка снова на месте, выезжающий режим выключаем
  matchMedia("(min-width: 900px)").addEventListener("change", (e) => { if (e.matches) closeDrawer(); });
  // «Повторить» под списком, если он не загрузился
  el.dialogsHint.addEventListener("click", async (e) => {
    if (!e.target.closest('[data-act="reload"]')) return;
    await loadDialogs();
    // Кнопка «Повторить» исчезла вместе с подсказкой — фокус на первый диалог, а не в никуда.
    // Только если клиент за это время не ушёл в другое место (например, в поле ввода)
    if (document.activeElement === document.body) (el.dialogsList.querySelector(".dialog") || el.newBtn).focus();
  });
  el.dialogsList.addEventListener("scroll", updateListEdge, { passive: true });
  // Высота списка меняется и без перерисовки: открыли чат, появилось пояснение, повернули экран
  new ResizeObserver(updateListEdge).observe(el.dialogsList);

  // Оглавление: прокрутка к вопросу и подсветка того, до которого дочитали
  el.outlineList.addEventListener("click", (e) => {
    const b = e.target.closest(".outline__item");
    const msg = b && document.getElementById(b.dataset.target);
    if (!msg) return;
    const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    msg.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
    // Нажатый пункт подсвечиваем сразу и держим, пока идёт прокрутка
    // (иначе у конца ленты подсветка прыгнула бы на последний вопрос)
    el.outlineList.querySelectorAll(".outline__item").forEach((x) =>
      (x === b ? x.setAttribute("aria-current", "true") : x.removeAttribute("aria-current")));
    // Удержание снимается, когда прокрутка закончилась (scrollend); таймер — на случай, если прокрутки не было
    outlinePinned = b;
    clearTimeout(pinTimer);
    pinTimer = setTimeout(() => { outlinePinned = null; }, 1500);
    // Короткая подсветка самого вопроса — видно, куда привело нажатие, даже если лента не сдвинулась
    msg.classList.remove("msg--flash");
    void msg.offsetWidth; // перезапуск анимации при повторном нажатии
    msg.classList.add("msg--flash");
    msg.addEventListener("animationend", () => msg.classList.remove("msg--flash"), { once: true });
  });
  el.scroll.addEventListener("scroll", () => requestAnimationFrame(markOutline), { passive: true });
  el.scroll.addEventListener("scrollend", () => { outlinePinned = null; clearTimeout(pinTimer); });

  // Esc: сначала закрывает выехавший список диалогов, потом сам чат
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || el.root.hidden) return;
    if (closeDrawer()) { el.dialogsBtn.focus(); return; }
    close();
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
