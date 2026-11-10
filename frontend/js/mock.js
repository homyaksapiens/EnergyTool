/* mock.js — заглушка бэкенда для наброска интерфейса.

   Повторяет HTTP API из CONTRACTS.md (раздел 4): те же пути, поля ответов и коды ошибок.
   Данные хранятся в localStorage браузера, поэтому заявка, отправленная из чата,
   появляется в панели менеджера (manager.html), если открыть её в том же браузере.

   ВАЖНО: ответы «бота» здесь — выдуманные примеры для проверки интерфейса.
   Это не факты из базы знаний, цен и сроков в них нет и быть не должно.

   Когда бэкенд будет готов, этот файл не понадобится: api.js переключится на настоящие запросы. */

const MockServer = (() => {
  const DB_KEY = "et-mock-db-v2"; // v2: сессии помнят номер вошедшего клиента (client_phone)
  const AUTH_KEY = "et-mock-auth"; // "1" — клиент как будто вошёл на сайт по номеру
  const PHONE = "+7 900 000-00-01"; // номер «вошедшего» клиента в наброске

  const isAuth = () => localStorage.getItem(AUTH_KEY) === "1";

  // Какой ответ вернёт следующий запрос к чату. Меняется панелью «Набросок» (sketch.js).
  // "auto" — выбрать по ходу диалога, остальные значения — принудительно один раз.
  const scenario = { next: "auto" };

  // ---------- Хранилище ----------

  function load() {
    const raw = localStorage.getItem(DB_KEY);
    if (raw) return JSON.parse(raw);
    const db = seed();
    save(db);
    return db;
  }

  function save(db) {
    localStorage.setItem(DB_KEY, JSON.stringify(db));
  }

  function reset() {
    localStorage.removeItem(DB_KEY);
  }

  function newId() {
    // 32 случайных шестнадцатеричных символа, как uuid без дефисов
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  function now(offsetMinutes = 0) {
    return new Date(Date.now() + offsetMinutes * 60000).toISOString();
  }

  // ---------- Примеры ответов (только для наброска) ----------

  const SRC = {
    askue: { source: "20-products/22-askue.md", title: "Облачная платформа учёта электроэнергии → Кому подходит" },
    portfolio: { source: "20-products/20-portfolio.md", title: "Обзор портфеля → Направления" },
    bill: { source: "10-energy/14-bill-growth.md", title: "Почему вырос счёт → С чего начать" },
    scope: { source: "00-reference/02-product-scope-and-limits.md", title: "Границы продуктов → Когда нужен специалист" },
    lead: { source: "00-reference/03-what-we-need.md", title: "Что нужно для расчёта и заявки" },
  };

  const REPLIES = {
    clarify: {
      answer: "Да, с этим можем помочь. Чтобы подобрать решение, уточните: о каком объекте речь — офис, магазин, производство — и сколько у вас точек учёта?",
      sources: [SRC.portfolio, SRC.bill],
    },
    answer: {
      answer: "Да, облачная платформа учёта собирает показания всех объектов в одном кабинете: видно расход по каждой точке и общую картину. Подойдут ли ваши счётчики, специалист проверит по их моделям. Какие счётчики у вас установлены сейчас?",
      sources: [SRC.askue, SRC.scope],
    },
    offer: {
      answer: "Для вашей задачи подходит облачная платформа учёта, а по росту счёта стоит проверить ценовую категорию. Точный расчёт по вашим счетам подготовит менеджер. Передать ему заявку с кратким описанием нашего разговора?",
      sources: [SRC.askue, SRC.bill, SRC.lead],
      offer_lead: true,
    },
    offtopic: {
      answer: "Я консультирую только по решениям EnergyTool для энергетики бизнеса. Могу помочь разобраться со счетами за электроэнергию, учётом или потерями в сети — расскажите о своей задаче.",
      sources: [],
    },
    fallback: {
      answer: "Не получилось подготовить ответ. Попробуйте переформулировать вопрос или оставьте заявку — специалист свяжется с вами и всё уточнит.",
      sources: [],
      fallback: true,
    },
  };

  // Выбор ответа в режиме "auto": первый вопрос — уточнение, третий — предложение заявки
  function pickAuto(session, text) {
    if (/погод|анекдот|рецепт|футбол|стих/i.test(text)) return "offtopic";
    const userTurns = session.messages.filter((m) => m.role === "user").length;
    if (userTurns <= 1) return "clarify";
    if (userTurns >= 3 && !session.lead_id) return "offer";
    return "answer";
  }

  // ---------- Обработчики путей ----------

  function createSession(db) {
    // Вошедший клиент: сессия запоминает его номер, чтобы попасть в «Ваши диалоги» (GET /api/me/sessions)
    const s = { session_id: newId(), created_at: now(), updated_at: now(), lead_id: null, client_phone: isAuth() ? PHONE : null, messages: [] };
    db.sessions.push(s);
    return s;
  }

  async function chat(db, body) {
    let session;
    if (body.session_id) {
      session = db.sessions.find((s) => s.session_id === body.session_id);
      if (!session) throw new ApiError(404, "Сессия не найдена");
    } else {
      session = createSession(db);
    }
    const text = String(body.message || "").trim();
    if (!text) throw new ApiError(422, "Пустое сообщение");

    let kind = scenario.next === "auto" ? null : scenario.next;
    scenario.next = "auto";
    if (kind === "lost") throw new ApiError(404, "Сессия не найдена");

    // Повтор после ошибки приходит с той же меткой — второй раз вопрос не записываем
    const last = session.messages[session.messages.length - 1];
    const isRepeat = body.client_message_id && last && last.client_message_id === body.client_message_id;
    if (!isRepeat) {
      session.messages.push({ role: "user", text, created_at: now(), client_message_id: body.client_message_id });
      save(db);
    }

    // Имитация времени ответа: обычно 2–4 с, «долгий ответ» — 14 с
    const delay = kind === "slow" ? 14000 : 1800 + Math.random() * 1800;
    await sleep(delay);

    if (kind === "error") throw new ApiError(502, "Модель сейчас не отвечает. Попробуйте ещё раз через минуту.");
    if (!kind || kind === "slow") kind = pickAuto(session, text);

    const reply = REPLIES[kind];
    const debug = {
      elapsed_ms: Math.round(delay),
      timings: { retrieval: 0.05, llm: +(delay / 1000 - 0.05).toFixed(2) },
      fallback: Boolean(reply.fallback),
      sources: reply.sources,
    };
    // Перечитать хранилище: пока «думали», данные могли поменяться в другой вкладке
    const fresh = load();
    const s = fresh.sessions.find((x) => x.session_id === session.session_id);
    if (!s) throw new ApiError(404, "Сессия не найдена"); // демо-данные сбросили во время ответа
    s.messages.push({ role: "assistant", text: reply.answer, created_at: now(), debug });
    s.updated_at = now();
    save(fresh);

    return { session_id: s.session_id, answer: reply.answer, offer_lead: Boolean(reply.offer_lead), debug };
  }

  function createLead(db, body) {
    const session = db.sessions.find((s) => s.session_id === body.session_id);
    if (!session) throw new ApiError(404, "Сессия не найдена");
    if (!body.consent) throw new ApiError(422, "Нужно согласие на обработку персональных данных");
    const digits = String(body.phone || "").replace(/\D/g, "");
    if (digits.length < 10 || digits.length > 11) throw new ApiError(422, "Проверьте номер телефона: нужно 10–11 цифр");
    for (const [field, label] of [["full_name", "ФИО"], ["company", "название компании"]]) {
      if (!String(body[field] || "").trim()) throw new ApiError(422, `Укажите ${label}`);
    }
    const lead = {
      lead_id: db.leads.reduce((m, l) => Math.max(m, l.lead_id), 0) + 1,
      created_at: now(),
      session_id: session.session_id,
      full_name: body.full_name.trim(),
      phone: body.phone.trim(),
      company: body.company.trim(),
      summary: makeSummary(session, body),
      status: "new",
    };
    db.leads.push(lead);
    session.lead_id = lead.lead_id;
    session.updated_at = now();
    save(db);
    return { lead_id: lead.lead_id, summary: lead.summary };
  }

  // Резюме в формате раздела 4.3. Настоящее составит бот; здесь — шаблон из реплик клиента
  function makeSummary(session, c) {
    const asks = session.messages.filter((m) => m.role === "user").map((m) => m.text);
    return [
      `**Клиент:** ${c.full_name}, ${c.company}, ${c.phone}`,
      `**Задача:** ${asks[0] || "клиент сразу оставил контакты, задачу не описал"}`,
      `**Что выяснили:** ${asks.slice(1).join("; ") || "—"}`,
      "**Подходящие решения:** (заглушка — подберёт бот)",
      "**Открытые вопросы:** (заглушка)",
      "**Следующий шаг:** позвонить клиенту, уточнить задачу",
    ].join("\n");
  }

  // filter — какие сессии брать: все (панель менеджера) или только вошедшего клиента
  function listSessions(db, filter = () => true) {
    return db.sessions
      .filter((s) => s.messages.length && filter(s)) // пустые «Новый диалог» без реплик не показываем
      .map((s) => ({
        session_id: s.session_id,
        created_at: s.created_at,
        updated_at: s.updated_at,
        messages: s.messages.length,
        lead_id: s.lead_id,
        preview: (s.messages.find((m) => m.role === "user") || {}).text || "",
      }))
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  }

  // ---------- Точка входа: тот же метод и путь, что у настоящего API ----------

  async function handle(method, path, body) {
    await sleep(150); // небольшая задержка, как у сети
    const db = load();
    let m;

    if (method === "POST" && path === "/api/sessions") {
      const s = createSession(db);
      save(db);
      return { session_id: s.session_id };
    }
    if (method === "GET" && path === "/api/sessions") return listSessions(db);
    if (method === "GET" && path === "/api/me/sessions") {
      return isAuth() ? listSessions(db, (s) => s.client_phone === PHONE) : [];
    }
    if (method === "GET" && (m = path.match(/^\/api\/sessions\/(\w+)$/))) {
      const s = db.sessions.find((x) => x.session_id === m[1]);
      if (!s) throw new ApiError(404, "Сессия не найдена");
      const { client_phone, ...pub } = s; // номер клиента наружу не отдаём
      return pub;
    }
    if (method === "POST" && path === "/api/chat") return chat(db, body);
    if (method === "POST" && path === "/api/leads") return createLead(db, body);
    if (method === "GET" && path === "/api/leads") {
      return [...db.leads].sort((a, b) => b.created_at.localeCompare(a.created_at));
    }
    if (method === "PATCH" && (m = path.match(/^\/api\/leads\/(\d+)$/))) {
      const lead = db.leads.find((l) => l.lead_id === Number(m[1]));
      if (!lead) throw new ApiError(404, "Заявка не найдена");
      lead.status = body.status;
      save(db);
      return { ok: true };
    }
    if (method === "GET" && path === "/api/me") {
      // Вход на сайт изображается галочкой в панели «Набросок»
      return isAuth() ? { authenticated: true, phone: PHONE } : { authenticated: false, phone: null };
    }
    if (method === "GET" && path === "/api/health") return { ok: true, model: "заглушка", kb_chunks: 236, dense: false };
    throw new ApiError(404, `Нет такого пути в заглушке: ${method} ${path}`);
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  // ---------- Стартовые данные: выдуманные диалоги ----------
  // Три диалога «вошедшего» клиента (+7 900 000-00-01), один с заявкой, и один диалог другого клиента:
  // его видит только менеджер, в «Ваших диалогах» его быть не должно

  function seed() {
    const s1 = newId();
    const s2 = newId();
    const s3 = newId();
    const s4 = newId();
    const msg = (role, text, min, debug) => ({ role, text, created_at: now(min), ...(debug ? { debug } : {}) });
    const dbg = (ms, sources) => ({ elapsed_ms: ms, timings: { retrieval: 0.04, llm: ms / 1000 - 0.04 }, fallback: false, sources });
    return {
      sessions: [
        {
          session_id: s1, created_at: now(-180), updated_at: now(-170), lead_id: 1, client_phone: PHONE,
          messages: [
            msg("user", "У нас несколько магазинов. Можно смотреть их расход в одном кабинете?", -180),
            msg("assistant", REPLIES.clarify.answer, -180, dbg(3920, REPLIES.clarify.sources)),
            msg("user", "Пять магазинов, счётчики стоят разные", -176),
            msg("assistant", REPLIES.offer.answer, -176, dbg(4210, REPLIES.offer.sources)),
          ],
        },
        {
          session_id: s2, created_at: now(-60), updated_at: now(-58), lead_id: null, client_phone: PHONE,
          messages: [
            msg("user", "Счёт за свет вырос в полтора раза, почему так?", -60),
            msg("assistant", REPLIES.clarify.answer, -60, dbg(3550, REPLIES.clarify.sources)),
          ],
        },
        {
          session_id: s3, created_at: now(-2900), updated_at: now(-2894), lead_id: null, client_phone: PHONE,
          messages: [
            msg("user", "Можно передавать показания счётчиков по фото?", -2900),
            msg("assistant", REPLIES.clarify.answer, -2900, dbg(3310, REPLIES.clarify.sources)),
            msg("user", "Офис и склад, по два счётчика", -2896),
            msg("assistant", REPLIES.answer.answer, -2894, dbg(4020, REPLIES.answer.sources)),
          ],
        },
        {
          session_id: s4, created_at: now(-30), updated_at: now(-29), lead_id: null, client_phone: null,
          messages: [
            msg("user", "В СНТ большие потери, как найти, где теряется электроэнергия?", -30),
            msg("assistant", REPLIES.clarify.answer, -30, dbg(3700, REPLIES.clarify.sources)),
          ],
        },
      ],
      leads: [
        {
          lead_id: 1, created_at: now(-170), session_id: s1,
          full_name: "Петров Пётр Петрович", phone: "+7 900 000-00-01", company: "ООО «Ромашка»",
          summary: [
            "**Клиент:** Петров Пётр Петрович, ООО «Ромашка», +7 900 000-00-01",
            "**Задача:** видеть расход нескольких магазинов в одном кабинете",
            "**Что выяснили:** пять магазинов, счётчики разных моделей",
            "**Подходящие решения:** облачная платформа учёта — общий кабинет для всех точек",
            "**Открытые вопросы:** модели счётчиков, ценовая категория",
            "**Следующий шаг:** демо платформы",
          ].join("\n"),
          status: "in_progress",
        },
      ],
    };
  }

  return { handle, reset, scenario, AUTH_KEY };
})();
