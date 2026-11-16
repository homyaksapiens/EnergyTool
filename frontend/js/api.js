/* api.js — запросы фронтенда к HTTP API приложения. */

// Ошибка API: код ответа и текст для человека из поля "detail"
class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

const Api = (() => {
  // timeoutMs — сколько ждать ответа. Без предела зависший ответ модели держал бы чат «занятым» до перезагрузки
  async function request(method, path, body, timeoutMs = 30000) {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), timeoutMs);
    let res;
    try {
      res = await fetch(path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
        signal: abort.signal,
      });
    } catch {
      if (abort.signal.aborted) {
        throw new ApiError(0, `Сервер не ответил за ${Math.round(timeoutMs / 1000)} секунд. Попробуйте ещё раз.`);
      }
      throw new ApiError(0, "Нет связи с сервером. Проверьте интернет и попробуйте ещё раз.");
    } finally {
      clearTimeout(timer);
    }
    let data = null;
    try { data = await res.json(); } catch { /* тело не JSON — оставляем null */ }
    if (!res.ok) {
      // FastAPI на ошибке проверки полей присылает detail списком — берём первое сообщение
      let detail = data && data.detail;
      if (Array.isArray(detail)) detail = detail[0] && detail[0].msg;
      throw new ApiError(res.status, detail || `Ошибка сервера (${res.status})`);
    }
    return data;
  }

  return {
    accounts: () => request("GET", "/api/auth/accounts"),
    login: (accountId) => request("POST", "/api/auth/login", { account_id: accountId }),
    logout: () => request("POST", "/api/auth/logout"),
    newSession: () => request("POST", "/api/sessions"),
    me: () => request("GET", "/api/me"), // вошёл ли клиент на сайт и его номер
    // clientMessageId — метка реплики: при «Повторить» уходит та же, чтобы сервер не записал вопрос дважды
    // Ответ модели может идти долго (резюме, поиск) — ждём до 90 с, потом ошибка с кнопкой «Повторить»
    chat: (sessionId, message, clientMessageId) =>
      request("POST", "/api/chat", { session_id: sessionId, message, client_message_id: clientMessageId }, 90000),
    listSessions: () => request("GET", "/api/sessions"),        // все сессии — только для панели менеджера
    mySessions: () => request("GET", "/api/me/sessions"),         // диалоги этого клиента: по номеру или по метке браузера
    getSession: (id) => request("GET", `/api/sessions/${encodeURIComponent(id)}`),
    createLead: (lead) => request("POST", "/api/leads", lead, 90000), // резюме для менеджера пишет бот — тоже может быть долго
    listLeads: () => request("GET", "/api/leads"),
    setLeadStatus: (id, status) => request("PATCH", `/api/leads/${id}`, { status }),
    managerLeads: (params) => request("GET", `/api/manager/leads?${new URLSearchParams(params)}`),
    managerSessions: (params) => request("GET", `/api/manager/sessions?${new URLSearchParams(params)}`),
    managerLead: (id) => request("GET", `/api/manager/leads/${encodeURIComponent(id)}`),
  };
})();
