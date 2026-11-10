/* api.js — все запросы фронта к бэкенду в одном месте. Пути и поля — CONTRACTS.md, раздел 4.

   Пока бэкенд не готов, запросы уходят в заглушку (mock.js).
   Включить настоящий бэкенд: поставить USE_MOCK = false или открыть страницу с ?mock=0. */

// Ошибка API: код ответа и текст для человека из поля "detail"
class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

const Api = (() => {
  const params = new URLSearchParams(location.search);
  const USE_MOCK = params.get("mock") !== "0" && typeof MockServer !== "undefined";

  async function request(method, path, body) {
    if (USE_MOCK) return MockServer.handle(method, path, body);

    let res;
    try {
      res = await fetch(path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch {
      throw new ApiError(0, "Нет связи с сервером. Проверьте интернет и попробуйте ещё раз.");
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
    isMock: USE_MOCK,
    newSession: () => request("POST", "/api/sessions"),
    me: () => request("GET", "/api/me"), // вошёл ли клиент на сайт и его номер
    // clientMessageId — метка реплики: при «Повторить» уходит та же, чтобы сервер не записал вопрос дважды
    chat: (sessionId, message, clientMessageId) =>
      request("POST", "/api/chat", { session_id: sessionId, message, client_message_id: clientMessageId }),
    listSessions: () => request("GET", "/api/sessions"),
    getSession: (id) => request("GET", `/api/sessions/${encodeURIComponent(id)}`),
    createLead: (lead) => request("POST", "/api/leads", lead),
    listLeads: () => request("GET", "/api/leads"),
    setLeadStatus: (id, status) => request("PATCH", `/api/leads/${id}`, { status }),
  };
})();
