/* sketch.js — панель «Набросок» для показа состояний интерфейса на заглушке.
   Видна только пока работает заглушка (mock.js). С настоящим бэкендом не появляется.
   Файл можно удалить вместе с mock.js, когда набросок утвердят и бэкенд будет готов. */

(() => {
  if (!Api.isMock) return;

  const OPTIONS = [
    ["auto", "Авто: уточнение → ответ → заявка"],
    ["clarify", "Уточняющий вопрос"],
    ["answer", "Обычный ответ"],
    ["offer", "Ответ + предложить заявку"],
    ["offtopic", "Вопрос не по теме"],
    ["slow", "Долгий ответ (14 с)"],
    ["fallback", "Запасной ответ модели"],
    ["error", "Ошибка сервера 502"],
    ["lost", "Сессия потеряна (404)"],
  ];

  const panel = document.createElement("aside");
  panel.className = "sketch";
  panel.setAttribute("aria-label", "Панель наброска");
  panel.innerHTML = `
    <details>
      <summary>Набросок · ответы-заглушки</summary>
      <label>Следующий ответ бота
        <select>${OPTIONS.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>
      </label>
      <label class="sketch__check">
        <input type="checkbox" data-auth> Клиент вошёл на сайт (+7 900 000-00-01)
      </label>
      <p>Ответы выдуманы для проверки интерфейса, это не факты из базы знаний.</p>
      <div class="sketch__links">
        <a href="manager.html" target="_blank" rel="noopener">Панель менеджера</a>
        <a href="?debug=1#chat">Режим проверки</a>
        <button type="button">Сбросить демо-данные</button>
      </div>
    </details>`;

  const select = panel.querySelector("select");
  select.addEventListener("change", () => { MockServer.scenario.next = select.value; });
  // После ответа сценарий сам возвращается в «Авто» — синхронизируем выпадающий список
  setInterval(() => { if (select.value !== MockServer.scenario.next) select.value = MockServer.scenario.next; }, 500);

  // Вход на сайт: перезагружаем страницу, чтобы чат заново спросил /api/me
  const auth = panel.querySelector("[data-auth]");
  auth.checked = localStorage.getItem(MockServer.AUTH_KEY) === "1";
  auth.addEventListener("change", () => {
    localStorage.setItem(MockServer.AUTH_KEY, auth.checked ? "1" : "0");
    location.reload();
  });

  panel.querySelector(".sketch__links button").addEventListener("click", () => {
    MockServer.reset();
    sessionStorage.clear();
    location.reload();
  });

  document.body.append(panel);
})();
