/* Общий вход и смена аккаунта. Роль и профиль приходят от API. */
const Auth = (() => {
  const syncKey = "et-account-change";
  let me = null, accounts = [], busy = false;
  const dialog = document.createElement("dialog");
  dialog.className = "auth-dialog";
  dialog.setAttribute("aria-labelledby", "auth-title");
  document.body.append(dialog);

  function show() {
    dialog.innerHTML = `<div class="auth-heading"><h2 id="auth-title">${me ? "Сменить аккаунт" : "Вход в EnergyTool"}</h2>
      ${me ? '<button type="button" class="btn btn-ghost btn-icon" data-close aria-label="Закрыть">'+UI.icon("close")+'</button>' : ''}</div>
      <p class="auth-note">Тестовый вход · выберите аккаунт</p>
      ${["user", "admin"].map(role => `<section><h3>${role === "user" ? "Пользователи" : "Администраторы"}</h3>
      ${accounts.filter(a => a.role === role).map(a => `<button class="auth-account" type="button" data-account="${UI.esc(a.id)}" ${me?.id === a.id ? 'aria-current="true"' : ''}>
      <span>${UI.icon("user")}<span><strong>${UI.esc(a.full_name)}</strong><small>${UI.esc(role === "user" ? a.company : "Панель менеджера")}</small></span></span>
      <span>${me?.id === a.id ? "Вы" : UI.icon("chevron")}</span></button>`).join("")}</section>`).join("")}
      <p class="auth-error" role="alert" hidden></p>${me ? '<button type="button" class="btn btn-ghost" data-logout>Выйти</button>' : ''}`;
    if (!dialog.open) dialog.showModal();
  }

  function route(user) {
    const manager = location.pathname.endsWith("/manager.html") || location.pathname.endsWith("/manager");
    if ((user.role === "admin") !== manager) {
      location.replace(user.role === "admin" ? "manager.html" : "index.html");
      return false;
    }
    return true;
  }

  async function init() {
    try {
      accounts = await Api.accounts();
      if (!accounts.length) return null; // старый режим без демонстрационного входа
      const profile = await Api.me();
      me = profile.authenticated ? profile : null;
      // Переписка вкладки не переносится в другой аккаунт, в том числе после входа из другой вкладки.
      if (sessionStorage.getItem("et-account-id") !== me?.id) sessionStorage.removeItem("et-session-id");
      if (me) sessionStorage.setItem("et-account-id", me.id);
      else sessionStorage.removeItem("et-account-id");
      if (!me) { show(); return null; }
      if (!route(me)) return null;
      document.querySelectorAll(".site-header__inner, .chat__tools, .mgr-header").forEach(container => {
        const b = document.createElement("button");
        b.className = "btn btn-ghost account-button";
        b.type = "button";
        b.innerHTML = `${UI.icon("user")}<span>${UI.esc(me.full_name.split(" ").slice(0, 2).join(" "))}<small>${me.role === "admin" ? "Администратор" : "Пользователь"}</small></span>`;
        b.setAttribute("aria-label", `${me.full_name}. Сменить аккаунт`);
        b.addEventListener("click", show);
        container.append(b);
      });
      return me;
    } catch (error) {
      console.warn("Ошибка загрузки входа", error);
      dialog.innerHTML = '<h2 id="auth-title">Не удалось загрузить вход</h2><p class="auth-note">Проверьте соединение и попробуйте ещё раз.</p><button class="btn btn-primary" type="button" data-reload>Повторить</button>';
      if (!dialog.open) dialog.showModal();
      return null;
    }
  }

  dialog.addEventListener("cancel", e => { if (!me || busy) e.preventDefault(); });
  dialog.addEventListener("click", async e => {
    if (busy) return;
    if (e.target.closest("[data-reload]")) { location.reload(); return; }
    if (e.target.closest("[data-close]")) { dialog.close(); return; }
    const button = e.target.closest("[data-account], [data-logout]");
    if (!button) return;
    if (button.dataset.account === me?.id) { dialog.close(); return; }
    busy = true;
    dialog.querySelectorAll("button").forEach(b => b.disabled = true);
    try {
      if (button.hasAttribute("data-logout")) await Api.logout();
      else await Api.login(button.dataset.account);
      sessionStorage.removeItem("et-session-id");
      localStorage.setItem(syncKey, `${Date.now()}-${Math.random()}`);
      location.replace("index.html");
    } catch (err) {
      const error = dialog.querySelector(".auth-error");
      error.textContent = err.message;
      error.hidden = false;
      dialog.querySelectorAll("button").forEach(b => b.disabled = false);
      busy = false;
    }
  });
  window.addEventListener("storage", e => { if (e.key === syncKey) location.reload(); });
  const ready = init();
  return { ready, show, get user() { return me; }, get enabled() { return accounts.length > 0; } };
})();
