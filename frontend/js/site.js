/* site.js — страница-заглушка сайта: кнопки, которые открывают чат. */

UI.hydrateIcons();

// data-open-chat — открыть чат; data-open-chat="lead" — открыть сразу с заявкой
document.querySelectorAll("[data-open-chat]").forEach((btn) => {
  btn.addEventListener("click", () => Chat.open({ intent: btn.dataset.openChat || undefined, opener: btn }));
});

// data-ask="вопрос" — открыть чат и сразу задать этот вопрос
document.querySelectorAll("[data-ask]").forEach((btn) => {
  btn.addEventListener("click", () => Chat.open({ ask: btn.dataset.ask, opener: btn }));
});

// Ссылка вида index.html#chat сразу открывает чат (удобно для показа жюри)
if (location.hash === "#chat") Chat.open();
