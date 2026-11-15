# EnergyTool — A/B с thinking на 3b607ea

Локальный прогон 15 ноября 2026 года. Код бота соответствует `3b607ea`. В обеих группах ответы генерировал Qwen3-4B Q4_K_M, фрагменты подбирались один раз с BGE-M3 без реранкера. Поисковые запросы для диалогов формировались с выключенным thinking. Порядок групп чередовался, каждому вопросу соответствовали один ответ с thinking и один без него.

| Метрика | Thinking выключен | Thinking включён |
|---|---:|---:|
| Демо-вопросы | 18 | 18 |
| Пустые ответы | 0 | 0 |
| Срабатывания общего лимита | 0 | 0 |
| Время генерации, с | 28,546 | 61,950 |
| Токены рассуждений | 0 | 2285 |
| Match Rate | 0,653 | 0,778 |
| Win Rate | 0,389 | 0,611 |
| Локальный Score | 0,574 | 0,728 |

Судья Huihui-Qwen3-VL-8B-Instruct оценивал каждую пару дважды с перестановкой ответов. Локальный Score равен `0,7 × Match Rate + 0,3 × Win Rate`, ничья даёт половину победы.

## Повтор прогона

Используйте рабочий каталог этого коммита с установленными зависимостями `bot/requirements.txt`. Передайте каталог с локальными `demo_questions.csv`, `demo_output.csv` и `knowledge-base.zip` через `--data`. Для генерации запустите llama.cpp с моделью Qwen3-4B Q4_K_M.

```powershell
llama-server -m <Qwen3-4B-Q4_K_M.gguf> --alias mts-qwen3-4b --host 127.0.0.1 --port 12347 -c 12288 -np 1 -ngl 99 --no-ui --reasoning auto --reasoning-format deepseek --reasoning-budget 128
```

Эмбеддер `text-embedding-bge-m3` и судья `huihui-qwen3-vl-8b-instruct-abliterated` по умолчанию запрашиваются у LM Studio на порту `1234`. Для отдельно поднятых OpenAI-совместимых серверов можно передать `--embeddings`, `--judge-url` и `--judge-model`.

```powershell
python eval/thinking_ab.py generate --data <папка_с_демо_данными>
python eval/thinking_ab.py judge
```

Скрипт передаёт `chat_template_kwargs.enable_thinking` отдельно для каждой группы, использует общий `max_tokens=3072`, температуру `0.6`, `top_p=0.95`, `top_k=20`, `min_p=0` и `repeat_penalty=1.0`. Проба проверяет, что сервер применил переключатель и бюджет. Сырые результаты записываются локально в `runs/thinking-ab-3b607ea`.

Пересчитать сводку по сохранённому прогону можно через `python eval/thinking_ab.py report --output <каталог_прогона>`.
