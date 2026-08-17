#!/usr/bin/env bash
# Автообновление session_log.md: добавляет снапшот состояния проекта.
# Запускается cron'ом каждые 10 минут (см. crontab -l).
set -euo pipefail

PROJECT="/home/alex/AntiUAV-Detector"
LOG="$PROJECT/docs/session_log.md"
LOCK="/tmp/koda_session_log.lock"
STAMP="$(date '+%Y-%m-%d %H:%M')"

# Не накапливать блоки, если прошлый запуск ещё идёт
exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

# Последний снапшот в логе (метка времени)
LAST="$(grep -oP '^<!-- snap: \K[0-9-]+ [0-9:]+' "$LOG" | tail -1 || true)"

# Пропуск, если снапшот уже записан в этой минуте
if [ -n "$LAST" ] && [ "$LAST" = "$STAMP" ]; then
  exit 0
fi

# --- Сбор состояния ---
GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo 'n/a')"
TRAIN_PID="$(pgrep -f 'ultralytics|yolo.*train' | head -1 || true)"
PY_PROC="$(pgrep -f 'train/|epoch20_eval|batch_test|simulate_intercept|live_detect' | tr '\n' ' ' || true)"
DISK="$(df -h "$PROJECT" | awk 'NR==2 {print $4" свободно из "$2" ("$5" занято)"}')"

# Файлы, изменённые за последние 60 минут
CHANGED="$(find "$PROJECT" -maxdepth 3 \( -path '*/venv' -o -path '*/.git' -o -path '*/runs' -o -path '*/__pycache__' \) -prune -o -type f -mmin -60 -printf '%TH:%TM %p\n' 2>/dev/null | sort | tail -15 || true)"

# --- Запись снапшота ---
cat >> "$LOG" <<EOF

<!-- snap: $STAMP -->
### Снапшот $STAMP (авто)
- **GPU**: $GPU
- **Обучение**: $( [ -n "$TRAIN_PID" ] && echo "идёт (PID $TRAIN_PID)" || echo "не запущено" )
- **Процессы**: ${PY_PROC:-нет}
- **Диск**: $DISK
- **Изменено за 60 мин**:
$CHANGED
EOF

# Ограничение размера: оставить последние 3000 строк
LINES="$(wc -l < "$LOG")"
if [ "$LINES" -gt 3000 ]; then
  tail -n 3000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
