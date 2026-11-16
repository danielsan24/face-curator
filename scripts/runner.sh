#!/bin/zsh
# launchd entry point: one try per week (Sunday 06:00). The run itself checks the month, AC power and the LAN,
# and exits quietly when it is not the day. Logs: ~/Library/Logs/face-curator/YYYY-MM.log
set -u
umask 077   # logs name people and photos: readable by this user only
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
HERE="${0:A:h}"
LOGDIR="$HOME/Library/Logs/face-curator"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(date +%Y-%m).log"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') scheduled try" >> "$LOG"
exec /usr/bin/caffeinate -i uv run --quiet --locked "$HERE/facecurator.py" run --scheduled >> "$LOG" 2>&1
