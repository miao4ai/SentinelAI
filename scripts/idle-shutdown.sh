#!/bin/bash
# Startup script: auto-shutdown the VM when idle, to avoid forgotten-on billing.
# Installs a cron job that stops the machine after IDLE_MIN minutes with no
# logged-in users and low CPU load.
set -e

IDLE_MIN=120

cat >/usr/local/bin/idle-check.sh <<'EOF'
#!/bin/bash
IDLE_MIN=120
STAMP=/var/run/last-active
now=$(date +%s)
users=$(who | wc -l)
load=$(awk '{print $1}' /proc/loadavg)
busy=$(awk -v l="$load" 'BEGIN{print (l>0.5)?1:0}')
if [ "$users" -gt 0 ] || [ "$busy" -eq 1 ]; then
  echo "$now" > "$STAMP"
  exit 0
fi
last=$(cat "$STAMP" 2>/dev/null || echo "$now")
idle=$(( (now - last) / 60 ))
if [ "$idle" -ge "$IDLE_MIN" ]; then
  logger "idle-shutdown: idle ${idle}min >= ${IDLE_MIN}min, shutting down"
  /sbin/shutdown -h now
fi
EOF

chmod +x /usr/local/bin/idle-check.sh
date +%s > /var/run/last-active
echo "*/5 * * * * root /usr/local/bin/idle-check.sh" > /etc/cron.d/idle-shutdown
