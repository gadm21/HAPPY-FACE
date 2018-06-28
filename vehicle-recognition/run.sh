echo 'Adding heartbeat_cron.txt with crontab'
crontab heartbeat_cron.txt


/usr/bin/python3 worker.py &
/usr/bin/python3 main.py &
