#!/bin/bash

rm -rf logs && mkdir logs
rm -rf pids && mkdir pids

celery -A worker multi start worker@%h --pool=solo --concurrency=1 --pidfile=pids/%n.pid --logfile=logs/%n.log
