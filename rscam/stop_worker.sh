#!/bin/bash

celery multi kill worker@%h --pidfile=pids/%n.pid --logfile=logs/%n.log

rm -rf pids
rm -rf logs
