#!/bin/bash

rabbitmqctl add_vhost PlantarPressure
rabbitmqctl add_user PlantarPressure pcadmin-01
rabbitmqctl set_permissions -p PlantarPressure PlantarPressure ".*" ".*" ".*"

