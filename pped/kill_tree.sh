#!/bin/bash

# 使用 ps 和 awk 获取所有子孙进程
killtree() {
    local pid=$1
    local pids=$(ps -eo pid,ppid | awk -v pid=$pid '$2==pid || $1==pid {print $1}')
    echo $pids | xargs kill -9 2>/dev/null
}

killtree $1