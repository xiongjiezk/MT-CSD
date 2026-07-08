#!/bin/bash

cur_branch=`git branch  |grep "*" |awk '{print $2}'`
echo "current branch:"
echo $cur_branch
git fetch origin && git reset --hard origin/$cur_branch
