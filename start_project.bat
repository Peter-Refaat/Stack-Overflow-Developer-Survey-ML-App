@echo off
title CS_42 Starter

echo Starting Backend...
start cmd /k "cd /d D:\CS_42\backend && python main.py"

echo Starting Frontend...
start cmd /k "cd /d D:\CS_42\frontend && npm run dev"

exit