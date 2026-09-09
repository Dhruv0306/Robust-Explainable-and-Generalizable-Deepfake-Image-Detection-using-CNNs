@echo off
REM Suppress CUDA capability warnings for newer GPUs
set PYTHONWARNINGS=ignore::UserWarning

REM Run the experiment
.venv\Scripts\python src\run_experiment.py %*
