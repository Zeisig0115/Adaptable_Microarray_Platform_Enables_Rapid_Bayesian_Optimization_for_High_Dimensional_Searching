# Project instructions

Always run Python code in this project with:

C:\anaconda3\envs\botorch\python.exe

Do not use system Python, `python`, `py`, or another conda environment unless explicitly asked.

For scripts:

"C:\anaconda3\envs\botorch\python.exe" script.py

For modules and tests:

"C:\anaconda3\envs\botorch\python.exe" -m pytest

When editing code that uses BoTorch, inspect the installed BoTorch package in this environment instead of guessing API details.
Use commands like:

"C:\anaconda3\envs\botorch\python.exe" -c "import botorch, inspect; print(botorch.__file__)"