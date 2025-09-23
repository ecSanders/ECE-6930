# Ensure uv is installed
pip install --upgrade uv

# Sync environment with lock file
uv sync --locked

try {
    # Run the driver Python file (path is relative to Program01)
    uv run tictactoe.py
}
finally {
    # Teardown: remove the uv-managed virtual environment
    if (Test-Path ".venv") {
        Remove-Item -Recurse -Force ".venv"
        Write-Host "Teardown complete: removed .venv"
    }
}
