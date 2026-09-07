# Yapuny worker installer (Windows)
$ErrorActionPreference = "Stop"
$tarball = "https://github.com/ketjandr/yapuny/archive/refs/heads/main.tar.gz"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv..."
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
# uv installs to %USERPROFILE%\.local\bin; make sure this shell sees it and the `yapuny` command
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:UV_TORCH_BACKEND = "auto"

Write-Host "Installing the Yapuny worker (auto-detecting GPU)..."
uv tool install --force --python 3.11 "yapuny @ $tarball"

Write-Host ""
Write-Host "Done. Starting the worker on http://localhost:8000"
Write-Host "Next time, just run:  yapuny"
Write-Host ""
yapuny
