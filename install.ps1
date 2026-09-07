# Yapuny worker installer (Windows).
$ErrorActionPreference = "Stop"
$repo = "ketjandr/yapuny"
$tarball = "https://github.com/$repo/archive/refs/heads/main.tar.gz"
$state = Join-Path $env:USERPROFILE ".yapuny"
$shaFile = Join-Path $state "installed_sha"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv..."
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
# uv installs to %USERPROFILE%\.local\bin; make sure this shell sees it and the `yapuny` command
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# latest commit on main (empty if offline); the sha we installed last time
$latest = ""
try { $latest = (Invoke-RestMethod "https://api.github.com/repos/$repo/commits/main").sha } catch {}
$current = ""
if (Test-Path $shaFile) { $current = (Get-Content $shaFile -Raw).Trim() }
$short = if ($latest) { $latest.Substring(0, 7) } else { "" }

# already current: skip the reinstall and just start (fast path)
if ($latest -and $latest -eq $current -and (Get-Command yapuny -ErrorAction SilentlyContinue)) {
  Write-Host "Worker already up to date ($short). Starting..."
  yapuny
  exit
}

if ($current -and $latest) { Write-Host "New worker version available ($short) - updating..." }
else { Write-Host "Installing the Yapuny worker..." }

$env:UV_TORCH_BACKEND = "auto"
# --force overwrites the existing tool; --refresh re-fetches the (mutable) main tarball
uv tool install --force --refresh --python 3.11 "yapuny @ $tarball"

# record what we just installed so the next run can tell update vs up-to-date
New-Item -ItemType Directory -Force -Path $state | Out-Null
if ($latest) { Set-Content -Path $shaFile -Value $latest -NoNewline }

Write-Host ""
Write-Host "Done. Starting the worker on http://localhost:8000"
Write-Host "Next time, just run:  yapuny"
Write-Host ""
yapuny
