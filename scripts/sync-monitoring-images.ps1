# Copy monitoring plots from artifacts to docs/images for README (GitHub display).
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$src = Join-Path $root "artifacts\plots"
$dst = Join-Path $root "docs\images"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$files = @(
    "monitoring_dashboard.png",
    "drift_psi.png"
)
foreach ($file in $files) {
    $from = Join-Path $src $file
    if (Test-Path $from) {
        Copy-Item $from (Join-Path $dst $file) -Force
        Write-Host "Copied $file -> docs/images/"
    }
}
