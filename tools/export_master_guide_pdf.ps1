param(
  [string]$HtmlPath = "${PSScriptRoot}\..\M3_Evaluation_Master_Guide.html",
  [string]$PdfPath  = "${PSScriptRoot}\..\M3_Evaluation_Master_Guide.pdf"
)

$HtmlFull = (Resolve-Path $HtmlPath).Path
$PdfFull  = (Resolve-Path (Split-Path $PdfPath -Parent)).Path + "\" + (Split-Path $PdfPath -Leaf)

function Find-Browser {
  $edgeCmd = Get-Command msedge -ErrorAction SilentlyContinue
  $chromeCmd = Get-Command chrome -ErrorAction SilentlyContinue
  $operaCmd = Get-Command opera -ErrorAction SilentlyContinue

  $candidates = @(
    $(if ($edgeCmd) { $edgeCmd.Source } else { $null }),
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles(x86)\Microsoft\Edge\Application\msedge.exe",
    $(if ($chromeCmd) { $chromeCmd.Source } else { $null }),
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "$env:ProgramFiles(x86)\Google\Chrome\Application\chrome.exe",
    $(if ($operaCmd) { $operaCmd.Source } else { $null }),
    "$env:LOCALAPPDATA\Programs\Opera GX\launcher.exe",
    "$env:LOCALAPPDATA\Programs\Opera GX\opera.exe",
    "$env:ProgramFiles\Opera GX\launcher.exe",
    "$env:ProgramFiles\Opera GX\opera.exe",
    "$env:ProgramFiles(x86)\Opera GX\launcher.exe",
    "$env:ProgramFiles(x86)\Opera GX\opera.exe"
  )

  $candidates = @(
    $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
  )

  if ($candidates.Count -gt 0) { return $candidates[0] }
  return $null
}

$browser = Find-Browser
if (-not $browser) {
  Write-Host "No Edge/Chrome found for headless PDF export." -ForegroundColor Yellow
  Write-Host "Fallback: open $HtmlFull in your browser, press Ctrl+P, Save as PDF." -ForegroundColor Yellow
  exit 1
}

# Convert Windows path to file:// URL
$fileUrl = "file:///" + ($HtmlFull -replace "\\", "/")

Write-Host "Exporting PDF..." -ForegroundColor Cyan
Write-Host "Browser: $browser" -ForegroundColor Gray
Write-Host "HTML:    $HtmlFull" -ForegroundColor Gray
Write-Host "PDF:     $PdfFull" -ForegroundColor Gray

& $browser --headless --disable-gpu --no-first-run --no-default-browser-check --print-to-pdf="$PdfFull" "$fileUrl"

if (Test-Path $PdfFull) {
  Write-Host "Done. Created: $PdfFull" -ForegroundColor Green
  exit 0
}

Write-Host "Export failed. Fallback: open HTML and print to PDF." -ForegroundColor Yellow
exit 2
