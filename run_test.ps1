# Image Classification API - Run Tests
# Usage: .\run_test.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Image Classification API - Run Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$dockerPath = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerPath) {
    Write-Host "Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

$composeCommand = "docker-compose"
$composeCheck = & docker-compose --version 2>$null
if (-not $composeCheck) {
    $composeCheck = & docker compose version 2>$null
    if ($composeCheck) {
        $composeCommand = "docker compose"
    } else {
        Write-Host "Docker Compose is not installed or not in PATH" -ForegroundColor Red
        Write-Host "Please install Docker Compose and try again" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Building and starting test container..." -ForegroundColor Green
Write-Host ""

& $composeCommand up test --build

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Tests completed!" -ForegroundColor Green
Write-Host "HTML Report: test_reports\test_report.html" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
