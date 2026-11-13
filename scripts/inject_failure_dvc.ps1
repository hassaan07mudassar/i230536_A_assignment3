# PowerShell Failure Injection: DVC Data Tampering
Write-Host "==> [FAILURE INJECTION] Corrupting data/raw/customer_churn.csv..." -ForegroundColor Yellow
Add-Content -Path "data/raw/customer_churn.csv" -Value "CORRUPT_99999,-999,9,9999.99,-500.00,0,0,0,1"
Write-Host "==> Data corrupted. Verify with dvc status." -ForegroundColor Red

