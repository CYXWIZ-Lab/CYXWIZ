Get-Process | Where-Object { $_.Name -like "*cyxwiz*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Output "Done"
