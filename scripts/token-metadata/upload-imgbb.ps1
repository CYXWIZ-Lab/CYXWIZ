# Upload cyxtoken.png to imgbb
$imagePath = "D:/Dev/CyxWiz_Claude/cyxtoken.png"
$imageBytes = [System.IO.File]::ReadAllBytes($imagePath)
$base64 = [Convert]::ToBase64String($imageBytes)

$body = @{
    image = $base64
    name = "cyxtoken"
}

$response = Invoke-RestMethod -Uri 'https://api.imgbb.com/1/upload?key=c7fa7c5c53f98b73ab43c41a6c6a0903' -Method Post -Body $body
Write-Output "Image uploaded successfully!"
Write-Output "URL: $($response.data.url)"
Write-Output "Display URL: $($response.data.display_url)"
