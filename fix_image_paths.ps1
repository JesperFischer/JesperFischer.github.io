# PowerShell script to fix image paths in markdown files
# This changes relative paths (../images/) to absolute paths (/images/)

$postsPath = "c:\Users\au645332\Documents\JesperFischer.github.io\_posts"
$markdownFiles = Get-ChildItem -Path $postsPath -Filter "*.md"

foreach ($file in $markdownFiles) {
    Write-Host "Processing: $($file.Name)"
    
    # Read the file content
    $content = Get-Content -Path $file.FullName -Raw
    
    # Replace relative image paths with absolute paths
    # Pattern: ../images/ becomes /images/
    $updatedContent = $content -replace '\.\./images/', '/images/'
    
    # Write back to file if changes were made
    if ($content -ne $updatedContent) {
        Set-Content -Path $file.FullName -Value $updatedContent -NoNewline
        Write-Host "  ✓ Fixed image paths in $($file.Name)" -ForegroundColor Green
    } else {
        Write-Host "  - No changes needed for $($file.Name)" -ForegroundColor Gray
    }
}

Write-Host "`nDone! All image paths have been updated." -ForegroundColor Cyan
