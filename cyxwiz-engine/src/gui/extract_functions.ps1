# PowerShell script to extract function ranges from node_editor.cpp
# This helps us identify exact line numbers for each function

$file = "D:\Dev\CyxWiz_Claude\cyxwiz-engine\src\gui\node_editor.cpp"

# Find all function definitions
Select-String -Path $file -Pattern "^(void|bool|std::string|std::vector|MLNode|ImVec2|unsigned int|const MLNode\*) NodeEditor::" | ForEach-Object {
    "$($_.LineNumber): $($_.Line)"
}
