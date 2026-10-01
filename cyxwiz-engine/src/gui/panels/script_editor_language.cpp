// Script Editor language and custom editor palettes.

#include "script_editor.h"

#include "../../core/argb_colour.h"
#include "../python_language.h"

#include <string>

namespace cyxwiz {

namespace {
// The palettes below are written 0xAARRGGBB; TextEditor draws ImU32
// 0xAABBGGRR (TOFIX133 P0 item 5).
TextEditor::Palette FromArgb(TextEditor::Palette palette) {
    for (auto& colour : palette) colour = ArgbToAbgr(colour);
    return palette;
}

}  // namespace

TextEditor::LanguageDefinition ScriptEditorPanel::CreatePythonLanguage() {
    return PythonLanguage();
}

// ==================== Custom Theme Palettes ====================

TextEditor::Palette ScriptEditorPanel::GetMonokaiPalette() {
    // Monokai theme - popular dark theme with vibrant colors
    return FromArgb(TextEditor::Palette{{
        0xfff8f8f2, // Default (light gray)
        0xfff92672, // Keyword (pink)
        0xffae81ff, // Number (purple)
        0xffe6db74, // String (yellow)
        0xffe6db74, // Char literal (yellow)
        0xfff8f8f2, // Punctuation (light gray)
        0xffa6e22e, // Preprocessor (green)
        0xfff8f8f2, // Identifier (light gray)
        0xff66d9ef, // Known identifier (cyan)
        0xffa6e22e, // Preproc identifier (green)
        0xff75715e, // Comment (gray)
        0xff75715e, // Multi-line comment (gray)
        0xff272822, // Background (dark gray-green)
        0xffe0e0e0, // Cursor (white)
        0x80494440, // Selection (translucent)
        0xa0ff5555, // Error marker (red)
        0x80f92672, // Breakpoint (pink)
        0xff90908a, // Line number (gray)
        0x40808080, // Current line fill
        0x30808080, // Current line fill inactive
        0x40808080  // Current line edge
    }});
}

TextEditor::Palette ScriptEditorPanel::GetDraculaPalette() {
    // Dracula theme - dark theme with purple accents
    return FromArgb(TextEditor::Palette{{
        0xfff8f8f2, // Default (foreground)
        0xffff79c6, // Keyword (pink)
        0xffbd93f9, // Number (purple)
        0xfff1fa8c, // String (yellow)
        0xfff1fa8c, // Char literal (yellow)
        0xfff8f8f2, // Punctuation (foreground)
        0xffff79c6, // Preprocessor (pink)
        0xfff8f8f2, // Identifier (foreground)
        0xff8be9fd, // Known identifier (cyan)
        0xff50fa7b, // Preproc identifier (green)
        0xff6272a4, // Comment (comment blue-gray)
        0xff6272a4, // Multi-line comment
        0xff282a36, // Background (dark purple-gray)
        0xfff8f8f2, // Cursor (white)
        0x8044475a, // Selection (translucent)
        0xa0ff5555, // Error marker (red)
        0x80ff79c6, // Breakpoint (pink)
        0xff6272a4, // Line number (comment color)
        0x40404050, // Current line fill
        0x30404050, // Current line fill inactive
        0x40404050  // Current line edge
    }});
}

TextEditor::Palette ScriptEditorPanel::GetOneDarkPalette() {
    // One Dark theme - Atom editor inspired
    return FromArgb(TextEditor::Palette{{
        0xffabb2bf, // Default (light gray)
        0xffc678dd, // Keyword (purple)
        0xffd19a66, // Number (orange)
        0xff98c379, // String (green)
        0xff98c379, // Char literal (green)
        0xffabb2bf, // Punctuation (light gray)
        0xffc678dd, // Preprocessor (purple)
        0xffe06c75, // Identifier (red)
        0xff61afef, // Known identifier (blue)
        0xffe5c07b, // Preproc identifier (yellow)
        0xff5c6370, // Comment (gray)
        0xff5c6370, // Multi-line comment (gray)
        0xff282c34, // Background (dark gray)
        0xffabb2bf, // Cursor (white)
        0x803e4451, // Selection (translucent)
        0xa0e06c75, // Error marker (red)
        0x80c678dd, // Breakpoint (purple)
        0xff4b5263, // Line number (gray)
        0x20ffffff, // Current line fill
        0x15ffffff, // Current line fill inactive
        0x20ffffff  // Current line edge
    }});
}

TextEditor::Palette ScriptEditorPanel::GetGitHubPalette() {
    // GitHub Light theme - clean light theme
    return FromArgb(TextEditor::Palette{{
        0xff24292e, // Default (dark gray)
        0xffd73a49, // Keyword (red)
        0xff005cc5, // Number (blue)
        0xff032f62, // String (dark blue)
        0xff032f62, // Char literal (dark blue)
        0xff24292e, // Punctuation (dark gray)
        0xff6f42c1, // Preprocessor (purple)
        0xff24292e, // Identifier (dark gray)
        0xff6f42c1, // Known identifier (purple)
        0xff22863a, // Preproc identifier (green)
        0xff6a737d, // Comment (gray)
        0xff6a737d, // Multi-line comment (gray)
        0xffffffff, // Background (white)
        0xff24292e, // Cursor (dark)
        0x400366d6, // Selection (translucent blue)
        0x40cb2431, // Error marker (red)
        0x40d73a49, // Breakpoint (red)
        0xff959da5, // Line number (light gray)
        0x10000000, // Current line fill
        0x08000000, // Current line fill inactive
        0x10000000  // Current line edge
    }});
}
} // namespace cyxwiz