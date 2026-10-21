#include "python_language.h"

#include "../core/python_tokenizer.h"

#include <string>

namespace cyxwiz {

namespace {
// TOFIX133 P0 item 4: the Python definition had no tokenizer, so only
// comments were coloured. Multi-line strings and comments are marked by the
// editor's Python pass (mPythonStrings).
bool TokenizePython(const char* in_begin, const char* in_end, const char*& out_begin, const char*& out_end,
                    TextEditor::PaletteIndex& palette) {
    pytokens::Kind kind;
    if (!pytokens::Next(in_begin, in_end, out_begin, out_end, kind)) return false;
    switch (kind) {
        case pytokens::Kind::String: palette = TextEditor::PaletteIndex::String; break;
        case pytokens::Kind::Number: palette = TextEditor::PaletteIndex::Number; break;
        case pytokens::Kind::Identifier: palette = TextEditor::PaletteIndex::Identifier; break;
        case pytokens::Kind::Punctuation: palette = TextEditor::PaletteIndex::Punctuation; break;
        case pytokens::Kind::Decorator: palette = TextEditor::PaletteIndex::Preprocessor; break;
    }
    return true;
}
}  // namespace

const TextEditor::LanguageDefinition& PythonLanguage() {
    static bool inited = false;
    static TextEditor::LanguageDefinition lang;

    if (!inited) {
        lang.mName = "Python";
        lang.mCaseSensitive = true;
        lang.mAutoIndentation = true;

        // Comments and strings: the editor's Python pass (# comments, ' and "
        // strings, triple-quoted strings over several lines), then the tokenizer.
        lang.mSingleLineComment = "#";
        lang.mPythonStrings = true;
        lang.mTokenize = TokenizePython;

        // Add preprocessor patterns for %% section markers
        lang.mPreprocChar = '%';

        // Python keywords
        static const char* const keywords[] = {
            "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
            "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
            "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
            "while", "with", "yield", "async", "await"
        };

        for (auto& k : keywords) {
            lang.mKeywords.insert(k);
        }

        // Built-in identifiers
        static const char* const identifiers[] = {
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
            "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate",
            "eval", "exec", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr",
            "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len",
            "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open",
            "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set", "setattr",
            "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip"
        };

        for (auto& i : identifiers) {
            TextEditor::Identifier id;
            id.mDeclaration = "Built-in function";
            lang.mIdentifiers.insert(std::make_pair(std::string(i), id));
        }

        inited = true;
    }

    return lang;
}

}  // namespace cyxwiz
