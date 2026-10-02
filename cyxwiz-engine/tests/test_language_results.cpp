// Language tool results (TOFIX133 P3): the JSON cyxwiz_intel.py returned
// for p3_scene.py (real output, 2026-10-02) as typed structs.
#include "../src/core/language_results.h"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::lang;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    const auto c = ParseCompletions(
        R"J([{"name": "load", "complete": "ad", "kind": "function", "detail": "load(fp: SupportsRead[str | bytes], *, cls=None)"},
            {"name": "loads", "complete": "ads", "kind": "function", "detail": "loads(s: str | bytes | bytearray)"}])J");
    Check(c.size() == 2 && c[0].name == "load" && c[0].complete == "ad" && c[0].kind == "function", "completions");

    const Description d = ParseDescription(R"J({"name": "load", "kind": "function", "signature": "load(fp)", "doc": "Deserialize fp", "module": "json"})J");
    Check(d.module == "json" && d.doc == "Deserialize fp", "description");

    const Hover h = ParseHover(
        R"J({"name": "load_vocab", "kind": "function", "signature": "load_vocab(path: Path) -> dict[str, int]", "type": "", "doc": "",
            "module": "sentiment_analysis_inference", "path": "D:\\tmp\\gui129\\nbproj\\sentiment_analysis_inference.py", "line": 199})J");
    Check(h.name == "load_vocab" && h.line == 199 && h.path.find("sentiment_analysis_inference.py") != std::string::npos, "hover");

    const auto s = ParseSignatures(
        R"J([{"name": "encode_text", "params": ["text: str", "vocab: dict[str, int]", "tokenizer_type: str", "lowercase: bool", "max_length: int"], "index": 2, "doc": ""}])J");
    Check(s.size() == 1 && s[0].params.size() == 5 && s[0].index == 2 && s[0].params[2] == "tokenizer_type: str", "signature");
    Check(ParseSignatures(R"J([{"name": "f", "params": [], "index": null}])J")[0].index == -1, "no active parameter");
    {
        Signature sig = s[0];
        sig.returns = "-> list[float]";
        sig.path = "D:/demo/sentiment_analysis_inference.py";
        sig.line = 209;
        const auto pieces = SignaturePieces(sig);
        Check(pieces.size() == 11 && pieces[0].text == "encode_text(" && pieces[5].active && pieces[5].text == "tokenizer_type: str" &&
                  pieces.back().text == ") -> list[float]",
              "signature pieces, the typed parameter marked");
        Check(SignatureFooter(sig) == "Parameter 3 of 5 \xC2\xB7 sentiment_analysis_inference.py:209", "signature footer");
        Signature none;
        none.name = "f";
        Check(SignatureFooter(none) == "No parameters" && SignaturePieces(none).size() == 2, "no parameters");
        sig.index = -1;
        sig.path.clear();
        sig.line = 0;
        sig.module = "mod";
        Check(SignatureFooter(sig) == "5 parameters \xC2\xB7 mod", "no parameter typed yet");
    }
    {
        Hover hf;
        hf.name = "load_vocab";
        hf.kind = "function";
        hf.signature = "load_vocab(path: Path) -> dict[str, int]";
        Check(HoverHeadline(hf) == "def load_vocab(path: Path) -> dict[str, int]", "hover headline: function");
        Hover v;
        v.name = "vocab";
        v.kind = "variable";
        v.type = "dict";
        Check(HoverHeadline(v) == "vocab: dict", "hover headline: variable type");
        Check(LocationLabel("", 12, "") == "line 12" && LocationLabel("/a/b.py", 0, "b") == "b.py" && LocationLabel("", 0, "json") == "json",
              "location labels");
    }

    const auto l = ParseLocations(R"J([{"name": "load_vocab", "path": "D:\\x\\a.py", "line": 199, "column": 4, "module": "a", "in_source": false}])J");
    Check(l.size() == 1 && l[0].line == 199 && l[0].column == 4 && !l[0].in_source, "definition");

    const auto p = ParseProblems(
        R"J([{"line": 2, "column": 0, "severity": "warning", "message": "'sys' imported but unused", "code": "UnusedImport"},
            {"line": 11, "column": 74, "severity": "error", "message": "undefined name 'max_len'", "code": "UndefinedName"},
            {"line": 12, "column": 16, "severity": "error", "message": "undefined name 'lables'", "code": "UndefinedName"}])J");
    Check(p.size() == 3 && !p[0].error && p[1].error && p[1].column == 74, "problems");
    Check(ProblemSummary(p) == "2 errors, 1 warning", "summary");
    Check(ProblemSummary({}) == "No problems", "no problems");
    Check(ProblemRange("print(len(ids), lables)", p[2]) == std::make_pair(16, 22), "underline the undefined name");
    Check(ProblemRange("import sys", p[0]) == std::make_pair(7, 10), "an unused import underlines its name, not 'import'");
    Problem tok{6, 0, false, "'sentiment_analysis_inference.tokenize' imported but unused", "UnusedImport"};
    Check(ProblemRange("from sentiment_analysis_inference import encode_text, load_vocab, tokenize", tok) ==
              std::make_pair(66, 74),
          "the imported name among several");
    Problem past{1, 5, true, "x", "UndefinedName"};
    Check(ProblemRange("x", past) == std::make_pair(5, 6), "past the end: one column");

    const std::string doc =
        "Deserialize ``fp`` (a ``.read()``-supporting file-like object containing\n"
        "a JSON document) to a Python object.\n\n"
        "``object_hook`` is an optional function that will be called with the\n"
        "result of any object literal decode (a ``dict``).\n\n"
        "Third paragraph is dropped.";
    Check(ReflowDoc(doc, 2) ==
              "Deserialize fp (a .read()-supporting file-like object containing a JSON document) to a Python object.\n\n"
              "object_hook is an optional function that will be called with the result of any object literal decode (a dict).",
          "docstring reflowed, two paragraphs, code marks dropped");
    Check(ReflowDoc("Example:\n\n    x = 1\n    y = 2", 2) == "Example:\n\n    x = 1\n    y = 2", "indented example keeps its lines");
    Check(std::string(ChipFor("function").letter) == "f" && ChipFor("class").role == 1 && ChipFor("module").role == 2, "kind chips");

    // Bad input never throws.
    Check(ParseCompletions("not json").empty() && ParseHover("[]").name.empty() && ParseProblems("{}").empty(), "bad input");

    std::cout << "language results: completions, hover, signatures, definitions, problems. OK\n";
    return 0;
}
