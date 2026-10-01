// .ipynb read/write (TOFIX133 P4, D3): cells, outputs of every kind, and a
// lossless round trip of what this Engine does not draw.
#include "../src/core/notebook_format.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::nb;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// A notebook as Jupyter writes it, with outputs of each kind.
const char* kNotebook = R"JSON({
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a1",
   "metadata": {"tags": ["intro"]},
   "source": ["## Sentiment model check\n", "Loads the vocabulary."]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b2",
   "metadata": {"collapsed": false},
   "outputs": [
    {"name": "stdout", "output_type": "stream", "text": ["Normal  16351\n", "Depression  15404\n"]},
    {"name": "stderr", "output_type": "stream", "text": "362 rows have no statement\n"},
    {"data": {"text/plain": ["5000"]}, "execution_count": 1, "metadata": {}, "output_type": "execute_result"}
   ],
   "source": "len(vocab)"
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c3",
   "metadata": {},
   "outputs": [
    {"data": {"image/png": "iVBORw0KGgo=", "text/plain": ["<Figure size 640x480 with 1 Axes>"], "application/vnd.custom+json": {"x": 1}},
     "metadata": {"needs_background": "light"}, "output_type": "display_data"},
    {"ename": "RuntimeError", "evalue": "Failed to reach embedded server", "output_type": "error",
     "traceback": ["Traceback (most recent call last)", "RuntimeError: Failed to reach embedded server"]}
   ],
   "source": ["df.plot()\n", "ensure_server_ready()"]
  },
  {"cell_type": "code", "execution_count": null, "id": "d4", "metadata": {}, "outputs": [], "source": []}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "custom": {"keep": true}},
 "nbformat": 4,
 "nbformat_minor": 5
})JSON";
}  // namespace

int main() {
    Notebook nb;
    std::string error;
    Check(ParseIpynb(kNotebook, nb, &error), "parse: " + error);
    Check(nb.cells.size() == 4, "four cells");
    Check(nb.cells[0].kind == CellKind::Markdown && nb.cells[0].source == "## Sentiment model check\nLoads the vocabulary.",
          "markdown source joined from lines");
    const auto& code = nb.cells[1];
    Check(code.kind == CellKind::Code && code.execution_count == 1 && code.source == "len(vocab)", "code cell, count, string source");
    Check(code.outputs.size() == 3, "three outputs");
    Check(code.outputs[0].kind == Output::Kind::Stream && code.outputs[0].stream_name == "stdout" &&
              code.outputs[0].text == "Normal  16351\nDepression  15404\n",
          "stdout stream");
    Check(code.outputs[1].stream_name == "stderr", "stderr kept apart");
    Check(code.outputs[2].kind == Output::Kind::Result && code.outputs[2].text == "5000", "execute_result text/plain");
    const auto& plot = nb.cells[2];
    Check(plot.outputs[0].kind == Output::Kind::Display && plot.outputs[0].png_base64 == "iVBORw0KGgo=", "image/png");
    Check(plot.outputs[1].kind == Output::Kind::Error && plot.outputs[1].ename == "RuntimeError" &&
              plot.outputs[1].traceback.size() == 2,
          "error output");
    Check(nb.cells[3].execution_count == 0 && nb.cells[3].source.empty(), "null count, empty source");

    // Lossless: what we do not draw (custom MIME, metadata, ids) survives.
    const std::string written = SerializeIpynb(nb);
    auto a = nlohmann::json::parse(kNotebook);
    a["cells"][1]["source"] = nlohmann::json::array({"len(vocab)"});  // a one-string source is written as lines, as Jupyter does
    const auto b = nlohmann::json::parse(written);
    if (a != b) std::cerr << nlohmann::json::diff(a, b).dump(1) << '\n';
    Check(a == b, "round trip is identical JSON");
    Notebook again;
    Check(ParseIpynb(written, again) && SerializeIpynb(again) == written, "stable over repeated saves");

    // A cell edited and run here is written fresh, with lines split as Jupyter does.
    nb.cells[3].source = "x = 1\nx";
    nb.cells[3].execution_count = 3;
    Output result;
    result.kind = Output::Kind::Result;
    result.text = "1";
    result.execution_count = 3;
    nb.cells[3].outputs.push_back(result);
    const auto c = nlohmann::json::parse(SerializeIpynb(nb));
    const auto& cell = c["cells"][3];
    Check(cell["source"].size() == 2 && cell["source"][0] == "x = 1\n" && cell["source"][1] == "x", "source split into lines");
    Check(cell["outputs"][0]["output_type"] == "execute_result" && cell["outputs"][0]["data"]["text/plain"][0] == "1" &&
              cell["execution_count"] == 3 && cell["id"] == "d4",
          "new result written; id kept");

    Check(!ParseIpynb("{\"nbformat\": 4}", nb, &error) && !error.empty(), "no cell list is an error");
    Check(!ParseIpynb("{\"cells\": [], \"nbformat\": 3}", nb, &error) && error.find("older") != std::string::npos,
          "nbformat 3 refused with a reason");

    // Errors from text: the Engine's form and a Python traceback.
    const Output engine = ErrorFromText("ModuleNotFoundError: No module named 'pandas'\n\nAt:\n  <string>(2): <module>\n");
    Check(engine.ename == "ModuleNotFoundError" && engine.evalue == "No module named 'pandas'" && engine.traceback.size() == 4,
          "Engine error form");
    const Output python = ErrorFromText(
        "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n    a: int = d['k']\nKeyError: 'score'");
    Check(python.ename == "KeyError" && python.evalue == "'score'", "Python traceback form");
    const Output dotted = ErrorFromText("pandas.errors.ParserError: Error tokenizing data");
    Check(dotted.ename == "pandas.errors.ParserError", "dotted exception name");
    const Output plain = ErrorFromText("Execution interrupted");
    Check(plain.ename == "Error" && plain.evalue == "Execution interrupted", "text without a name");

    const std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0xFF};
    Check(EncodeBase64(png) == "iVBORw0KGgoA/w==", "base64 encode");
    Check(DecodeBase64("iVBORw0K\nGgoA/w==\n") == png, "base64 decode skips line breaks");
    Check(DecodeBase64("not*base64").empty(), "invalid base64 is empty");
    Check(StripAnsi("\x1b[0;31mValueError\x1b[0m: bad") == "ValueError: bad", "ANSI colours dropped");

    std::cout << "notebook format: cells, streams, results, images, errors, lossless round trip. OK\n";
    return 0;
}
