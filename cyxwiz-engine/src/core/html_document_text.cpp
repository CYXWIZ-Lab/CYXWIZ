#include "html_document_text.h"
#include <algorithm>
#include <arrow/api.h>
#include <arrow/util/utf8.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>
#ifdef CYXWIZ_HAS_HTML_PARSER
#include <lexbor/core/base.h>
#include <lexbor/dom/interfaces/character_data.h>
#include <lexbor/html/parser.h>
#endif
namespace cyxwiz {
namespace {
#ifdef CYXWIZ_HAS_HTML_PARSER
std::string Trim(std::string s) {
  auto first = s.find_first_not_of(" \r\n\t\f");
  return first == std::string::npos
             ? ""
             : s.substr(first, s.find_last_not_of(" \r\n\t\f") - first + 1);
}
bool Block(lxb_tag_id_t tag) {
  switch (tag) {
  case LXB_TAG_P:
  case LXB_TAG_DIV:
  case LXB_TAG_BR:
  case LXB_TAG_HR:
  case LXB_TAG_H1:
  case LXB_TAG_H2:
  case LXB_TAG_H3:
  case LXB_TAG_H4:
  case LXB_TAG_H5:
  case LXB_TAG_H6:
  case LXB_TAG_LI:
  case LXB_TAG_TR:
  case LXB_TAG_TABLE:
  case LXB_TAG_CENTER:
  case LXB_TAG_BLOCKQUOTE:
    return true;
  default:
    return false;
  }
}
struct Extracted {
  std::string title, text;
  size_t nodes = 0;
};
Extracted Extract(const std::string &html, const std::string &begin_comment,
                  const std::string &end_comment,
                  const std::function<bool()> &cancelled) {
  const bool marked = !begin_comment.empty();
  const auto check_cancel = [&] {
    if (cancelled && cancelled())
      throw std::runtime_error("HTML extraction cancelled");
  };
  check_cancel();
  if (html.find('\0') != std::string::npos || !arrow::util::ValidateUTF8(html))
    throw std::runtime_error("HTML input must be UTF-8 without NUL");
  if (html.size() > 1024 * 1024)
    throw std::runtime_error("HTML input limit");
  std::unique_ptr<lxb_html_document_t, decltype(&lxb_html_document_destroy)>
      doc(lxb_html_document_create(), lxb_html_document_destroy);
  if (!doc || lxb_html_document_parse_chunk_begin(doc.get()) != LXB_STATUS_OK)
    throw std::runtime_error("HTML parser initialization failed");
  for (size_t offset = 0; offset < html.size(); offset += 32768) {
    check_cancel();
    if (lxb_html_document_parse_chunk(
            doc.get(),
            reinterpret_cast<const lxb_char_t *>(html.data() + offset),
            std::min<size_t>(32768, html.size() - offset)) != LXB_STATUS_OK)
      throw std::runtime_error("HTML parsing failed");
  }
  check_cancel();
  if (lxb_html_document_parse_chunk_end(doc.get()) != LXB_STATUS_OK ||
      !doc->body)
    throw std::runtime_error("HTML parser completion failed");
  Extracted result;
  size_t title_size = 0;
  auto title = lxb_html_document_title(doc.get(), &title_size);
  if (title)
    result.title.assign(reinterpret_cast<const char *>(title), title_size);
  bool active = !marked, space = false;
  size_t begins = 0, ends = 0;
  auto boundary = [&] {
    space = false;
    if (!result.text.empty() && result.text.back() != '\n')
      result.text += "\n\n";
  };
  struct Visit {
    lxb_dom_node_t *node;
    size_t depth;
    bool exit;
  };
  std::vector<Visit> pending{{lxb_dom_interface_node(doc->body), 0, false}};
  while (!pending.empty()) {
    check_cancel();
    if (result.text.size() > 1024 * 1024)
      throw std::runtime_error("HTML output limit");
    auto [node, depth, exit] = pending.back();
    pending.pop_back();
    if (exit) {
      if (active && Block(node->local_name))
        boundary();
      continue;
    }
    if (++result.nodes > 200000 || depth > 256)
      throw std::runtime_error("HTML tree limit");
    if (node->local_name == LXB_TAG_SCRIPT ||
        node->local_name == LXB_TAG_STYLE ||
        node->local_name == LXB_TAG_TEMPLATE)
      continue;
    if (node->type == LXB_DOM_NODE_TYPE_COMMENT && marked) {
      auto data = lxb_dom_interface_character_data(node);
      auto comment = Trim(std::string(
          reinterpret_cast<const char *>(data->data.data), data->data.length));
      if (comment == begin_comment) {
        if (++begins != 1 || ends)
          throw std::runtime_error("Ambiguous begin marker");
        active = true;
      } else if (comment == end_comment) {
        if (++ends != 1 || begins != 1)
          throw std::runtime_error("Ambiguous end marker");
        active = false;
      }
    }
    if (active && Block(node->local_name))
      boundary();
    if (active &&
        (node->local_name == LXB_TAG_TD || node->local_name == LXB_TAG_TH))
      space = true;
    if (active && node->type == LXB_DOM_NODE_TYPE_TEXT) {
      auto data = lxb_dom_interface_character_data(node);
      std::string text(reinterpret_cast<const char *>(data->data.data),
                       data->data.length);
      for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = text[i];
        if (c == 0xc2 && i + 1 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0xa0) {
          space = true;
          ++i;
          continue;
        }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f') {
          space = true;
          continue;
        }
        if (space && !result.text.empty() && result.text.back() != '\n')
          result.text += ' ';
        space = false;
        result.text += text[i];
      }
    }
    pending.push_back({node, depth, true});
    for (auto child = node->last_child; child; child = child->prev) {
      if (pending.size() >= 200000)
        throw std::runtime_error("HTML pending node limit");
      pending.push_back({child, depth + 1, false});
    }
  }
  if (marked && (begins != 1 || ends != 1))
    throw std::runtime_error("Missing exact body marker pair");
  result.text = Trim(result.text);
  return result;
}

#endif
void Require(const arrow::Status &status) {
  if (!status.ok())
    throw std::runtime_error(status.ToString());
}
} // namespace
bool HtmlDocumentParserAvailable() {
#ifdef CYXWIZ_HAS_HTML_PARSER
  return true;
#else
  return false;
#endif
}
std::shared_ptr<arrow::Table> CleanHtmlDocumentTable(
    const std::shared_ptr<arrow::Table> &input, const std::string &column,
    const std::string &begin_comment, const std::string &end_comment,
    const std::function<bool()> &cancelled) {
#ifndef CYXWIZ_HAS_HTML_PARSER
  (void)input;
  (void)column;
  (void)begin_comment;
  (void)end_comment;
  (void)cancelled;
  throw std::runtime_error("Parsed HTML unavailable: build with html-parser "
                           "vcpkg feature and CYXWIZ_ENABLE_HTML_PARSER=ON");
#else
  if (!input)
    throw std::runtime_error("HTML input table is null");
  Require(input->ValidateFull());
  if (begin_comment.empty() != end_comment.empty() ||
      (!begin_comment.empty() && begin_comment == end_comment) ||
      begin_comment.size() > 4096 || end_comment.size() > 4096)
    throw std::runtime_error("HTML requires two different comment markers or "
                             "neither (maximum 4096 bytes each)");
  static const bool initialized = [] {
    arrow::util::InitializeUTF8();
    return true;
  }();
  (void)initialized;
  if (begin_comment.find('\0') != std::string::npos ||
      end_comment.find('\0') != std::string::npos ||
      !arrow::util::ValidateUTF8(begin_comment) ||
      !arrow::util::ValidateUTF8(end_comment))
    throw std::runtime_error("HTML markers must be UTF-8 without NUL");
  const auto indices = input->schema()->GetAllFieldIndices(column);
  if (indices.size() != 1)
    throw std::runtime_error("HTML text column must exist exactly once: " +
                             column);
  const auto source = input->column(indices[0]);
  if (source->type()->id() != arrow::Type::STRING &&
      source->type()->id() != arrow::Type::LARGE_STRING)
    throw std::runtime_error(
        "HTML text column must have string or large_string type");
  const std::vector<std::string> names{
      column + "_cleaned", column + "_html_title", column + "_html_policy"};
  for (const auto &name : names)
    if (!input->schema()->GetAllFieldIndices(name).empty())
      throw std::runtime_error("HTML output column already exists: " + name);
  const std::string policy =
      nlohmann::json({{"mode", "parsed_html_v1"},
                      {"parser", "lexbor"},
                      {"parser_version", LEXBOR_VERSION_STRING},
                      {"begin_comment", begin_comment},
                      {"end_comment", end_comment},
                      {"input_byte_limit", 1048576},
                      {"output_byte_limit", 1048576},
                      {"table_output_byte_limit", 67108864},
                      {"node_limit", 200000},
                      {"depth_limit", 256},
                      {"paragraphs", true}})
          .dump();
  arrow::StringBuilder texts, titles, policies;
  size_t total_bytes = 0;
  int64_t row = 0;
  for (const auto &chunk : source->chunks()) {
    for (int64_t i = 0; i < chunk->length(); ++i, ++row) {
      if (cancelled && cancelled())
        throw std::runtime_error("HTML extraction cancelled");
      if (chunk->IsNull(i)) {
        Require(texts.AppendNull());
        Require(titles.AppendNull());
        Require(policies.AppendNull());
        continue;
      }
      auto value =
          source->type()->id() == arrow::Type::STRING
              ? std::static_pointer_cast<arrow::StringArray>(chunk)->GetView(i)
              : std::static_pointer_cast<arrow::LargeStringArray>(chunk)
                    ->GetView(i);
      if (value.size() > 1048576)
        throw std::runtime_error("HTML input exceeds 1 MiB at row " +
                                 std::to_string(row));
      Extracted result;
      try {
        result =
            Extract(std::string(value), begin_comment, end_comment, cancelled);
      } catch (const std::exception &e) {
        throw std::runtime_error("HTML row " + std::to_string(row) + ": " +
                                 e.what());
      }
      const size_t bytes =
          result.text.size() + result.title.size() + policy.size();
      if (bytes > 67108864 - total_bytes)
        throw std::runtime_error("HTML table output byte limit exceeded");
      total_bytes += bytes;
      Require(texts.Append(result.text));
      Require(titles.Append(result.title));
      Require(policies.Append(policy));
    }
  }
  auto output = input;
  arrow::StringBuilder *builders[]{&texts, &titles, &policies};
  for (size_t i = 0; i < names.size(); ++i) {
    auto array = builders[i]->Finish();
    if (!array.ok())
      throw std::runtime_error(array.status().ToString());
    auto added = output->AddColumn(
        output->num_columns(), arrow::field(names[i], arrow::utf8()),
        std::make_shared<arrow::ChunkedArray>(*array));
    if (!added.ok())
      throw std::runtime_error(added.status().ToString());
    output = *added;
  }
  if (cancelled && cancelled())
    throw std::runtime_error("HTML extraction cancelled");
  Require(output->ValidateFull());
  return output;
#endif
}
} // namespace cyxwiz

