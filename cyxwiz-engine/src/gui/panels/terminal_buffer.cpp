#include "terminal_buffer.h"

#include <vterm.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

constexpr std::uint32_t kDefaultForeground = 0xD4D4D4;
constexpr std::uint32_t kDefaultBackground = 0x101418;
constexpr std::size_t kMaxScrollbackLines = 2000;
constexpr std::size_t kMaxOutboundBytes = 64 * 1024;

constexpr std::array<std::uint32_t, 16> kAnsiColors{
    0x1E1E1E, 0xCD3131, 0x0DBC79, 0xE5E510, 0x2472C8, 0xBC3FBC,
    0x11A8CD, 0xE5E5E5, 0x666666, 0xF14C4C, 0x23D18B, 0xF5F543,
    0x3B8EEA, 0xD670D6, 0x29B8DB, 0xFFFFFF,
};

int ClampDimension(std::size_t value, int minimum) {
  return static_cast<int>(std::clamp<std::size_t>(
      value, static_cast<std::size_t>(minimum),
      static_cast<std::size_t>(std::numeric_limits<int>::max())));
}

void AppendUtf8(std::string &output, char32_t value) {
  if (value <= 0x7F) {
    output.push_back(static_cast<char>(value));
  } else if (value <= 0x7FF) {
    output.push_back(static_cast<char>(0xC0 | (value >> 6)));
    output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
  } else if (value <= 0xFFFF) {
    output.push_back(static_cast<char>(0xE0 | (value >> 12)));
    output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
  } else {
    output.push_back(static_cast<char>(0xF0 | (value >> 18)));
    output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (value & 0x3F)));
  }
}

VTermColor RgbColor(std::uint32_t rgb) {
  VTermColor color{};
  vterm_color_rgb(&color, static_cast<std::uint8_t>((rgb >> 16) & 0xFF),
                  static_cast<std::uint8_t>((rgb >> 8) & 0xFF),
                  static_cast<std::uint8_t>(rgb & 0xFF));
  return color;
}

} // namespace

struct TerminalBuffer::Impl {
  using StoredLine = std::vector<VTermScreenCell>;

  Impl(std::size_t requested_columns, std::size_t requested_rows)
      : columns(ClampDimension(requested_columns, 2)),
        rows(ClampDimension(requested_rows, 1)), vt(vterm_new(rows, columns)) {
    if (!vt)
      throw std::bad_alloc();

    vterm_set_utf8(vt, 1);
    vterm_output_set_callback(vt, &Impl::OnOutput, this);
    state = vterm_obtain_state(vt);
    screen = vterm_obtain_screen(vt);
    if (!state || !screen) {
      vterm_free(vt);
      vt = nullptr;
      throw std::runtime_error("libvterm failed to create terminal state");
    }

    vterm_screen_enable_altscreen(screen, 1);
    vterm_screen_enable_reflow(screen, true);
    vterm_screen_set_damage_merge(screen, VTERM_DAMAGE_SCROLL);
    vterm_screen_set_callbacks(screen, &kScreenCallbacks, this);

    const VTermColor foreground = RgbColor(kDefaultForeground);
    const VTermColor background = RgbColor(kDefaultBackground);
    vterm_screen_set_default_colors(screen, &foreground, &background);
    for (std::size_t index = 0; index < kAnsiColors.size(); ++index) {
      const VTermColor color = RgbColor(kAnsiColors[index]);
      vterm_state_set_palette_color(state, static_cast<int>(index), &color);
    }
    vterm_screen_reset(screen, 1);
    vterm_screen_flush_damage(screen);
  }

  ~Impl() {
    if (vt)
      vterm_free(vt);
  }

  static void OnOutput(const char *bytes, std::size_t length, void *user) {
    auto &self = *static_cast<Impl *>(user);
    const std::size_t available = self.outbound.size() < kMaxOutboundBytes
                                      ? kMaxOutboundBytes - self.outbound.size()
                                      : 0;
    self.outbound.append(bytes, std::min(length, available));
  }

  static int OnMoveCursor(VTermPos position, VTermPos, int visible,
                          void *user) {
    auto &self = *static_cast<Impl *>(user);
    self.cursor_row = std::max(position.row, 0);
    self.cursor_column = std::max(position.col, 0);
    self.cursor_visible = visible != 0;
    return 1;
  }

  static int OnTermProperty(VTermProp property, VTermValue *value, void *user) {
    auto &self = *static_cast<Impl *>(user);
    if (property == VTERM_PROP_ALTSCREEN)
      self.alternate_screen = value->boolean != 0;
    else if (property == VTERM_PROP_CURSORVISIBLE)
      self.cursor_visible = value->boolean != 0;
    return 1;
  }

  static int OnResize(int new_rows, int new_columns, void *user) {
    auto &self = *static_cast<Impl *>(user);
    self.rows = std::max(new_rows, 1);
    self.columns = std::max(new_columns, 2);
    return 1;
  }

  static int OnScrollbackPush(int count, const VTermScreenCell *cells,
                              void *user) {
    auto &self = *static_cast<Impl *>(user);
    self.scrollback.emplace_back(cells, cells + std::max(count, 0));
    if (self.scrollback.size() > kMaxScrollbackLines)
      self.scrollback.pop_front();
    return 1;
  }

  static int OnScrollbackPop(int count, VTermScreenCell *cells, void *user) {
    auto &self = *static_cast<Impl *>(user);
    if (self.scrollback.empty())
      return 0;
    const StoredLine line = std::move(self.scrollback.back());
    self.scrollback.pop_back();
    const int copied = std::min(count, static_cast<int>(line.size()));
    std::copy_n(line.begin(), copied, cells);
    if (copied < count)
      std::memset(cells + copied, 0,
                  static_cast<std::size_t>(count - copied) * sizeof(*cells));
    return 1;
  }

  static int OnScrollbackClear(void *user) {
    static_cast<Impl *>(user)->scrollback.clear();
    return 1;
  }

  std::uint32_t ConvertColor(VTermColor color) const {
    vterm_screen_convert_color_to_rgb(screen, &color);
    return (static_cast<std::uint32_t>(color.rgb.red) << 16) |
           (static_cast<std::uint32_t>(color.rgb.green) << 8) |
           static_cast<std::uint32_t>(color.rgb.blue);
  }

  TerminalBuffer::Cell ConvertCell(const VTermScreenCell &source) const {
    TerminalBuffer::Cell result;
    result.foreground = ConvertColor(source.fg);
    result.background = ConvertColor(source.bg);
    if (source.attrs.reverse)
      std::swap(result.foreground, result.background);
    if (source.attrs.conceal)
      result.foreground = result.background;
    result.bold = source.attrs.bold != 0;
    result.underline = source.attrs.underline != 0;
    result.strike = source.attrs.strike != 0;
    if (source.chars[0] != 0)
      result.codepoint = static_cast<char32_t>(source.chars[0]);
    for (std::size_t index = 1;
         index < VTERM_MAX_CHARS_PER_CELL && source.chars[index] != 0;
         ++index) {
      result.combining[result.combining_count++] =
          static_cast<char32_t>(source.chars[index]);
    }
    return result;
  }

  const TerminalBuffer::Line &BuildLine(std::size_t visible_row,
                                        const StoredLine *stored) const {
    line_cache.assign(static_cast<std::size_t>(columns),
                      TerminalBuffer::Cell{});
    int continuation_columns = 0;
    for (int column = 0; column < columns; ++column) {
      if (continuation_columns > 0) {
        line_cache[static_cast<std::size_t>(column)].continuation = true;
        --continuation_columns;
        continue;
      }

      VTermScreenCell source{};
      if (stored) {
        if (static_cast<std::size_t>(column) >= stored->size())
          continue;
        source = (*stored)[static_cast<std::size_t>(column)];
      } else if (!vterm_screen_get_cell(
                     screen, VTermPos{static_cast<int>(visible_row), column},
                     &source)) {
        continue;
      }
      line_cache[static_cast<std::size_t>(column)] = ConvertCell(source);
      continuation_columns = std::max(static_cast<int>(source.width) - 1, 0);
    }
    return line_cache;
  }

  static const VTermScreenCallbacks kScreenCallbacks;

  int columns = 80;
  int rows = 24;
  VTerm *vt = nullptr;
  VTermState *state = nullptr;
  VTermScreen *screen = nullptr;
  std::deque<StoredLine> scrollback;
  mutable TerminalBuffer::Line line_cache;
  std::string outbound;
  int cursor_column = 0;
  int cursor_row = 0;
  bool cursor_visible = true;
  bool alternate_screen = false;
};

const VTermScreenCallbacks TerminalBuffer::Impl::kScreenCallbacks{
    nullptr,
    nullptr,
    &TerminalBuffer::Impl::OnMoveCursor,
    &TerminalBuffer::Impl::OnTermProperty,
    nullptr,
    &TerminalBuffer::Impl::OnResize,
    &TerminalBuffer::Impl::OnScrollbackPush,
    &TerminalBuffer::Impl::OnScrollbackPop,
    &TerminalBuffer::Impl::OnScrollbackClear,
};

TerminalBuffer::TerminalBuffer(std::size_t columns, std::size_t rows)
    : impl_(std::make_unique<Impl>(columns, rows)) {}

TerminalBuffer::~TerminalBuffer() = default;
TerminalBuffer::TerminalBuffer(TerminalBuffer &&) noexcept = default;
TerminalBuffer &TerminalBuffer::operator=(TerminalBuffer &&) noexcept = default;

void TerminalBuffer::Resize(std::size_t columns, std::size_t rows) {
  vterm_set_size(impl_->vt, ClampDimension(rows, 1),
                 ClampDimension(columns, 2));
  vterm_screen_flush_damage(impl_->screen);
}

void TerminalBuffer::Feed(std::string_view bytes) {
  std::size_t consumed = 0;
  while (consumed < bytes.size()) {
    const std::size_t count = vterm_input_write(
        impl_->vt, bytes.data() + consumed, bytes.size() - consumed);
    if (count == 0)
      break;
    consumed += count;
  }
  vterm_screen_flush_damage(impl_->screen);
}

void TerminalBuffer::Clear() {
  impl_->scrollback.clear();
  impl_->outbound.clear();
  vterm_screen_reset(impl_->screen, 1);
  vterm_screen_flush_damage(impl_->screen);
}

std::string TerminalBuffer::TakeOutboundData() {
  std::string result = std::move(impl_->outbound);
  impl_->outbound.clear();
  return result;
}

std::string TerminalBuffer::EncodeCharacter(char32_t codepoint,
                                            std::uint8_t modifiers) {
  vterm_keyboard_unichar(impl_->vt, static_cast<std::uint32_t>(codepoint),
                         static_cast<VTermModifier>(modifiers));
  return TakeOutboundData();
}

std::string TerminalBuffer::EncodeKey(Key key, std::uint8_t modifiers) {
  VTermKey vterm_key = VTERM_KEY_NONE;
  switch (key) {
  case Key::Enter:
    vterm_key = VTERM_KEY_ENTER;
    break;
  case Key::Tab:
    vterm_key = VTERM_KEY_TAB;
    break;
  case Key::Backspace:
    vterm_key = VTERM_KEY_BACKSPACE;
    break;
  case Key::Escape:
    vterm_key = VTERM_KEY_ESCAPE;
    break;
  case Key::Up:
    vterm_key = VTERM_KEY_UP;
    break;
  case Key::Down:
    vterm_key = VTERM_KEY_DOWN;
    break;
  case Key::Left:
    vterm_key = VTERM_KEY_LEFT;
    break;
  case Key::Right:
    vterm_key = VTERM_KEY_RIGHT;
    break;
  case Key::Insert:
    vterm_key = VTERM_KEY_INS;
    break;
  case Key::Delete:
    vterm_key = VTERM_KEY_DEL;
    break;
  case Key::Home:
    vterm_key = VTERM_KEY_HOME;
    break;
  case Key::End:
    vterm_key = VTERM_KEY_END;
    break;
  case Key::PageUp:
    vterm_key = VTERM_KEY_PAGEUP;
    break;
  case Key::PageDown:
    vterm_key = VTERM_KEY_PAGEDOWN;
    break;
  case Key::Function1:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(1));
    break;
  case Key::Function2:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(2));
    break;
  case Key::Function3:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(3));
    break;
  case Key::Function4:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(4));
    break;
  case Key::Function5:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(5));
    break;
  case Key::Function6:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(6));
    break;
  case Key::Function7:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(7));
    break;
  case Key::Function8:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(8));
    break;
  case Key::Function9:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(9));
    break;
  case Key::Function10:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(10));
    break;
  case Key::Function11:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(11));
    break;
  case Key::Function12:
    vterm_key = static_cast<VTermKey>(VTERM_KEY_FUNCTION(12));
    break;
  }
  vterm_keyboard_key(impl_->vt, vterm_key,
                     static_cast<VTermModifier>(modifiers));
  return TakeOutboundData();
}

std::string TerminalBuffer::EncodePaste(std::string_view text) {
  vterm_keyboard_start_paste(impl_->vt);
  std::string result = TakeOutboundData();
  result.append(text);
  vterm_keyboard_end_paste(impl_->vt);
  result += TakeOutboundData();
  return result;
}

std::size_t TerminalBuffer::Columns() const {
  return static_cast<std::size_t>(impl_->columns);
}

std::size_t TerminalBuffer::Rows() const {
  return static_cast<std::size_t>(impl_->rows);
}

std::size_t TerminalBuffer::TotalLineCount() const {
  const std::size_t history =
      impl_->alternate_screen ? 0 : impl_->scrollback.size();
  return history + Rows();
}

const TerminalBuffer::Line &TerminalBuffer::LineAt(std::size_t index) const {
  if (!impl_->alternate_screen && index < impl_->scrollback.size())
    return impl_->BuildLine(0, &impl_->scrollback[index]);
  const std::size_t history =
      impl_->alternate_screen ? 0 : impl_->scrollback.size();
  const std::size_t row =
      std::min(index >= history ? index - history : 0, Rows() - 1);
  return impl_->BuildLine(row, nullptr);
}

std::string TerminalBuffer::PlainText() const {
  std::string output;
  for (std::size_t row = 0; row < TotalLineCount(); ++row) {
    const Line &line = LineAt(row);
    std::size_t end = line.size();
    while (end > 0 &&
           (line[end - 1].codepoint == U' ' || line[end - 1].continuation))
      --end;
    for (std::size_t column = 0; column < end; ++column) {
      const Cell &cell = line[column];
      if (cell.continuation)
        continue;
      AppendUtf8(output, cell.codepoint);
      for (std::size_t index = 0; index < cell.combining_count; ++index)
        AppendUtf8(output, cell.combining[index]);
    }
    if (row + 1 < TotalLineCount())
      output.push_back('\n');
  }
  return output;
}

std::size_t TerminalBuffer::CursorColumn() const {
  return static_cast<std::size_t>(std::max(impl_->cursor_column, 0));
}

std::size_t TerminalBuffer::CursorRow() const {
  return static_cast<std::size_t>(std::max(impl_->cursor_row, 0));
}

bool TerminalBuffer::CursorVisible() const { return impl_->cursor_visible; }

bool TerminalBuffer::AlternateScreen() const { return impl_->alternate_screen; }

} // namespace cyxwiz
