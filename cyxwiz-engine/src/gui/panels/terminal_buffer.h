#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cyxwiz {

class TerminalBuffer {
public:
  enum class Key : std::uint8_t {
    Enter,
    Tab,
    Backspace,
    Escape,
    Up,
    Down,
    Left,
    Right,
    Insert,
    Delete,
    Home,
    End,
    PageUp,
    PageDown,
    Function1,
    Function2,
    Function3,
    Function4,
    Function5,
    Function6,
    Function7,
    Function8,
    Function9,
    Function10,
    Function11,
    Function12,
  };

  enum Modifier : std::uint8_t {
    NoModifier = 0,
    Shift = 1 << 0,
    Alt = 1 << 1,
    Control = 1 << 2,
  };

  struct Cell {
    char32_t codepoint = U' ';
    std::array<char32_t, 5> combining{};
    std::uint8_t combining_count = 0;
    std::uint32_t foreground = 0xD4D4D4;
    std::uint32_t background = 0x101418;
    bool bold = false;
    bool underline = false;
    bool strike = false;
    bool continuation = false;
  };

  using Line = std::vector<Cell>;

  TerminalBuffer(std::size_t columns = 80, std::size_t rows = 24);
  ~TerminalBuffer();

  TerminalBuffer(const TerminalBuffer &) = delete;
  TerminalBuffer &operator=(const TerminalBuffer &) = delete;
  TerminalBuffer(TerminalBuffer &&) noexcept;
  TerminalBuffer &operator=(TerminalBuffer &&) noexcept;

  void Resize(std::size_t columns, std::size_t rows);
  void Feed(std::string_view bytes);
  void Clear();
  std::string TakeOutboundData();
  std::string EncodeCharacter(char32_t codepoint,
                              std::uint8_t modifiers = NoModifier);
  std::string EncodeKey(Key key, std::uint8_t modifiers = NoModifier);
  std::string EncodePaste(std::string_view text);

  std::size_t Columns() const;
  std::size_t Rows() const;
  std::size_t TotalLineCount() const;
  const Line &LineAt(std::size_t index) const;
  std::string PlainText() const;

  std::size_t CursorColumn() const;
  std::size_t CursorRow() const;
  bool CursorVisible() const;
  bool AlternateScreen() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cyxwiz
