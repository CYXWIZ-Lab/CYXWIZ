// One-line Python tokenizer for the Script Editor colouring (TOFIX133 P0
// item 4). Comments and strings that span lines are handled by the editor;
// this finds the next token on a line. No ImGui, no editor types.
#pragma once

namespace cyxwiz::pytokens {

enum class Kind { String, Number, Identifier, Punctuation, Decorator };

// Finds the token that starts at `begin` (after any spaces). Returns false at
// the end of the input or when the character starts no token (the caller
// moves on by one character).
bool Next(const char* begin, const char* end, const char*& out_begin, const char*& out_end, Kind& kind);

}  // namespace cyxwiz::pytokens
