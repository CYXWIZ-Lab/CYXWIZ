#pragma once

#include <string>

namespace scripting {

class IScriptOutputSink {
public:
  virtual ~IScriptOutputSink() = default;

  virtual void AppendScriptOutput(const std::string &source,
                                  const std::string &text,
                                  bool is_error = false) = 0;
};

} // namespace scripting
