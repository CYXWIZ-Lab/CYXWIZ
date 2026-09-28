#pragma once

#include <string>

namespace scripting {

class IScriptOutputSink {
public:
  virtual ~IScriptOutputSink() = default;

  virtual void AppendScriptOutput(const std::string &source,
                                  const std::string &text,
                                  bool is_error = false) = 0;

  // A script run named `source` finished after its output. Sinks that group
  // a run's output show its outcome and duration; the default ignores it.
  virtual void EndScriptOutput(const std::string & /*source*/, bool /*success*/,
                               bool /*cancelled*/, double /*seconds*/) {}
};

} // namespace scripting
