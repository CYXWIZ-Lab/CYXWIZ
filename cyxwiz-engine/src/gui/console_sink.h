#pragma once

// Compatibility include for out-of-tree users. Runtime logging is now owned
// by the GUI-independent core sink and store.
#include "../core/runtime_log_sink.h"

namespace gui {

template <typename Mutex>
using ConsoleSink = cyxwiz::RuntimeLogSink<Mutex>;

using ConsoleSinkMt = cyxwiz::RuntimeLogSinkMt;
using ConsoleSinkSt = cyxwiz::RuntimeLogSinkSt;

} // namespace gui
