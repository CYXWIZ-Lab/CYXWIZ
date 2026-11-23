#pragma once

// How fast the Engine draws its frames right now. It slows down when nothing
// needs drawing: after a second without input, when another program is in
// front, and when its window is minimised. Training, and a plugin with
// something moving (a running simulation), keep it at full speed.
//
// The application decides the pace each frame; Tools > Diagnostics > Log
// Frame-rate Changes writes one line per change, and Monitor > Frames shows
// the current one.

#include <atomic>

namespace cyxwiz {

enum class FramePace { Full, NotInFront, Idle, Minimised };

// The pace for a frame. A minimised window draws nothing whatever else is
// true; an idle Engine waits longest of the visible states; training (or a
// running simulation) keeps a visible, focused Engine at full speed.
constexpr FramePace PaceOf(bool minimised, bool idle, bool keep_full_rate, bool focused) {
    if (minimised) return FramePace::Minimised;
    if (idle && !keep_full_rate) return FramePace::Idle;
    if (!focused) return FramePace::NotInFront;
    return FramePace::Full;
}

// "Full speed", "30 frames a second (another window is in front)", ...
constexpr const char* PaceText(FramePace pace) {
    switch (pace) {
        case FramePace::Full: return "Full speed";
        case FramePace::NotInFront: return "30 frames a second (another window is in front)";
        case FramePace::Idle: return "10 frames a second (idle: no input for a second)";
        case FramePace::Minimised: return "Not drawing (the window is minimised)";
    }
    return "Full speed";
}

// The pace of the last frame, for readers on the interface thread or others.
inline std::atomic<FramePace>& CurrentFramePaceSlot() {
    static std::atomic<FramePace> pace{FramePace::Full};
    return pace;
}
inline FramePace CurrentFramePace() { return CurrentFramePaceSlot().load(std::memory_order_relaxed); }

}  // namespace cyxwiz
