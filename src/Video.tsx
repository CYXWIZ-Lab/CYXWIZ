import { AbsoluteFill, Sequence } from "remotion";
import { IntroScene } from "./components/IntroScene";
import { OldDesignScene } from "./components/OldDesignScene";
import { NewDesignScene } from "./components/NewDesignScene";
import { KeyFixesScene } from "./components/KeyFixesScene";
import { SetupTutorialScene } from "./components/SetupTutorialScene";
import { OutroScene } from "./components/OutroScene";

// Frame timing (30fps)
const INTRO_START = 0;
const INTRO_DURATION = 150; // 5 seconds

const OLD_DESIGN_START = INTRO_DURATION;
const OLD_DESIGN_DURATION = 450; // 15 seconds

const NEW_DESIGN_START = OLD_DESIGN_START + OLD_DESIGN_DURATION;
const NEW_DESIGN_DURATION = 450; // 15 seconds

const KEY_FIXES_START = NEW_DESIGN_START + NEW_DESIGN_DURATION;
const KEY_FIXES_DURATION = 540; // 18 seconds

const SETUP_START = KEY_FIXES_START + KEY_FIXES_DURATION;
const SETUP_DURATION = 810; // 27 seconds

const OUTRO_START = SETUP_START + SETUP_DURATION;
const OUTRO_DURATION = 300; // 10 seconds

export const PythonInterpreterVideo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a0f" }}>
      <Sequence from={INTRO_START} durationInFrames={INTRO_DURATION}>
        <IntroScene />
      </Sequence>

      <Sequence from={OLD_DESIGN_START} durationInFrames={OLD_DESIGN_DURATION}>
        <OldDesignScene />
      </Sequence>

      <Sequence from={NEW_DESIGN_START} durationInFrames={NEW_DESIGN_DURATION}>
        <NewDesignScene />
      </Sequence>

      <Sequence from={KEY_FIXES_START} durationInFrames={KEY_FIXES_DURATION}>
        <KeyFixesScene />
      </Sequence>

      <Sequence from={SETUP_START} durationInFrames={SETUP_DURATION}>
        <SetupTutorialScene />
      </Sequence>

      <Sequence from={OUTRO_START} durationInFrames={OUTRO_DURATION}>
        <OutroScene />
      </Sequence>
    </AbsoluteFill>
  );
};
