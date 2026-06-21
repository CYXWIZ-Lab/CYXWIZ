import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  Sequence,
} from "remotion";

interface FixItemProps {
  before: string;
  after: string;
  title: string;
  delay: number;
}

const FixItem: React.FC<FixItemProps> = ({ before, after, title, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const slideIn = spring({
    frame: frame - delay,
    fps,
    config: { damping: 15, stiffness: 100 },
  });

  const opacity = interpolate(frame - delay, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const arrowOpacity = interpolate(frame - delay, [30, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const afterOpacity = interpolate(frame - delay, [50, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  if (frame < delay) return null;

  return (
    <div
      style={{
        transform: `translateX(${(1 - slideIn) * -50}px)`,
        opacity,
        marginBottom: 30,
      }}
    >
      <div
        style={{
          fontSize: 20,
          fontWeight: 700,
          fontFamily: "system-ui, sans-serif",
          color: "#e2e8f0",
          marginBottom: 12,
        }}
      >
        {title}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        {/* Before */}
        <div
          style={{
            flex: 1,
            background: "rgba(239, 68, 68, 0.1)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: 8,
            padding: "12px 16px",
            fontFamily: "monospace",
            fontSize: 14,
            color: "#f87171",
          }}
        >
          <div style={{ color: "#64748b", fontSize: 12, marginBottom: 4 }}>BEFORE</div>
          {before}
        </div>

        {/* Arrow */}
        <div
          style={{
            opacity: arrowOpacity,
            fontSize: 32,
            color: "#3b82f6",
          }}
        >
          ???
        </div>

        {/* After */}
        <div
          style={{
            flex: 1,
            opacity: afterOpacity,
            background: "rgba(34, 197, 94, 0.1)",
            border: "1px solid rgba(34, 197, 94, 0.3)",
            borderRadius: 8,
            padding: "12px 16px",
            fontFamily: "monospace",
            fontSize: 14,
            color: "#4ade80",
          }}
        >
          <div style={{ color: "#64748b", fontSize: 12, marginBottom: 4 }}>AFTER</div>
          {after}
        </div>
      </div>
    </div>
  );
};

export const KeyFixesScene: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  const fixes = [
    {
      title: "1. Base Prefix Detection (Critical Fix)",
      before: "sys.base_prefix = build/bin/Release",
      after: "Read from pyvenv.cfg -> C:\\Python312",
      delay: 45,
    },
    {
      title: "2. Race Condition Protection",
      before: "Multiple venv creations -> Permission denied",
      after: "Check existing folder before creating",
      delay: 135,
    },
    {
      title: "3. Environment Variable Isolation",
      before: "Inherit system PYTHONHOME/PYTHONPATH",
      after: "Clear all + PYTHONNOUSERSITE=1",
      delay: 225,
    },
    {
      title: "4. Venv Creation Mode",
      before: "python -I -m venv (isolated mode)",
      after: "python -m venv (full stdlib access)",
      delay: 315,
    },
    {
      title: "5. Initialization Timing",
      before: "Eager init at app launch",
      after: "Lazy init on first Python use",
      delay: 405,
    },
  ];

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(180deg, #0a0a1a 0%, #0a0a2a 100%)",
        padding: 60,
      }}
    >
      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          marginBottom: 40,
        }}
      >
        <h1
          style={{
            fontSize: 56,
            fontWeight: 800,
            fontFamily: "system-ui, sans-serif",
            color: "#3b82f6",
            margin: 0,
            display: "flex",
            alignItems: "center",
            gap: 20,
          }}
        >
          <span style={{ fontSize: 48 }}>???</span>
          Key Fixes (2026-03-19)
        </h1>
      </div>

      {/* Fixes List */}
      <div style={{ maxWidth: 1000 }}>
        {fixes.map((fix, index) => (
          <FixItem key={index} {...fix} />
        ))}
      </div>

      {/* Summary badge */}
      <Sequence from={480}>
        <div
          style={{
            position: "absolute",
            bottom: 60,
            right: 60,
            background: "linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)",
            border: "1px solid rgba(59, 130, 246, 0.4)",
            borderRadius: 16,
            padding: "20px 30px",
            opacity: interpolate(frame, [480, 510], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          <div style={{ color: "#94a3b8", fontSize: 14, marginBottom: 8 }}>RESULT</div>
          <div style={{ color: "#22c55e", fontSize: 24, fontWeight: 700, fontFamily: "system-ui, sans-serif" }}>
            ? Stable Python Integration
          </div>
          <div style={{ color: "#64748b", fontSize: 16, marginTop: 8 }}>
            No more "ModuleNotFoundError: threading"
          </div>
        </div>
      </Sequence>
    </AbsoluteFill>
  );
};
