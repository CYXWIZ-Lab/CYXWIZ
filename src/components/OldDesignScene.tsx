import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  Sequence,
} from "remotion";

interface ProblemCardProps {
  title: string;
  description: string;
  icon: string;
  delay: number;
}

const ProblemCard: React.FC<ProblemCardProps> = ({ title, description, icon, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame: frame - delay,
    fps,
    config: { damping: 15, stiffness: 120 },
  });

  const opacity = interpolate(frame - delay, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  if (frame < delay) return null;

  return (
    <div
      style={{
        transform: `scale(${scale})`,
        opacity,
        background: "linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.1) 100%)",
        border: "1px solid rgba(239, 68, 68, 0.3)",
        borderRadius: 16,
        padding: "24px 28px",
        width: 380,
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <span style={{ fontSize: 28 }}>{icon}</span>
        <span
          style={{
            fontSize: 20,
            fontWeight: 700,
            fontFamily: "system-ui, sans-serif",
            color: "#f87171",
          }}
        >
          {title}
        </span>
      </div>
      <p
        style={{
          fontSize: 16,
          fontFamily: "system-ui, sans-serif",
          color: "#cbd5e1",
          lineHeight: 1.5,
          margin: 0,
        }}
      >
        {description}
      </p>
    </div>
  );
};

export const OldDesignScene: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  const titleY = interpolate(frame, [0, 30], [-20, 0], {
    extrapolateRight: "clamp",
  });

  const problems = [
    {
      title: "Early Initialization",
      description: "Python initialized before engine config was applied, locking in the wrong interpreter",
      icon: "???",
      delay: 45,
    },
    {
      title: "No Environment Isolation",
      description: "PYTHONHOME and PYTHONPATH not cleared, causing package leakage from system Python",
      icon: "???",
      delay: 90,
    },
    {
      title: "Wrong sys.path",
      description: "Packages resolved from multiple environments, causing version conflicts",
      icon: "???",
      delay: 135,
    },
    {
      title: "Venv Path Bugs",
      description: "Windows venv detection failed - looked for Lib/ in Scripts/ directory",
      icon: "???",
      delay: 180,
    },
    {
      title: "Base Prefix Detection",
      description: "pybind11 set sys.base_prefix incorrectly to engine build directory",
      icon: "???",
      delay: 225,
    },
    {
      title: "Race Conditions",
      description: "Multiple venv creation attempts when opening projects quickly",
      icon: "???",
      delay: 270,
    },
  ];

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(180deg, #0a0a1a 0%, #1a0a0a 100%)",
        padding: 60,
      }}
    >
      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          transform: `translateY(${titleY}px)`,
          marginBottom: 50,
        }}
      >
        <h1
          style={{
            fontSize: 56,
            fontWeight: 800,
            fontFamily: "system-ui, sans-serif",
            color: "#ef4444",
            margin: 0,
            display: "flex",
            alignItems: "center",
            gap: 20,
          }}
        >
          <span style={{ fontSize: 48 }}>???</span>
          The Old Design - Problems
        </h1>
        <p
          style={{
            fontSize: 24,
            fontFamily: "system-ui, sans-serif",
            color: "#94a3b8",
            marginTop: 10,
          }}
        >
          Critical issues that plagued the previous Python interpreter integration
        </p>
      </div>

      {/* Problem Cards Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 24,
          maxWidth: 1200,
        }}
      >
        {problems.map((problem, index) => (
          <ProblemCard key={index} {...problem} />
        ))}
      </div>

      {/* Error log animation */}
      <Sequence from={320}>
        <div
          style={{
            position: "absolute",
            bottom: 60,
            left: 60,
            right: 60,
            background: "rgba(0, 0, 0, 0.8)",
            borderRadius: 12,
            padding: 20,
            fontFamily: "monospace",
            fontSize: 14,
            color: "#ef4444",
            border: "1px solid rgba(239, 68, 68, 0.3)",
          }}
        >
          <div style={{ color: "#64748b", marginBottom: 8 }}># Typical error log</div>
          <div>[ERROR] ModuleNotFoundError: No module named 'threading'</div>
          <div style={{ color: "#fbbf24" }}>[WARN] sys.base_prefix = D:\CyxWiz\build\bin\Release (WRONG!)</div>
          <div>[ERROR] SRE module mismatch - mixing stdlib from different Python versions</div>
        </div>
      </Sequence>
    </AbsoluteFill>
  );
};
