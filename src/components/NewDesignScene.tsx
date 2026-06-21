import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: string;
  delay: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ title, description, icon, delay }) => {
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
        background: "linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.1) 100%)",
        border: "1px solid rgba(34, 197, 94, 0.3)",
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
            color: "#4ade80",
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

export const NewDesignScene: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  const titleY = interpolate(frame, [0, 30], [-20, 0], {
    extrapolateRight: "clamp",
  });

  const features = [
    {
      title: "Lazy Initialization",
      description: "Python initializes only on first use (console/script), not at app launch",
      icon: "???",
      delay: 45,
    },
    {
      title: "Environment Isolation",
      description: "Clears PYTHONHOME, PYTHONPATH, sets PYTHONNOUSERSITE=1 to prevent leakage",
      icon: "???",
      delay: 90,
    },
    {
      title: "Project Venvs",
      description: "Auto-creates project-specific virtual environment in <project>/python/",
      icon: "???",
      delay: 135,
    },
    {
      title: "pyvenv.cfg Detection",
      description: "Reads base Python from pyvenv.cfg instead of relying on runtime values",
      icon: "???",
      delay: 180,
    },
    {
      title: "Clear Diagnostics",
      description: "User-friendly logging with visual indicators (?, ?, ?) during init",
      icon: "???",
      delay: 225,
    },
    {
      title: "Python 3.12 Required",
      description: "Strict version requirement ensures ABI compatibility with pycyxwiz module",
      icon: "???",
      delay: 270,
    },
  ];

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(180deg, #0a0a1a 0%, #0a1a0a 100%)",
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
            color: "#22c55e",
            margin: 0,
            display: "flex",
            alignItems: "center",
            gap: 20,
          }}
        >
          <span style={{ fontSize: 48 }}>???</span>
          The New Design
        </h1>
        <p
          style={{
            fontSize: 24,
            fontFamily: "system-ui, sans-serif",
            color: "#94a3b8",
            marginTop: 10,
          }}
        >
          A complete redesign focused on reliability, isolation, and user experience
        </p>
      </div>

      {/* Feature Cards Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 24,
          maxWidth: 1200,
        }}
      >
        {features.map((feature, index) => (
          <FeatureCard key={index} {...feature} />
        ))}
      </div>

      {/* Architecture diagram hint */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          left: 60,
          right: 60,
          opacity: interpolate(frame, [320, 350], [0, 1], { extrapolateRight: "clamp" }),
          display: "flex",
          justifyContent: "center",
          gap: 40,
          alignItems: "center",
        }}
      >
        <div style={boxStyle}>engine_config.json</div>
        <Arrow />
        <div style={boxStyle}>python_env.json</div>
        <Arrow />
        <div style={{ ...boxStyle, background: "rgba(34, 197, 94, 0.2)", borderColor: "rgba(34, 197, 94, 0.5)" }}>
          Python 3.12 Venv
        </div>
      </div>
    </AbsoluteFill>
  );
};

const boxStyle: React.CSSProperties = {
  padding: "12px 24px",
  background: "rgba(59, 130, 246, 0.2)",
  border: "1px solid rgba(59, 130, 246, 0.4)",
  borderRadius: 8,
  fontFamily: "monospace",
  fontSize: 16,
  color: "#94a3b8",
};

const Arrow: React.FC = () => (
  <div style={{ color: "#64748b", fontSize: 24 }}>???</div>
);
