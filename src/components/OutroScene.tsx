import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

export const OutroScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const logoScale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const textOpacity = interpolate(frame, [30, 60], [0, 1], {
    extrapolateRight: "clamp",
  });

  const linksOpacity = interpolate(frame, [90, 120], [0, 1], {
    extrapolateRight: "clamp",
  });

  const glowIntensity = interpolate(
    frame,
    [0, 150, 300],
    [0, 0.8, 0.5],
    { extrapolateRight: "clamp" }
  );

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #0f0f23 100%)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {/* Background glow */}
      <div
        style={{
          position: "absolute",
          width: 800,
          height: 800,
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(139, 92, 246, ${glowIntensity * 0.3}) 0%, transparent 70%)`,
          filter: "blur(80px)",
        }}
      />

      {/* Check mark */}
      <div
        style={{
          transform: `scale(${logoScale})`,
          fontSize: 100,
          marginBottom: 30,
        }}
      >
        <div
          style={{
            width: 140,
            height: 140,
            borderRadius: "50%",
            background: "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 80,
            color: "white",
            boxShadow: "0 0 60px rgba(34, 197, 94, 0.4)",
          }}
        >
          ?
        </div>
      </div>

      {/* Title */}
      <div
        style={{
          opacity: textOpacity,
          fontSize: 56,
          fontWeight: 800,
          fontFamily: "system-ui, sans-serif",
          background: "linear-gradient(135deg, #22c55e 0%, #3b82f6 50%, #8b5cf6 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          textAlign: "center",
          marginBottom: 20,
        }}
      >
        Python Integration Complete
      </div>

      {/* Subtitle */}
      <div
        style={{
          opacity: textOpacity,
          fontSize: 28,
          fontFamily: "system-ui, sans-serif",
          color: "#94a3b8",
          textAlign: "center",
          maxWidth: 800,
          lineHeight: 1.6,
        }}
      >
        The CyxWiz Engine now features a robust, isolated Python environment
        <br />
        for reliable ML workflow execution
      </div>

      {/* Key points */}
      <div
        style={{
          opacity: linksOpacity,
          display: "flex",
          gap: 40,
          marginTop: 50,
        }}
      >
        {[
          { icon: "???", text: "Project Isolation" },
          { icon: "???", text: "Auto Venv" },
          { icon: "???", text: "Clear Diagnostics" },
        ].map((item, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "12px 24px",
              background: "rgba(59, 130, 246, 0.1)",
              border: "1px solid rgba(59, 130, 246, 0.3)",
              borderRadius: 30,
            }}
          >
            <span style={{ fontSize: 24 }}>{item.icon}</span>
            <span
              style={{
                fontSize: 18,
                fontFamily: "system-ui, sans-serif",
                color: "#e2e8f0",
              }}
            >
              {item.text}
            </span>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          opacity: linksOpacity,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 12,
        }}
      >
        <div
          style={{
            fontSize: 20,
            fontFamily: "system-ui, sans-serif",
            color: "#64748b",
          }}
        >
          CyxWiz Engine - Decentralized ML Compute Platform
        </div>
        <div
          style={{
            fontSize: 16,
            fontFamily: "monospace",
            color: "#8b5cf6",
          }}
        >
          docs/python_interpreter_design.md
        </div>
      </div>
    </AbsoluteFill>
  );
};
