import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

export const IntroScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animations
  const logoScale = spring({
    frame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const titleOpacity = interpolate(frame, [20, 50], [0, 1], {
    extrapolateRight: "clamp",
  });

  const titleY = interpolate(frame, [20, 50], [30, 0], {
    extrapolateRight: "clamp",
  });

  const subtitleOpacity = interpolate(frame, [50, 80], [0, 1], {
    extrapolateRight: "clamp",
  });

  const glowIntensity = interpolate(
    frame,
    [0, 75, 150],
    [0, 1, 0.6],
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
      {/* Background glow effect */}
      <div
        style={{
          position: "absolute",
          width: 600,
          height: 600,
          borderRadius: "50%",
          background: `radial-gradient(circle, rgba(59, 130, 246, ${glowIntensity * 0.3}) 0%, transparent 70%)`,
          filter: "blur(60px)",
        }}
      />

      {/* Python Logo */}
      <div
        style={{
          transform: `scale(${logoScale})`,
          fontSize: 120,
          marginBottom: 30,
        }}
      >
        <svg width="120" height="120" viewBox="0 0 256 255">
          <defs>
            <linearGradient id="pythonBlue" x1="12.959%" y1="12.039%" x2="79.639%" y2="78.201%">
              <stop stopColor="#387EB8" offset="0%" />
              <stop stopColor="#366994" offset="100%" />
            </linearGradient>
            <linearGradient id="pythonYellow" x1="19.128%" y1="20.579%" x2="90.742%" y2="88.429%">
              <stop stopColor="#FFE052" offset="0%" />
              <stop stopColor="#FFC331" offset="100%" />
            </linearGradient>
          </defs>
          <path
            d="M126.916.072c-64.832 0-60.784 28.115-60.784 28.115l.072 29.128h61.868v8.745H41.631S.145 61.355.145 126.77c0 65.417 36.21 63.097 36.21 63.097h21.61v-30.356s-1.165-36.21 35.632-36.21h61.362s34.475.557 34.475-33.319V33.97S194.67.072 126.916.072zM92.802 19.66a11.12 11.12 0 0 1 11.13 11.13 11.12 11.12 0 0 1-11.13 11.13 11.12 11.12 0 0 1-11.13-11.13 11.12 11.12 0 0 1 11.13-11.13z"
            fill="url(#pythonBlue)"
          />
          <path
            d="M128.757 254.126c64.832 0 60.784-28.115 60.784-28.115l-.072-29.127H127.6v-8.745h86.441s41.486 4.705 41.486-60.712c0-65.416-36.21-63.096-36.21-63.096h-21.61v30.355s1.165 36.21-35.632 36.21h-61.362s-34.475-.557-34.475 33.32v56.013s-5.235 33.897 62.518 33.897zm34.114-19.586a11.12 11.12 0 0 1-11.13-11.13 11.12 11.12 0 0 1 11.13-11.131 11.12 11.12 0 0 1 11.13 11.13 11.12 11.12 0 0 1-11.13 11.13z"
            fill="url(#pythonYellow)"
          />
        </svg>
      </div>

      {/* CyxWiz Logo Text */}
      <div
        style={{
          opacity: titleOpacity,
          transform: `translateY(${titleY}px)`,
          fontSize: 72,
          fontWeight: 800,
          fontFamily: "system-ui, -apple-system, sans-serif",
          background: "linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: "-2px",
        }}
      >
        CyxWiz Engine
      </div>

      {/* Subtitle */}
      <div
        style={{
          opacity: subtitleOpacity,
          marginTop: 20,
          fontSize: 36,
          fontWeight: 500,
          fontFamily: "system-ui, -apple-system, sans-serif",
          color: "#94a3b8",
        }}
      >
        Python Interpreter Redesign
      </div>

      {/* Version badge */}
      <div
        style={{
          opacity: subtitleOpacity,
          marginTop: 30,
          padding: "8px 20px",
          borderRadius: 20,
          background: "rgba(59, 130, 246, 0.2)",
          border: "1px solid rgba(59, 130, 246, 0.4)",
          fontSize: 18,
          fontFamily: "monospace",
          color: "#60a5fa",
        }}
      >
        v2026.03.19
      </div>
    </AbsoluteFill>
  );
};
