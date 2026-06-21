import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
  Sequence,
} from "remotion";

interface StepProps {
  stepNumber: number;
  title: string;
  description: string;
  code?: string;
  delay: number;
}

const Step: React.FC<StepProps> = ({ stepNumber, title, description, code, delay }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame: frame - delay,
    fps,
    config: { damping: 15, stiffness: 100 },
  });

  const opacity = interpolate(frame - delay, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  if (frame < delay) return null;

  return (
    <div
      style={{
        transform: `scale(${scale})`,
        opacity,
        display: "flex",
        gap: 24,
        marginBottom: 30,
      }}
    >
      {/* Step number */}
      <div
        style={{
          width: 60,
          height: 60,
          borderRadius: "50%",
          background: "linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 28,
          fontWeight: 800,
          fontFamily: "system-ui, sans-serif",
          color: "white",
          flexShrink: 0,
        }}
      >
        {stepNumber}
      </div>

      {/* Content */}
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 24,
            fontWeight: 700,
            fontFamily: "system-ui, sans-serif",
            color: "#e2e8f0",
            marginBottom: 8,
          }}
        >
          {title}
        </div>
        <div
          style={{
            fontSize: 18,
            fontFamily: "system-ui, sans-serif",
            color: "#94a3b8",
            marginBottom: code ? 12 : 0,
          }}
        >
          {description}
        </div>
        {code && (
          <div
            style={{
              background: "rgba(0, 0, 0, 0.5)",
              borderRadius: 8,
              padding: "12px 16px",
              fontFamily: "monospace",
              fontSize: 14,
              color: "#a5f3fc",
              border: "1px solid rgba(59, 130, 246, 0.3)",
            }}
          >
            {code}
          </div>
        )}
      </div>
    </div>
  );
};

export const SetupTutorialScene: React.FC = () => {
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateRight: "clamp",
  });

  const steps = [
    {
      stepNumber: 1,
      title: "Install Python 3.12",
      description: "Download from python.org (NOT Windows Store). Check 'Add to PATH' during installation.",
      code: "https://www.python.org/downloads/release/python-3128/",
      delay: 60,
    },
    {
      stepNumber: 2,
      title: "Launch CyxWiz Engine",
      description: "On first run, the engine auto-detects Python 3.12 and creates engine_config.json",
      delay: 180,
    },
    {
      stepNumber: 3,
      title: "Configure in Preferences",
      description: "Go to Preferences > Python. Use 'Auto-Detect' or 'Browse' to select Python 3.12.x",
      delay: 300,
    },
    {
      stepNumber: 4,
      title: "Create a New Project",
      description: "A virtual environment is automatically created in <project>/python/",
      code: "my_project/python/Scripts/python.exe",
      delay: 420,
    },
    {
      stepNumber: 5,
      title: "Install Packages",
      description: "Use pip in the project venv to install ML packages",
      code: "my_project/python/Scripts/pip install numpy torch",
      delay: 540,
    },
    {
      stepNumber: 6,
      title: "Use Python Console",
      description: "Open the Python console to run scripts with full environment isolation",
      delay: 660,
    },
  ];

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(180deg, #0a0a1a 0%, #0a1a2a 100%)",
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
            color: "#8b5cf6",
            margin: 0,
            display: "flex",
            alignItems: "center",
            gap: 20,
          }}
        >
          <span style={{ fontSize: 48 }}>???</span>
          Setup Guide
        </h1>
        <p
          style={{
            fontSize: 24,
            fontFamily: "system-ui, sans-serif",
            color: "#94a3b8",
            marginTop: 10,
          }}
        >
          How to configure Python interpreter in CyxWiz Engine
        </p>
      </div>

      {/* Steps */}
      <div style={{ display: "flex", gap: 60 }}>
        <div style={{ flex: 1 }}>
          {steps.slice(0, 3).map((step) => (
            <Step key={step.stepNumber} {...step} />
          ))}
        </div>
        <div style={{ flex: 1 }}>
          {steps.slice(3).map((step) => (
            <Step key={step.stepNumber} {...step} />
          ))}
        </div>
      </div>

      {/* Config file preview */}
      <Sequence from={720}>
        <div
          style={{
            position: "absolute",
            bottom: 40,
            left: 60,
            right: 60,
            background: "rgba(0, 0, 0, 0.8)",
            borderRadius: 12,
            padding: 20,
            opacity: interpolate(frame, [720, 750], [0, 1], { extrapolateRight: "clamp" }),
            border: "1px solid rgba(139, 92, 246, 0.3)",
          }}
        >
          <div style={{ color: "#8b5cf6", fontSize: 14, marginBottom: 8, fontFamily: "monospace" }}>
            engine_config.json
          </div>
          <pre
            style={{
              fontFamily: "monospace",
              fontSize: 14,
              color: "#94a3b8",
              margin: 0,
            }}
          >
{`{
  "system_python_path": "C:\\\\Python312\\\\python.exe",
  "auto_create_venv": true,
  "default_venv_packages": ["numpy", "torch", "scikit-learn"]
}`}
          </pre>
        </div>
      </Sequence>
    </AbsoluteFill>
  );
};
