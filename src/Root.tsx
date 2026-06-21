import { Composition } from "remotion";
import { PythonInterpreterVideo } from "./Video";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="PythonInterpreterVideo"
        component={PythonInterpreterVideo}
        durationInFrames={30 * 90} // 90 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
