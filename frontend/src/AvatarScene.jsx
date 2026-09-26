import { Canvas } from "@react-three/fiber";
import { Environment } from "@react-three/drei";

import Avatar from "./Avatar";

function AvatarScene() {
  return (
    <Canvas
      camera={{
        position: [-0.8, 1.8, 7],
        fov: 12,
        filmOffset: -3.8,
      }}
      shadows
    >
      <ambientLight intensity={1.2} />

      <directionalLight
        position={[2, 4, 3]}
        intensity={2}
        castShadow
      />

      <Environment preset="studio" />

      <Avatar />
    </Canvas>
  );
}

export default AvatarScene;