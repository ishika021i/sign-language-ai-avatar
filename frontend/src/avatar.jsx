import { useEffect, useState } from "react";
import { useGLTF } from "@react-three/drei";

function Avatar() {
  const { scene } = useGLTF("/models/isl-avatar.glb");
  const [boneNames, setBoneNames] = useState([]);

  useEffect(() => {
    const bones = [];

    scene.traverse((object) => {
      if (object.isBone) {
        bones.push(object.name);
      }
    });

    setBoneNames(bones);
  }, [scene]);

  return (
    <>
      <primitive
        object={scene}
        scale={1}
        position={[0, 0, 0]}
      />

      <div
        style={{
          position: "absolute",
          top: "10px",
          left: "10px",
          zIndex: 100,
          background: "rgba(0, 0, 0, 0.85)",
          color: "#00ffcc",
          padding: "15px",
          maxHeight: "300px",
          overflowY: "auto",
          fontSize: "12px",
          fontFamily: "monospace",
        }}
      >
        <strong>Avatar Bones</strong>

        {boneNames.map((name, index) => (
          <div key={index}>{name}</div>
        ))}
      </div>
    </>
  );
}

useGLTF.preload("/models/isl-avatar.glb");

export default Avatar;