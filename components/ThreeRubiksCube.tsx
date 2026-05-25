"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Move } from "@/lib/types";

type ThreeRubiksCubeProps = {
  activeMove?: Move;
  resetKey: number;
  autoRotate: boolean;
};

type Cubie = THREE.Group & {
  userData: {
    homeX: number;
    homeY: number;
    homeZ: number;
    x: number;
    y: number;
    z: number;
  };
};

const spacing = 1.04;
const cubeletSize = 1;
const stickerOffset = cubeletSize / 2 + 0.012;
const stickerSize = 0.91;

const faceColors = {
  front: 0x18dce8,
  back: 0x35c96f,
  right: 0xff8a00,
  left: 0xe43f2f,
  top: 0xf8f8f0,
  bottom: 0xf6d84a
};

const moveConfig = {
  U: { axis: "y", layer: 1, direction: -1 },
  "U'": { axis: "y", layer: 1, direction: 1 },
  D: { axis: "y", layer: -1, direction: 1 },
  "D'": { axis: "y", layer: -1, direction: -1 },
  L: { axis: "x", layer: -1, direction: 1 },
  "L'": { axis: "x", layer: -1, direction: -1 },
  R: { axis: "x", layer: 1, direction: -1 },
  "R'": { axis: "x", layer: 1, direction: 1 },
  F: { axis: "z", layer: 1, direction: -1 },
  "F'": { axis: "z", layer: 1, direction: 1 },
  B: { axis: "z", layer: -1, direction: 1 },
  "B'": { axis: "z", layer: -1, direction: -1 }
} satisfies Record<Move, { axis: "x" | "y" | "z"; layer: number; direction: number }>;

function makeSticker(material: THREE.Material, face: keyof typeof faceColors) {
  const geometry = new THREE.PlaneGeometry(stickerSize, stickerSize);
  const sticker = new THREE.Mesh(geometry, material);

  if (face === "front") {
    sticker.position.z = stickerOffset;
  }

  if (face === "back") {
    sticker.position.z = -stickerOffset;
    sticker.rotation.y = Math.PI;
  }

  if (face === "right") {
    sticker.position.x = stickerOffset;
    sticker.rotation.y = Math.PI / 2;
  }

  if (face === "left") {
    sticker.position.x = -stickerOffset;
    sticker.rotation.y = -Math.PI / 2;
  }

  if (face === "top") {
    sticker.position.y = stickerOffset;
    sticker.rotation.x = -Math.PI / 2;
  }

  if (face === "bottom") {
    sticker.position.y = -stickerOffset;
    sticker.rotation.x = Math.PI / 2;
  }

  return sticker;
}

function createCubie(
  x: number,
  y: number,
  z: number,
  bodyGeometry: THREE.BoxGeometry,
  bodyMaterial: THREE.Material,
  stickerMaterials: Record<keyof typeof faceColors, THREE.Material>
) {
  const cubie = new THREE.Group() as Cubie;
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);

  cubie.userData = { homeX: x, homeY: y, homeZ: z, x, y, z };
  cubie.position.set(x * spacing, y * spacing, z * spacing);
  cubie.add(body);

  if (z === 1) cubie.add(makeSticker(stickerMaterials.front, "front"));
  if (z === -1) cubie.add(makeSticker(stickerMaterials.back, "back"));
  if (x === 1) cubie.add(makeSticker(stickerMaterials.right, "right"));
  if (x === -1) cubie.add(makeSticker(stickerMaterials.left, "left"));
  if (y === 1) cubie.add(makeSticker(stickerMaterials.top, "top"));
  if (y === -1) cubie.add(makeSticker(stickerMaterials.bottom, "bottom"));

  return cubie;
}

function easeInOutCubic(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

export function ThreeRubiksCube({ activeMove, resetKey, autoRotate }: ThreeRubiksCubeProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const rootRef = useRef<THREE.Group>();
  const cubiesRef = useRef<Cubie[]>([]);
  const animatingRef = useRef(false);
  const lastMoveRef = useRef<Move>();
  const draggingRef = useRef(false);
  const pointerRef = useRef({ x: 0, y: 0 });
  const autoRotateRef = useRef(autoRotate);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    const host = hostRef.current;

    if (!host) {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, host.clientWidth / host.clientHeight, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    const root = new THREE.Group();
    const cubies: Cubie[] = [];

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, host.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.cursor = "grab";
    host.appendChild(renderer.domElement);

    camera.position.set(4.9, 4.2, 6.1);
    camera.lookAt(0, 0, 0);
    root.rotation.set(-0.04, -0.16, 0);
    scene.add(root);

    scene.add(new THREE.AmbientLight(0xffffff, 1.22));

    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(4, 5, 6);
    scene.add(keyLight);

    const cyanLight = new THREE.PointLight(0x22d3ee, 8, 10);
    cyanLight.position.set(-3, 1.8, 4);
    scene.add(cyanLight);

    const bodyGeometry = new THREE.BoxGeometry(cubeletSize, cubeletSize, cubeletSize, 2, 2, 2);
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0x050505,
      roughness: 0.58,
      metalness: 0.08
    });
    const createStickerMaterial = (color: number, emissiveIntensity = 0.045) =>
      new THREE.MeshStandardMaterial({
        color,
        roughness: 0.36,
        metalness: 0.02,
        emissive: color,
        emissiveIntensity
      });

    const stickerMaterials: Record<keyof typeof faceColors, THREE.Material> = {
      front: createStickerMaterial(faceColors.front, 0.1),
      back: createStickerMaterial(faceColors.back),
      right: createStickerMaterial(faceColors.right),
      left: createStickerMaterial(faceColors.left),
      top: createStickerMaterial(faceColors.top),
      bottom: createStickerMaterial(faceColors.bottom)
    };

    for (let x = -1; x <= 1; x += 1) {
      for (let y = -1; y <= 1; y += 1) {
        for (let z = -1; z <= 1; z += 1) {
          const cubie = createCubie(x, y, z, bodyGeometry, bodyMaterial, stickerMaterials);
          cubies.push(cubie);
          root.add(cubie);
        }
      }
    }

    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    rootRef.current = root;
    cubiesRef.current = cubies;

    const resize = () => {
      if (!hostRef.current || !rendererRef.current || !cameraRef.current) {
        return;
      }

      const width = hostRef.current.clientWidth;
      const height = hostRef.current.clientHeight;
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };

    let frame = 0;
    const render = () => {
      frame = window.requestAnimationFrame(render);
      if (autoRotateRef.current && !draggingRef.current && !animatingRef.current) {
        root.rotation.y += 0.0022;
      }
      renderer.render(scene, camera);
    };

    const handlePointerDown = (event: PointerEvent) => {
      draggingRef.current = true;
      pointerRef.current = { x: event.clientX, y: event.clientY };
      renderer.domElement.style.cursor = "grabbing";
      renderer.domElement.setPointerCapture(event.pointerId);
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!draggingRef.current) {
        return;
      }

      const dx = event.clientX - pointerRef.current.x;
      const dy = event.clientY - pointerRef.current.y;
      pointerRef.current = { x: event.clientX, y: event.clientY };

      root.rotation.y += dx * 0.007;
      root.rotation.x += dy * 0.007;
      root.rotation.x = THREE.MathUtils.clamp(root.rotation.x, -1.25, 1.25);
    };

    const handlePointerUp = (event: PointerEvent) => {
      draggingRef.current = false;
      renderer.domElement.style.cursor = "grab";
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
    };

    window.addEventListener("resize", resize);
    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);
    renderer.domElement.addEventListener("pointercancel", handlePointerUp);
    resize();
    render();

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp);
      renderer.domElement.removeEventListener("pointercancel", handlePointerUp);
      renderer.dispose();
      bodyGeometry.dispose();
      bodyMaterial.dispose();
      Object.values(stickerMaterials).forEach((material) => material.dispose());
      host.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    if (!activeMove || activeMove === lastMoveRef.current || animatingRef.current) {
      return;
    }

    const root = rootRef.current;

    if (!root) {
      return;
    }

    lastMoveRef.current = activeMove;
    animatingRef.current = true;

    const config = moveConfig[activeMove];
    const pivot = new THREE.Group();
    const selected = cubiesRef.current.filter((cubie) => Math.round(cubie.userData[config.axis]) === config.layer);
    const targetRotation = config.direction * (Math.PI / 2);
    const start = performance.now();
    const duration = 420;

    root.add(pivot);
    selected.forEach((cubie) => pivot.attach(cubie));

    const rotate = (time: number) => {
      const progress = Math.min((time - start) / duration, 1);
      const rotation = easeInOutCubic(progress) * targetRotation;
      pivot.rotation[config.axis] = rotation;

      if (progress < 1) {
        window.requestAnimationFrame(rotate);
        return;
      }

      pivot.rotation[config.axis] = targetRotation;
      selected.forEach((cubie) => {
        root.attach(cubie);
        cubie.position.x = Math.round(cubie.position.x / spacing) * spacing;
        cubie.position.y = Math.round(cubie.position.y / spacing) * spacing;
        cubie.position.z = Math.round(cubie.position.z / spacing) * spacing;
        cubie.userData.x = Math.round(cubie.position.x / spacing);
        cubie.userData.y = Math.round(cubie.position.y / spacing);
        cubie.userData.z = Math.round(cubie.position.z / spacing);
      });
      root.remove(pivot);
      animatingRef.current = false;
    };

    window.requestAnimationFrame(rotate);
  }, [activeMove]);

  useEffect(() => {
    if (!activeMove) {
      lastMoveRef.current = undefined;
    }
  }, [activeMove]);

  useEffect(() => {
    const root = rootRef.current;

    if (!root) {
      return;
    }

    root.rotation.set(-0.04, -0.16, 0);
    cubiesRef.current.forEach((cubie) => {
      const { homeX, homeY, homeZ } = cubie.userData;
      cubie.position.set(homeX * spacing, homeY * spacing, homeZ * spacing);
      cubie.rotation.set(0, 0, 0);
      cubie.quaternion.identity();
      cubie.userData.x = homeX;
      cubie.userData.y = homeY;
      cubie.userData.z = homeZ;
    });
    animatingRef.current = false;
    lastMoveRef.current = undefined;
  }, [resetKey]);

  return <div ref={hostRef} className="h-full w-full" />;
}
