"use client";

import { ThreeRubiksCube } from "./ThreeRubiksCube";
import type { Move } from "@/lib/types";

type CubeStageProps = {
  activeMove?: Move;
  resetKey: number;
  autoRotate: boolean;
};

export function CubeStage({ activeMove, resetKey, autoRotate }: CubeStageProps) {
  return (
    <section className="flex min-h-[54vh] w-full flex-1 items-center justify-center">
      <div className="h-[min(66vh,620px)] w-[min(82vw,760px)]" aria-label="Rubik's Cube WebGL viewport">
        <ThreeRubiksCube activeMove={activeMove} resetKey={resetKey} autoRotate={autoRotate} />
      </div>
    </section>
  );
}
