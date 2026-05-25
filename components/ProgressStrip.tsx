"use client";

import type { Move } from "@/lib/types";

type ProgressStripProps = {
  moves: Move[];
  currentIndex: number;
};

export function ProgressStrip({ moves, currentIndex }: ProgressStripProps) {
  if (moves.length === 0) {
    return null;
  }

  return (
    <aside className="fixed left-5 top-1/2 z-20 hidden max-h-[76vh] w-[132px] -translate-y-1/2 flex-col rounded-[28px] border border-white/10 bg-white/[0.06] p-3 shadow-[0_20px_70px_rgba(0,0,0,0.45)] backdrop-blur md:flex">
      <div className="mb-3 px-2 text-[10px] font-bold uppercase tracking-[0.16em] text-white/45">
        <span>Sequence</span>
      </div>
      <div className="grid grid-cols-2 gap-2 overflow-y-auto pr-1">
        {moves.map((move, index) => (
          <button
            key={`${move}-${index}`}
            type="button"
            className={`flex h-10 w-10 items-center justify-center rounded-full border text-xs font-bold transition ${
              index < currentIndex
                ? "border-cyan-300/25 bg-cyan-300/10 text-cyan-100/55"
                : index === currentIndex
                  ? "border-cyan-300 bg-cyan-300 text-black shadow-[0_0_22px_rgba(34,211,238,0.72)]"
                  : "border-white/10 bg-black/70 text-white/75"
            }`}
          >
            {move}
          </button>
        ))}
      </div>
    </aside>
  );
}
