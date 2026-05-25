"use client";

import type { Move } from "@/lib/types";

type MoveTesterProps = {
  busy: boolean;
  onMove: (move: Move) => void;
};

const moves: Move[] = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"];

export function MoveTester({ busy, onMove }: MoveTesterProps) {
  return (
    <aside className="fixed right-5 top-1/2 z-20 hidden -translate-y-1/2 flex-col gap-2 rounded-[28px] border border-white/10 bg-white/[0.06] p-3 shadow-[0_20px_70px_rgba(0,0,0,0.45)] backdrop-blur md:flex">
      <div className="mb-1 px-2 text-center text-[10px] font-bold uppercase tracking-[0.18em] text-white/45">Faces</div>
      <div className="grid grid-cols-2 gap-2">
        {moves.map((move) => (
          <button
            key={move}
            type="button"
            disabled={busy}
            onClick={() => onMove(move)}
            className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-black/70 text-xs font-bold text-white transition hover:border-cyan-300 hover:bg-cyan-300 hover:text-black disabled:cursor-not-allowed disabled:opacity-40"
          >
            {move}
          </button>
        ))}
      </div>
    </aside>
  );
}
