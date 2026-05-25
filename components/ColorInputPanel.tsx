"use client";

import { useMemo, useState } from "react";
import type { CubeColor, CubeState } from "@/lib/types";
import { solvedCubeState } from "@/lib/cubeBridge";

const colors: CubeColor[] = ["white", "yellow", "red", "orange", "blue", "green"];

const swatches: Record<CubeColor, string> = {
  white: "bg-white",
  yellow: "bg-yellow-300",
  red: "bg-red-500",
  orange: "bg-orange-500",
  blue: "bg-blue-600",
  green: "bg-green-500"
};

const faces = ["Up", "Right", "Front", "Down", "Left", "Back"];

type ColorInputPanelProps = {
  open: boolean;
  onClose: () => void;
  onApply: (state: CubeState) => void;
};

export function ColorInputPanel({ open, onClose, onApply }: ColorInputPanelProps) {
  const [selectedColor, setSelectedColor] = useState<CubeColor>("white");
  const [stickers, setStickers] = useState<CubeState>([...solvedCubeState]);

  const counts = useMemo(() => {
    return colors.reduce<Record<CubeColor, number>>((acc, color) => {
      acc[color] = stickers.filter((sticker) => sticker === color).length;
      return acc;
    }, {} as Record<CubeColor, number>);
  }, [stickers]);

  const isValid = stickers.length === 54 && colors.every((color) => counts[color] === 9);

  if (!open) {
    return null;
  }

  const updateSticker = (index: number) => {
    setStickers((current) => current.map((color, stickerIndex) => (stickerIndex === index ? selectedColor : color)));
  };

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/80 px-4 py-6">
      <div className="max-h-full w-full max-w-3xl overflow-auto border border-cyan-300/20 bg-[#070707] p-4 shadow-[0_0_40px_rgba(34,211,238,0.12)]">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h2 className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-300">Add Colors</h2>
          <button className="border border-white/10 bg-black px-3 py-1 text-sm text-white/70" type="button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="mb-4 flex flex-wrap gap-2">
          {colors.map((color) => (
            <button
              key={color}
              type="button"
              onClick={() => setSelectedColor(color)}
              className={`h-9 w-9 rounded border ${swatches[color]} ${
                selectedColor === color ? "border-ink ring-2 ring-accent" : "border-line"
              }`}
              aria-label={color}
              title={color}
            />
          ))}
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {faces.map((face, faceIndex) => (
            <div key={face} className="border border-white/10 bg-[#0b0b0b] p-3">
              <div className="mb-2 text-sm font-medium text-white/75">{face}</div>
              <div className="grid aspect-square grid-cols-3 gap-1">
                {Array.from({ length: 9 }, (_, sticker) => {
                  const index = faceIndex * 9 + sticker;
                  return (
                    <button
                      key={index}
                      type="button"
                      onClick={() => updateSticker(index)}
                      className={`rounded border border-ink/20 ${swatches[stickers[index]]}`}
                      aria-label={`${face} sticker ${sticker + 1}`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2 text-xs text-muted">
            {colors.map((color) => (
              <span key={color} className={counts[color] === 9 ? "text-green-700" : "text-accent"}>
                {color}: {counts[color]}/9
              </span>
            ))}
          </div>
            <button
            type="button"
            disabled={!isValid}
            onClick={() => onApply(stickers)}
            className="h-10 border border-cyan-300 bg-cyan-300 px-4 text-sm font-bold uppercase tracking-[0.08em] text-black disabled:cursor-not-allowed disabled:opacity-45"
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}
