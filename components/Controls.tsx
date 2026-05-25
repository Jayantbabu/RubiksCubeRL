"use client";

type ControlsProps = {
  busy: boolean;
  solving: boolean;
  onAddColors: () => void;
  onShuffle: () => void;
  onSolve: () => void;
  onReset: () => void;
};

export function Controls({ busy, solving, onAddColors, onShuffle, onSolve, onReset }: ControlsProps) {
  const buttons = [
    { label: "Solve", action: onSolve },
    { label: "Shuffle", action: onShuffle },
    { label: "Reset", action: onReset },
    { label: "Add Colors", action: onAddColors }
  ];

  return (
    <div className="grid w-full max-w-2xl grid-cols-2 gap-3 sm:grid-cols-4">
      {buttons.map((button) => (
        <button
          key={button.label}
          type="button"
          onClick={button.action}
          disabled={busy}
          className="h-12 rounded-full border border-white/10 bg-white px-5 text-xs font-bold uppercase tracking-[0.08em] text-black shadow-[0_12px_35px_rgba(0,0,0,0.35)] transition hover:-translate-y-0.5 hover:bg-cyan-300 hover:shadow-[0_0_28px_rgba(34,211,238,0.24)] disabled:cursor-not-allowed disabled:opacity-45"
        >
          <span className="flex items-center justify-center gap-2">
            {button.label === "Solve" && solving ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-black/20 border-t-black" />
            ) : null}
            {button.label === "Solve" && solving ? "Solving" : button.label}
          </span>
        </button>
      ))}
    </div>
  );
}
