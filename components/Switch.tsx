"use client";

type SwitchProps = {
  checked: boolean;
  label: string;
  offLabel: string;
  onLabel: string;
  onChange: (checked: boolean) => void;
};

export function Switch({ checked, label, offLabel, onLabel, onChange }: SwitchProps) {
  return (
    <label className="flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.06] px-4 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-white/70 backdrop-blur">
      <span className="hidden sm:inline">{label}</span>
      <span className={!checked ? "text-cyan-300" : "text-white/35"}>{offLabel}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-7 w-12 rounded-full border transition ${
          checked ? "border-cyan-300 bg-cyan-300/25" : "border-white/15 bg-black/70"
        }`}
      >
        <span
          className={`absolute top-1 h-5 w-5 rounded-full bg-white transition ${
            checked ? "left-6 shadow-[0_0_18px_rgba(34,211,238,0.75)]" : "left-1"
          }`}
        />
      </button>
      <span className={checked ? "text-cyan-300" : "text-white/35"}>{onLabel}</span>
    </label>
  );
}
