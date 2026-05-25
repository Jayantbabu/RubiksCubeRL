"use client";

import type { SolverAnalytics, SolverStep } from "@/lib/types";

type AnalyticsPanelProps = {
  analytics?: SolverAnalytics;
};

type SeriesKey = "confidence" | "valueEstimate" | "reward";

function formatMs(ms: number) {
  return `${(ms / 1000).toFixed(2)}s`;
}

function Sparkline({ steps, seriesKey }: { steps: SolverStep[]; seriesKey: SeriesKey }) {
  const values = steps.map((step) => step[seriesKey]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 36 - ((value - min) / spread) * 30;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox="0 0 100 40" className="h-24 w-full overflow-visible">
      <polyline fill="none" stroke="#22D3EE" strokeWidth="2.4" points={points} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function AnalyticsPanel({ analytics }: AnalyticsPanelProps) {
  if (!analytics) {
    return null;
  }

  const stats = [
    ["RL moves", analytics.moveCount.toString()],
    ["Baseline moves", analytics.baselineMoveCount.toString()],
    ["RL time", formatMs(analytics.solveTimeMs)],
    ["Avg confidence", `${Math.round(analytics.averageConfidence * 100)}%`],
    ["Reward", analytics.totalReward.toFixed(2)],
    ["Final value", analytics.finalValueEstimate.toFixed(2)]
  ];

  const rlWidth = `${Math.max(18, (analytics.moveCount / analytics.baselineMoveCount) * 100)}%`;

  return (
    <section className="w-full max-w-5xl space-y-4">
      <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {stats.map(([label, value]) => (
          <div key={label} className="border border-white/10 bg-[#0b0b0b] p-3">
            <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/40">{label}</div>
            <div className="mt-1 text-xl font-semibold text-cyan-300">{value}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-4">
        <div className="border border-white/10 bg-[#0b0b0b] p-4">
          <div className="text-xs font-bold uppercase tracking-[0.12em] text-white/55">Confidence</div>
          <Sparkline steps={analytics.steps} seriesKey="confidence" />
        </div>
        <div className="border border-white/10 bg-[#0b0b0b] p-4">
          <div className="text-xs font-bold uppercase tracking-[0.12em] text-white/55">Value Estimate</div>
          <Sparkline steps={analytics.steps} seriesKey="valueEstimate" />
        </div>
        <div className="border border-white/10 bg-[#0b0b0b] p-4">
          <div className="text-xs font-bold uppercase tracking-[0.12em] text-white/55">Reward</div>
          <Sparkline steps={analytics.steps} seriesKey="reward" />
        </div>
        <div className="border border-white/10 bg-[#0b0b0b] p-4">
          <div className="text-xs font-bold uppercase tracking-[0.12em] text-white/55">Move Comparison</div>
          <div className="mt-6 space-y-4">
            <div>
              <div className="mb-1 flex justify-between text-xs text-white/45">
                <span>RL</span>
                <span>{analytics.moveCount}</span>
              </div>
              <div className="h-3 bg-white/10">
                <div className="h-3 bg-cyan-300" style={{ width: rlWidth }} />
              </div>
            </div>
            <div>
              <div className="mb-1 flex justify-between text-xs text-white/45">
                <span>Baseline</span>
                <span>{analytics.baselineMoveCount}</span>
              </div>
              <div className="h-3 bg-white/10">
                <div className="h-3 bg-white/35" style={{ width: "100%" }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
