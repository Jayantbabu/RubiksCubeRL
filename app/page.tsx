"use client";

import { useState } from "react";
import { AnalyticsPanel } from "@/components/AnalyticsPanel";
import { ColorInputPanel } from "@/components/ColorInputPanel";
import { Controls } from "@/components/Controls";
import { CubeStage } from "@/components/CubeStage";
import { MoveTester } from "@/components/MoveTester";
import { ProgressStrip } from "@/components/ProgressStrip";
import { Switch } from "@/components/Switch";
import { applyMove, getCubeState, resetCube, setCubeState } from "@/lib/cubeBridge";
import { requestSolverSolution } from "@/lib/solverClient";
import { generateScramble } from "@/lib/scramble";
import type { CubeState, Move, SolverAnalytics, SolverMode } from "@/lib/types";

export default function Home() {
  const [busy, setBusy] = useState(false);
  const [colorPanelOpen, setColorPanelOpen] = useState(false);
  const [, setStatus] = useState("Ready.");
  const [activeMove, setActiveMove] = useState<Move>();
  const [solutionMoves, setSolutionMoves] = useState<Move[]>([]);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(-1);
  const [analytics, setAnalytics] = useState<SolverAnalytics>();
  const [cubeResetKey, setCubeResetKey] = useState(0);
  const [solverMode, setSolverMode] = useState<SolverMode>("rl");
  const [autoRotate, setAutoRotate] = useState(true);
  const [solving, setSolving] = useState(false);

  const runMoves = async (moves: Move[], label: string) => {
    setSolutionMoves(moves);
    setCurrentMoveIndex(0);

    for (let index = 0; index < moves.length; index += 1) {
      setCurrentMoveIndex(index);
      setActiveMove(undefined);
      await new Promise((resolve) => window.setTimeout(resolve, 20));
      setActiveMove(moves[index]);
      setStatus(`${label}: ${moves[index]} (${index + 1}/${moves.length})`);
      await applyMove(moves[index]);
    }

    setActiveMove(undefined);
    setCurrentMoveIndex(moves.length);
  };

  const handleApplyColors = async (state: CubeState) => {
    setBusy(true);
    setAnalytics(undefined);
    setStatus("Applying validated sticker colors.");

    try {
      await setCubeState(state);
      setColorPanelOpen(false);
      setStatus("Cube state updated.");
    } finally {
      setBusy(false);
    }
  };

  const handleShuffle = async () => {
    setBusy(true);
    setAnalytics(undefined);
    const scramble = generateScramble();

    try {
      await runMoves(scramble, "Shuffle");
      setStatus(`Scramble complete: ${scramble.join(" ")}`);
    } finally {
      setBusy(false);
    }
  };

  const handleSolve = async () => {
    setBusy(true);
    setSolving(true);
    setAnalytics(undefined);
    setStatus(`Requesting ${solverMode} solution.`);

    try {
      const cubeState = await getCubeState();
      const result = await requestSolverSolution(cubeState, solverMode);
      await runMoves(result.moves, "Solve");
      setAnalytics(result.analytics);
      setStatus(`Solved with ${solverMode} solver.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Solve failed.");
    } finally {
      setSolving(false);
      setBusy(false);
    }
  };

  const handleReset = async () => {
    setBusy(true);
    setAnalytics(undefined);
    setSolutionMoves([]);
    setCurrentMoveIndex(-1);
    setActiveMove(undefined);
    setStatus("Resetting cube.");

    try {
      await resetCube();
      setCubeResetKey((key) => key + 1);
      setStatus("Cube reset.");
    } finally {
      setBusy(false);
    }
  };

  const handleManualMove = async (move: Move) => {
    setBusy(true);
    setAnalytics(undefined);
    setSolutionMoves([move]);
    setCurrentMoveIndex(0);
    setActiveMove(move);
    setStatus(`Manual move: ${move}`);

    try {
      await applyMove(move);
      setCurrentMoveIndex(1);
      setStatus(`Manual move complete: ${move}`);
    } finally {
      setActiveMove(undefined);
      setBusy(false);
    }
  };

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-center gap-7 overflow-hidden bg-black px-4 pb-24 pt-8 text-white">
      <div className="fixed top-5 z-20 flex flex-wrap items-center justify-center gap-3">
        <Switch
          checked={solverMode === "kociemba"}
          label="Solver"
          offLabel="Custom RL"
          onLabel="Kociemba"
          onChange={(checked) => setSolverMode(checked ? "kociemba" : "rl")}
        />
        <Switch
          checked={autoRotate}
          label="Cube"
          offLabel="Static"
          onLabel="Auto"
          onChange={setAutoRotate}
        />
      </div>
      <CubeStage activeMove={activeMove} resetKey={cubeResetKey} autoRotate={autoRotate} />
      <Controls
        busy={busy}
        solving={solving}
        onAddColors={() => setColorPanelOpen(true)}
        onShuffle={handleShuffle}
        onSolve={handleSolve}
        onReset={handleReset}
      />
      <MoveTester busy={busy} onMove={handleManualMove} />
      <ProgressStrip moves={solutionMoves} currentIndex={currentMoveIndex} />
      {/* <AnalyticsPanel analytics={analytics} /> */}
      <ColorInputPanel open={colorPanelOpen} onClose={() => setColorPanelOpen(false)} onApply={handleApplyColors} />
    </main>
  );
}
