"use client";

import { Users, Loader2, XCircle, ArrowLeft, ChevronRight } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import {
  FLSimulation,
  formatDuration,
  parseMetrics,
} from "../../types/fl-simulation";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  Filler,
} from "chart.js";
import annotationPlugin from "chartjs-plugin-annotation";
import { Line, Bar } from "react-chartjs-2";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loading } from "../ui/loading";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  ChartLegend,
  Filler,
  annotationPlugin,
);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FLSimulationDetailsPageProps {
  simulationId: number;
  simulationName: string;
  onBack: () => void;
}

interface Client {
  id: number;
  client_name: string;
  model_type: string;
  metrics: string;
}

interface ClientMetrics {
  global: {
    post_fl: {
      loss: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      specificity?: number;
      roc_auc?: number;
      class_gap?: number;
      leukemia_accuracy?: number;
      healthy_accuracy?: number;
      num_samples?: number;
      num_healthy_samples?: number;
      num_leukemia_samples?: number;
      confusion_matrix?: ConfusionMatrix;
      evaluated_at?: string;
    };
    improvement: {
      loss: number;
      accuracy: number;
      precision?: number;
      recall?: number;
      f1_score?: number;
      specificity?: number;
      roc_auc?: number;
      class_gap?: number;
      healthy_accuracy?: number;
      leukemia_accuracy?: number;
    };
  };
  data_heterogeneity?: {
    total_samples: number;
    train_samples: number;
    val_samples: number;
    class_distribution: {
      leukemia: number;
      healthy: number;
      leukemia_pct: number;
      healthy_pct: number;
    };
    imbalance_ratio: number;
    partition_id: number;
  };
  rounds: Array<{
    round: number;
    training?: {
      train_loss: number;
      train_accuracy: number;
      val_loss?: number;
      distill_loss?: number;
      training_time?: number;
      num_examples?: number;
      consensus_weight?: number;
      balanced_accuracy?: number;
    };
    validation?: {
      loss: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      evaluated_at: string;
      healthy_accuracy?: number;
      leukemia_accuracy?: number;
      class_gap?: number;
    };
  }>;
}

interface ConfusionMatrix {
  TP: number;
  FN: number;
  FP: number;
  TN: number;
}

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

const ACCENT = "#111827";
const MUTED = "#6B7280";
const BORDER = "#E5E7EB";
const SURFACE = "#F9FAFB";
const WHITE = "#FFFFFF";

// Per-client palette: muted, intentional, non-rainbow
const CLIENT_PALETTE = [
  { line: "#1D4ED8", fill: "rgba(29,78,216,0.06)" },
  { line: "#059669", fill: "rgba(5,150,105,0.06)" },
  { line: "#B45309", fill: "rgba(180,83,9,0.06)" },
  { line: "#7C3AED", fill: "rgba(124,58,237,0.06)" },
  { line: "#DB2777", fill: "rgba(219,39,119,0.06)" },
];

const VAL_COLOR = "#1D4ED8";
const TRAIN_COLOR = "#6B7280";
const BALANCED_COLOR = "#D97706";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pct(v: number | undefined, decimals = 1) {
  return v != null ? `${(v * 100).toFixed(decimals)}%` : "—";
}

function fmt(v: number | undefined, decimals = 3) {
  return v != null ? v.toFixed(decimals) : "—";
}

function smartYLimit(
  values: number[],
  padding = 0.08,
  allowNegative = false,
  snap = 0,
): { min: number; max: number; stepSize?: number } {
  const valid = values.filter((v) => v != null && !isNaN(v));
  if (valid.length === 0) return { min: 0, max: 1 };
  const lo = Math.min(...valid);
  const hi = Math.max(...valid);
  const range = hi - lo;
  if (range < 0.01) {
    const c = (hi + lo) / 2;
    return {
      min: allowNegative ? c - 0.01 : Math.max(0, c - 0.01),
      max: c + 0.01,
    };
  }
  const pad = range * padding;
  let min = allowNegative ? lo - pad : Math.max(0, lo - pad);
  let max = hi + pad;
  if (snap > 0) {
    min = Math.floor(min / snap) * snap;
    max = Math.ceil(max / snap) * snap;
    return { min, max, stepSize: snap };
  }
  return { min, max };
}

function gapSeverity(gap: number) {
  if (gap <= 0.1)
    return {
      label: "Balanced",
      cls: "text-emerald-700 bg-emerald-50 border-emerald-200",
    };
  if (gap <= 0.2)
    return {
      label: "Moderate",
      cls: "text-amber-700 bg-amber-50 border-amber-200",
    };
  return { label: "High gap", cls: "text-red-700 bg-red-50 border-red-200" };
}

// ---------------------------------------------------------------------------
// Shared chart options factory
// ---------------------------------------------------------------------------

function lineOptions(
  limits: { min: number; max: number; stepSize?: number },
  isPercent = false,
  bestRound?: number,
) {
  const annotations: Record<string, any> = {};
  if (bestRound && bestRound > 0) {
    annotations.best = {
      type: "line" as const,
      xMin: String(bestRound),
      xMax: String(bestRound),
      borderColor: "rgba(16,185,129,0.6)",
      borderWidth: 1.5,
      borderDash: [4, 3],
      label: {
        display: true,
        content: `Best · R${bestRound}`,
        position: "start" as const,
        backgroundColor: "rgba(16,185,129,0.9)",
        color: "#fff",
        font: { size: 10, weight: "500" as const },
        padding: { x: 6, y: 2 },
        borderRadius: 3,
      },
    };
  }
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index" as const, intersect: false },
    plugins: {
      legend: {
        display: true,
        position: "top" as const,
        align: "end" as const,
        labels: {
          usePointStyle: true,
          pointStyleWidth: 8,
          padding: 20,
          font: { size: 11, family: "'DM Mono', monospace" },
          color: MUTED,
        },
      },
      tooltip: {
        backgroundColor: WHITE,
        titleColor: ACCENT,
        bodyColor: MUTED,
        borderColor: BORDER,
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        titleFont: { size: 11, weight: "500" as const },
        bodyFont: { size: 11 },
        callbacks: {
          label: (ctx: any) => {
            const v = ctx.raw;
            if (v == null) return "";
            return isPercent
              ? `  ${ctx.dataset.label}: ${(v * 100).toFixed(1)}%`
              : `  ${ctx.dataset.label}: ${v.toFixed(4)}`;
          },
        },
      },
      annotation: { annotations },
    },
    scales: {
      y: {
        min: limits.min,
        max: limits.max,
        grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
        border: { display: false },
        ticks: {
          stepSize: limits.stepSize,
          font: { size: 10, family: "'DM Mono', monospace" },
          color: MUTED,
          padding: 8,
          callback: (v: any) =>
            isPercent
              ? `${(Number(v) * 100).toFixed(0)}%`
              : Number(v).toFixed(3),
        },
      },
      x: {
        grid: { display: false },
        border: { display: false },
        ticks: {
          font: { size: 10, family: "'DM Mono', monospace" },
          color: MUTED,
          padding: 6,
        },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Atom: StatPill — inline label/value badge
// ---------------------------------------------------------------------------

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-500">
      <span className="text-gray-300 font-light">/</span>
      <span>{label}</span>
      <span className="font-semibold text-gray-900">{value}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Atom: KpiCard
// ---------------------------------------------------------------------------

function KpiCard({
  label,
  value,
  sub,
  highlight,
}: {
  label: string;
  value: string;
  sub?: string;
  highlight?: "green" | "amber" | "red" | "neutral";
}) {
  const valueColor =
    highlight === "green"
      ? "text-emerald-700"
      : highlight === "amber"
        ? "text-amber-700"
        : highlight === "red"
          ? "text-red-700"
          : "text-gray-900";

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-5 flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
        {label}
      </span>
      <span className={`text-2xl font-semibold tabular-nums ${valueColor}`}>
        {value}
      </span>
      {sub && <span className="text-xs text-gray-400 font-medium">{sub}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Atom: SectionHeader
// ---------------------------------------------------------------------------

function SectionHeader({
  title,
  aside,
}: {
  title: string;
  aside?: React.ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between mb-5">
      <h2 className="text-sm font-semibold text-gray-900 tracking-tight">
        {title}
      </h2>
      {aside && <div className="flex items-center gap-2">{aside}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Atom: ChartCard
// ---------------------------------------------------------------------------

function ChartCard({
  title,
  sub,
  children,
}: {
  title: string;
  sub?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-gray-100 bg-white p-5">
      <div className="mb-4">
        <p className="text-xs font-semibold text-gray-900">{title}</p>
        {sub && <p className="text-[11px] text-gray-400 mt-0.5">{sub}</p>}
      </div>
      <div className="h-64">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Atom: Tag
// ---------------------------------------------------------------------------

function Tag({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-semibold border ${className}`}
    >
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// ConfusionMatrixCard
// ---------------------------------------------------------------------------

function ConfusionMatrixCard({
  cm,
  accuracy,
}: {
  cm?: ConfusionMatrix;
  accuracy?: number;
}) {
  if (!cm) {
    return (
      <div className="h-44 flex items-center justify-center text-xs text-gray-300">
        No data
      </div>
    );
  }
  const { TP = 0, FN = 0, FP = 0, TN = 0 } = cm;
  const total = TP + FN + FP + TN;

  const Cell = ({
    value,
    label,
  }: {
    value: number;
    label: string;
  }) => {
    const frac = total > 0 ? value / total : 0;
    // Standard red: hex #dc2626, rgb(220, 38, 38)
    const opacity = 0.04 + frac * 0.86;
    const bg = `rgba(220, 38, 38, ${opacity})`;
    const isDark = opacity > 0.45;
    const textColor = isDark ? "text-white" : "text-red-900";
    const labelColor = isDark ? "text-red-100/80" : "text-red-500/70";

    return (
      <div
        className={`rounded-lg border border-red-100/20 flex flex-col items-center justify-center gap-0.5 py-4 ${textColor} transition-colors duration-300 shadow-sm`}
        style={{ background: bg }}
      >
        <span className="text-2xl font-bold tabular-nums tracking-tight">{value}</span>
        <span className={`text-[10px] font-bold uppercase tracking-widest ${labelColor}`}>{label}</span>
      </div>
    );
  };

  return (
    <div>
      <div className="grid grid-cols-2 gap-1.5 mb-1.5 pl-20 text-center">
        <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
          Pred +
        </p>
        <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
          Pred −
        </p>
      </div>
      <div className="flex gap-1.5">
        <div className="w-20 flex flex-col gap-1.5">
          <div className="flex-1 flex items-center justify-end pr-3">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider leading-tight text-right">
              Actual
              <br />+
            </span>
          </div>
          <div className="flex-1 flex items-center justify-end pr-3">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider leading-tight text-right">
              Actual
              <br />−
            </span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-1.5 flex-1">
          <Cell value={TP} label="TP" />
          <Cell value={FN} label="FN" />
          <Cell value={FP} label="FP" />
          <Cell value={TN} label="TN" />
        </div>
      </div>
      {accuracy != null && (
        <p className="text-center text-[11px] font-medium text-gray-400 mt-4">
          Accuracy{" "}
          <span className="font-bold text-gray-900 bg-gray-100 px-2 py-0.5 rounded ml-1">
            {pct(accuracy)}
          </span>
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ClassBar
// ---------------------------------------------------------------------------

function ClassBar({
  label,
  value,
  pctVal,
  color,
}: {
  label: string;
  value: number;
  pctVal: number;
  color: string;
}) {
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="text-xs font-medium text-gray-600">{label}</span>
        <span className="text-xs tabular-nums text-gray-400">
          {value.toLocaleString()} · {pctVal.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pctVal}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function FLSimulationDetailsPage({
  simulationId,
  simulationName,
  onBack,
}: FLSimulationDetailsPageProps) {
  const [simulation, setSimulation] = useState<FLSimulation | null>(null);
  const [clients, setClients] = useState<Client[]>([]);
  const [selectedClient, setSelectedClient] = useState<string>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // ---- Fetch ----
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setLoadError(null);
        const simRes = await fetch(
          `${API_BASE_PATH}/fl_simulations/${simulationId}`,
        );
        if (!simRes.ok) throw new Error("Failed to fetch simulation");
        const simData: FLSimulation = await simRes.json();
        setSimulation(simData);
        if (simData.status === "completed") {
          const mRes = await fetch(
            `${API_BASE_PATH}/client-simulation-metrics/simulation/${simulationId}`,
          );
          if (!mRes.ok) throw new Error("Failed to fetch client metrics");
          const mData = await mRes.json();
          setClients(
            mData.map((r: any) => ({
              id: r.client_id,
              client_name: r.clients?.client_name || `Client ${r.client_id}`,
              model_type: r.clients?.model_type || "unknown",
              metrics: JSON.stringify(r.metrics),
            })),
          );
        }
        setIsLoading(false);
      } catch (err) {
        console.error(err);
        setLoadError("Failed to load simulation data");
        toast.error("Failed to load simulation data");
        setIsLoading(false);
      }
    };
    fetchData();
  }, [simulationId]);

  // ---- Parse metrics ----
  const clientMetricsMap = useMemo(() => {
    const map = new Map<number, ClientMetrics>();
    clients.forEach((c) => {
      try {
        map.set(c.id, JSON.parse(c.metrics) as ClientMetrics);
      } catch {}
    });
    return map;
  }, [clients]);

  const maxRounds = useMemo(() => {
    let m = 0;
    clientMetricsMap.forEach((cm) => {
      if (cm.rounds) m = Math.max(m, cm.rounds.length);
    });
    return m;
  }, [clientMetricsMap]);

  const roundLabels = useMemo(
    () => Array.from({ length: maxRounds }, (_, i) => `${i + 1}`),
    [maxRounds],
  );

  const bestModelRound = useMemo(() => {
    const bmr = simulation?.aggregate_metrics?.best_model_round;
    return bmr && typeof bmr === "number" && bmr > 0 ? bmr : undefined;
  }, [simulation]);

  // ---- Loss chart ----
  const lossChartData = useMemo(() => {
    if (selectedClient === "all") {
      const avg = (key: "val_loss" | "train_loss") =>
        roundLabels.map((_, ri) => {
          const vals: number[] = [];
          clients.forEach((c) => {
            const v = clientMetricsMap.get(c.id)?.rounds?.[ri]?.training?.[key];
            if (v != null) vals.push(v);
          });
          return vals.length
            ? vals.reduce((a, b) => a + b, 0) / vals.length
            : null;
        });
      return {
        labels: roundLabels,
        datasets: [
          {
            label: "Val loss",
            data: avg("val_loss"),
            borderColor: VAL_COLOR,
            backgroundColor: "rgba(29,78,216,0.05)",
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 2,
            fill: true,
            spanGaps: true,
          },
          {
            label: "Train loss",
            data: avg("train_loss"),
            borderColor: TRAIN_COLOR,
            backgroundColor: "transparent",
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 1.5,
            borderDash: [4, 3],
            fill: false,
            spanGaps: true,
          },
        ],
      };
    }
    const cm = clientMetricsMap.get(parseInt(selectedClient));
    const rounds = cm?.rounds ?? [];
    const info = clients.find((c) => c.id === parseInt(selectedClient));
    return {
      labels: rounds.map((r) => `${r.round}`),
      datasets: [
        {
          label: `${info?.client_name ?? "Client"} · val`,
          data: rounds.map((r) => r.training?.val_loss ?? null),
          borderColor: VAL_COLOR,
          backgroundColor: "rgba(29,78,216,0.05)",
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 2,
          fill: true,
          spanGaps: true,
        },
        {
          label: `${info?.client_name ?? "Client"} · train`,
          data: rounds.map((r) => r.training?.train_loss ?? null),
          borderColor: TRAIN_COLOR,
          backgroundColor: "transparent",
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 1.5,
          borderDash: [4, 3],
          fill: false,
          spanGaps: true,
        },
      ],
    };
  }, [clients, clientMetricsMap, roundLabels, selectedClient]);

  const lossLimits = useMemo(() => {
    const vals: number[] = [];
    lossChartData.datasets.forEach((ds) =>
      ds.data.forEach((v) => {
        if (v != null) vals.push(v as number);
      }),
    );
    return smartYLimit(vals, 0.2, false, 0.05);
  }, [lossChartData]);

  // ---- Accuracy chart ----
  const accChartData = useMemo(() => {
    if (selectedClient === "all") {
      const avg = (source: "validation" | "training", key: string) =>
        roundLabels.map((_, ri) => {
          const vals: number[] = [];
          clients.forEach((c) => {
            const v = (clientMetricsMap.get(c.id)?.rounds?.[ri] as any)?.[
              source
            ]?.[key];
            if (v != null) vals.push(v);
          });
          return vals.length
            ? vals.reduce((a, b) => a + b, 0) / vals.length
            : null;
        });
      // balanced = avg of leukemia_acc and healthy_acc from validation
      const avgBalanced = roundLabels.map((_, ri) => {
        const vals: number[] = [];
        clients.forEach((c) => {
          const vr = clientMetricsMap.get(c.id)?.rounds?.[ri]?.validation;
          if (vr?.leukemia_accuracy != null && vr?.healthy_accuracy != null) {
            vals.push((vr.leukemia_accuracy + vr.healthy_accuracy) / 2);
          }
        });
        return vals.length
          ? vals.reduce((a, b) => a + b, 0) / vals.length
          : null;
      });
      return {
        labels: roundLabels,
        datasets: [
          {
            label: "Val acc",
            data: avg("validation", "accuracy"),
            borderColor: VAL_COLOR,
            backgroundColor: "rgba(29,78,216,0.05)",
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 2,
            fill: true,
            spanGaps: true,
          },
          {
            label: "Train acc",
            data: avg("training", "train_accuracy"),
            borderColor: TRAIN_COLOR,
            backgroundColor: "transparent",
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 1.5,
            borderDash: [4, 3],
            fill: false,
            spanGaps: true,
          },
          {
            label: "Balanced acc",
            data: avgBalanced,
            borderColor: BALANCED_COLOR,
            backgroundColor: "rgba(217,119,6,0.05)",
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 1.5,
            borderDash: [2, 3],
            fill: false,
            spanGaps: true,
          },
        ],
      };
    }
    const cm = clientMetricsMap.get(parseInt(selectedClient));
    const rounds = cm?.rounds ?? [];
    const info = clients.find((c) => c.id === parseInt(selectedClient));
    const balancedData = rounds.map((r) => {
      const leuk = r.validation?.leukemia_accuracy;
      const heal = r.validation?.healthy_accuracy;
      return leuk != null && heal != null ? (leuk + heal) / 2 : null;
    });
    return {
      labels: rounds.map((r) => `${r.round}`),
      datasets: [
        {
          label: `${info?.client_name ?? "Client"} · val acc`,
          data: rounds.map((r) => r.validation?.accuracy ?? null),
          borderColor: VAL_COLOR,
          backgroundColor: "rgba(29,78,216,0.05)",
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 2,
          fill: true,
          spanGaps: true,
        },
        {
          label: `${info?.client_name ?? "Client"} · train acc`,
          data: rounds.map((r) => r.training?.train_accuracy ?? null),
          borderColor: TRAIN_COLOR,
          backgroundColor: "transparent",
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 1.5,
          borderDash: [4, 3],
          fill: false,
          spanGaps: true,
        },
        {
          label: `${info?.client_name ?? "Client"} · balanced acc`,
          data: balancedData,
          borderColor: BALANCED_COLOR,
          backgroundColor: "rgba(217,119,6,0.05)",
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 1.5,
          borderDash: [2, 3],
          fill: false,
          spanGaps: true,
        },
      ],
    };
  }, [clients, clientMetricsMap, roundLabels, selectedClient]);

  const accLimits = useMemo(() => {
    const vals: number[] = [];
    accChartData.datasets.forEach((ds) =>
      ds.data.forEach((v) => {
        if (v != null) vals.push(v as number);
      }),
    );
    return smartYLimit(vals, 0.2, false, 0.05);
  }, [accChartData]);

  // ---- Delta chart ----
  /* const accDeltaChartData = useMemo(() => {
    if (selectedClient === "all") {
      const avgAccs = roundLabels.map((_, ri) => {
        const vals: number[] = [];
        clients.forEach((c) => {
          const rd = clientMetricsMap.get(c.id)?.rounds?.[ri];
          const v = rd?.validation?.accuracy ?? rd?.training?.train_accuracy;
          if (v != null) vals.push(v);
        });
        return vals.length
          ? vals.reduce((a, b) => a + b, 0) / vals.length
          : null;
      });
      const deltas = avgAccs.slice(1).map((v, i) => {
        const prev = avgAccs[i];
        return v != null && prev != null ? v - prev : null;
      });
      return {
        labels: roundLabels.slice(1),
        datasets: [
          {
            label: "Δ accuracy",
            data: deltas,
            borderColor: VAL_COLOR,
            backgroundColor: "rgba(29,78,216,0.05)",
            tension: 0.35,
            pointRadius: 3,
            borderWidth: 2,
            fill: true,
            spanGaps: true,
          },
        ],
      };
    }
    const cm = clientMetricsMap.get(parseInt(selectedClient));
    const rounds = cm?.rounds ?? [];
    const valAccs = rounds.map(
      (r) => r.validation?.accuracy ?? r.training?.train_accuracy ?? null,
    );
    const deltas = valAccs.slice(1).map((v, i) => {
      const prev = valAccs[i];
      return v != null && prev != null ? v - prev : null;
    });
    const info = clients.find((c) => c.id === parseInt(selectedClient));
    return {
      labels: rounds.slice(1).map((r) => `${r.round}`),
      datasets: [
        {
          label: `${info?.client_name ?? "Client"} · Δ`,
          data: deltas,
          borderColor: VAL_COLOR,
          backgroundColor: "rgba(29,78,216,0.05)",
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 2,
          fill: true,
          spanGaps: true,
        },
      ],
    };
  }, [clients, clientMetricsMap, roundLabels, selectedClient]);

  const deltaLimits = useMemo(() => {
    const vals: number[] = [];
    accDeltaChartData.datasets.forEach((ds) =>
      ds.data.forEach((v) => {
        if (v != null) vals.push(v as number);
      }),
    );
    return smartYLimit(vals, 0.35, true, 0.05);
  }, [accDeltaChartData]); */

  // ---- Data het chart ----
  const dataHetChart = useMemo(() => {
    const items: { name: string; leukemia: number; healthy: number }[] = [];
    clients.forEach((c) => {
      const dh = clientMetricsMap.get(c.id)?.data_heterogeneity;
      if (dh)
        items.push({
          name: c.client_name,
          leukemia: dh.class_distribution.leukemia,
          healthy: dh.class_distribution.healthy,
        });
    });
    if (!items.length) return null;
    return {
      labels: items.map((i) => i.name),
      datasets: [
        {
          label: "ALL",
          data: items.map((i) => i.leukemia),
          backgroundColor: "rgba(220,38,38,0.7)",
          borderColor: "rgba(220,38,38,0.9)",
          borderWidth: 1,
        },
        {
          label: "Healthy",
          data: items.map((i) => i.healthy),
          backgroundColor: "rgba(16,185,129,0.7)",
          borderColor: "rgba(16,185,129,0.9)",
          borderWidth: 1,
        },
      ],
    };
  }, [clients, clientMetricsMap]);

  const totalSamples = useMemo(() => {
    let t = 0;
    clients.forEach((c) => {
      t += clientMetricsMap.get(c.id)?.data_heterogeneity?.total_samples ?? 0;
    });
    return t;
  }, [clients, clientMetricsMap]);

  const dataHetOptions = useMemo(
    () => ({
      indexAxis: "y" as const,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top" as const,
          align: "end" as const,
          labels: {
            usePointStyle: true,
            pointStyleWidth: 8,
            padding: 20,
            font: { size: 11, family: "'DM Mono', monospace" },
            color: MUTED,
          },
        },
        tooltip: {
          backgroundColor: WHITE,
          titleColor: ACCENT,
          bodyColor: MUTED,
          borderColor: BORDER,
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: (ctx: any) => {
              const v = ctx.raw;
              const di = ctx.dataIndex;
              const l = dataHetChart?.datasets[0]?.data[di] ?? 0;
              const h = dataHetChart?.datasets[1]?.data[di] ?? 0;
              const t = l + h;
              const p = t > 0 ? ((v / t) * 100).toFixed(1) : "0";
              return `  ${ctx.dataset.label}: ${v.toLocaleString()} (${p}%)`;
            },
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          beginAtZero: true,
          grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
          border: { display: false },
          ticks: {
            font: { size: 10, family: "'DM Mono', monospace" },
            color: MUTED,
          },
        },
        y: {
          stacked: true,
          grid: { display: false },
          border: { display: false },
          ticks: { font: { size: 11 }, color: MUTED },
        },
      },
    }),
    [dataHetChart],
  );

  // ---- Post-FL aggregates ----
  const postFlStats = useMemo(() => {
    const accs: number[] = [],
      gaps: number[] = [],
      balAccs: number[] = [];
    let worst = { name: "", acc: 1 };
    clients.forEach((c) => {
      const pf = clientMetricsMap.get(c.id)?.global?.post_fl;
      if (!pf) return;
      accs.push(pf.accuracy);
      gaps.push(pf.class_gap ?? 0);
      // balanced accuracy = avg of per-class accuracies
      const leuk = pf.leukemia_accuracy ?? 0;
      const heal = pf.healthy_accuracy ?? 0;
      if (pf.leukemia_accuracy != null && pf.healthy_accuracy != null) {
        balAccs.push((leuk + heal) / 2);
      }
      if (pf.accuracy < worst.acc)
        worst = { name: c.client_name, acc: pf.accuracy };
    });
    const avg = (a: number[]) =>
      a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    return {
      avgAcc: avg(accs),
      avgGap: avg(gaps),
      avgBalAcc: avg(balAccs),
      worst,
    };
  }, [clients, clientMetricsMap]);

  const selectedCM =
    selectedClient !== "all"
      ? clientMetricsMap.get(parseInt(selectedClient))
      : null;
  const selectedInfo =
    selectedClient !== "all"
      ? clients.find((c) => c.id === parseInt(selectedClient))
      : null;

  // ===========================================================================
  // Render
  // ===========================================================================

  if (isLoading && !simulation)
    return <Loading className="min-h-[400px]" text="Loading…" />;
  if (!simulation)
    return (
      <div className="flex items-center justify-center min-h-[400px] text-sm text-gray-400">
        Simulation not found.
      </div>
    );

  // Running / failed states
  if (simulation.status === "running" || simulation.status === "pending") {
    return (
      <div className="flex flex-col items-center justify-center min-h-[480px] gap-4">
        <Loading size="lg" />
        <p className="text-sm font-medium text-gray-700">Simulation running…</p>
        <p className="text-xs text-gray-400">
          Results will appear here when training completes.
        </p>
      </div>
    );
  }

  if (simulation.status === "failed") {
    return (
      <div className="flex flex-col items-center justify-center min-h-[480px] gap-3">
        <XCircle className="w-10 h-10 text-red-400" />
        <p className="text-sm font-semibold text-gray-900">Simulation failed</p>
        {simulation.error_message && (
          <p className="text-xs text-gray-500 max-w-sm text-center">
            {simulation.error_message}
          </p>
        )}
      </div>
    );
  }

  const metrics = parseMetrics(simulation.aggregate_metrics);

  return (
    <div className="min-h-screen bg-gray-50/60">
      {/* ── Top bar ── */}
      <div className="sticky top-0 z-20 bg-white/90 backdrop-blur border-b border-gray-100">
        <div className="px-6 h-14 flex items-center gap-3">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-900 transition-colors font-medium"
          >
            <ArrowLeft size={13} />
            Back
          </button>
          <span className="text-gray-200 select-none">/</span>
          <span className="text-xs font-semibold text-gray-900 truncate">
            {simulationName}
          </span>

          <div className="ml-auto flex items-center gap-4">
            <StatPill label="ID" value={`#${simulationId}`} />
            <StatPill label="Clients" value={String(clients.length)} />
            {simulation.duration != null && (
              <StatPill
                label="Duration"
                value={formatDuration(simulation.duration)}
              />
            )}
            {bestModelRound && (
              <StatPill label="Best" value={`R${bestModelRound}`} />
            )}
            <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-md">
              Completed
            </span>
          </div>
        </div>
      </div>

      {/* ── Body ── */}
      <div className="px-6 py-8 space-y-10">
        {/* Client selector */}
        <div className="flex items-center justify-between">
          <Select value={selectedClient} onValueChange={setSelectedClient}>
            <SelectTrigger className="w-60 h-8 text-xs font-medium border-gray-200 bg-white rounded-lg shadow-none">
              <SelectValue placeholder="Select client" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">
                <span className="text-xs">All clients · overview</span>
              </SelectItem>
              {clients.map((c, idx) => (
                <SelectItem key={c.id} value={c.id.toString()}>
                  <div className="flex items-center gap-2 text-xs">
                    <div
                      className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                      style={{
                        background:
                          CLIENT_PALETTE[idx % CLIENT_PALETTE.length].line,
                      }}
                    />
                    {c.client_name}
                    <span className="text-gray-400">{c.model_type}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* ── Config summary ── */}
        {simulation.configs && (
          <section>
            <SectionHeader title="Run configuration" />
            <div className="rounded-xl border border-gray-100 bg-white overflow-hidden">
              <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-gray-100">
                {/* FL core */}
                <div className="p-5 space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-gray-300">
                    FL core
                  </p>
                  {[
                    [
                      "Rounds / epochs",
                      `${simulation.configs.num_rounds ?? simulation.configs.num_server_rounds} / ${simulation.configs.local_epochs}`,
                    ],
                    [
                      "Base LR",
                      `${(simulation.configs.learning_rate ?? simulation.configs.lr ?? 0).toFixed(4)}`,
                    ],
                    ["Batch size", String(simulation.configs.batch_size)],
                    [
                      "Weight decay",
                      String(simulation.configs.weight_decay ?? "—"),
                    ],
                  ].map(([k, v]) => (
                    <div
                      key={k}
                      className="flex justify-between items-baseline"
                    >
                      <span className="text-xs text-gray-400">{k}</span>
                      <span className="text-xs font-semibold text-gray-900 tabular-nums">
                        {v}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Strategy */}
                <div className="p-5 space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-gray-300">
                    Strategy
                  </p>
                  {[
                    [
                      "Minority boost",
                      String(
                        simulation.configs.minority_boost ??
                          simulation.configs.training_config?.minority_boost ??
                          "—",
                      ),
                    ],
                    [
                      "Consensus momentum",
                      String(
                        simulation.configs.consensus_momentum ??
                          simulation.configs.training_config
                            ?.consensus_momentum ??
                          "—",
                      ),
                    ],
                    [
                      "Focal γ",
                      String(
                        simulation.configs.focal_gamma ??
                          simulation.configs.training_config?.focal_gamma ??
                          "—",
                      ),
                    ],
                  ].map(([k, v]) => (
                    <div
                      key={k}
                      className="flex justify-between items-baseline"
                    >
                      <span className="text-xs text-gray-400">{k}</span>
                      <span className="text-xs font-semibold text-gray-900 tabular-nums">
                        {v}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Distillation */}
                <div className="p-5 space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-gray-300">
                    Distillation
                  </p>
                  {[
                    ["Temperature", String(simulation.configs.temperature)],
                    [
                      "Distill epochs",
                      String(simulation.configs.distill_epochs),
                    ],
                    [
                      "Weight base",
                      String(
                        simulation.configs.distill_weight_base ??
                          simulation.configs.training_config
                            ?.distill_weight_base ??
                          "—",
                      ),
                    ],
                  ].map(([k, v]) => (
                    <div
                      key={k}
                      className="flex justify-between items-baseline"
                    >
                      <span className="text-xs text-gray-400">{k}</span>
                      <span className="text-xs font-semibold text-gray-900 tabular-nums">
                        {v}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Partitioning */}
                <div className="p-5 space-y-3">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-gray-300">
                    Partitioning
                  </p>
                  {[
                    ["Dirichlet α", String(simulation.configs.dirichlet_alpha)],
                    ["Seed", String(simulation.configs.dirichlet_seed ?? "—")],
                    [
                      "Min partition",
                      String(
                        simulation.configs.dirichlet_min_partition_size ?? "—",
                      ),
                    ],
                  ].map(([k, v]) => (
                    <div
                      key={k}
                      className="flex justify-between items-baseline"
                    >
                      <span className="text-xs text-gray-400">{k}</span>
                      <span className="text-xs font-semibold text-gray-900 tabular-nums">
                        {v}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Post-FL: All clients overview ── */}
        {selectedClient === "all" && (
          <section>
            <SectionHeader
              title="Post-FL evaluation"
              aside={
                <span className="text-xs text-gray-400">
                  balanced test set · 50/50 split
                </span>
              }
            />

            {/* KPI row */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
              <KpiCard label="Avg accuracy" value={pct(postFlStats.avgAcc)} />
              <KpiCard
                label="Avg balanced acc"
                value={pct(postFlStats.avgBalAcc)}
                sub="Per-class avg"
              />
              <KpiCard
                label="Avg class gap"
                value={pct(postFlStats.avgGap)}
                sub={gapSeverity(postFlStats.avgGap).label}
                highlight={
                  postFlStats.avgGap <= 0.1
                    ? "green"
                    : postFlStats.avgGap <= 0.2
                      ? "amber"
                      : "red"
                }
              />
              <KpiCard
                label="Weakest"
                value={pct(postFlStats.worst.acc)}
                sub={postFlStats.worst.name}
              />
              {bestModelRound && (
                <KpiCard
                  label="Best model"
                  value={`Round ${bestModelRound}`}
                  sub="Used for evaluation"
                  highlight="green"
                />
              )}
            </div>

            {/* Comparison table */}
            <div className="rounded-xl border border-gray-100 bg-white overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-100">
                    {[
                      "Client",
                      "Accuracy",
                      "Balanced acc",
                      "Class gap",
                      "Leukemia acc",
                      "Healthy acc",
                      "Precision",
                      "Recall",
                    ].map((h) => (
                      <th
                        key={h}
                        className={`py-3 px-4 font-semibold uppercase tracking-wider text-[10px] text-gray-400 ${h === "Client" ? "text-left" : "text-right"}`}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {clients.map((c, idx) => {
                    const pf = clientMetricsMap.get(c.id)?.global?.post_fl;
                    if (!pf) return null;
                    const gap = pf.class_gap ?? 0;
                    const sev = gapSeverity(gap);
                    return (
                      <tr
                        key={c.id}
                        className="border-b border-gray-50 last:border-0 hover:bg-gray-50/60 transition-colors"
                      >
                        <td className="py-3.5 px-4">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                              style={{
                                background:
                                  CLIENT_PALETTE[idx % CLIENT_PALETTE.length]
                                    .line,
                              }}
                            />
                            <span className="font-semibold text-gray-900">
                              {c.client_name}
                            </span>
                            <span className="text-gray-400">
                              {c.model_type}
                            </span>
                          </div>
                        </td>
                        <td className="py-3.5 px-4 text-right font-semibold text-gray-900 tabular-nums">
                          {pct(pf.accuracy)}
                        </td>
                        <td className="py-3.5 px-4 text-right tabular-nums text-amber-700 font-medium">
                          {pf.leukemia_accuracy != null &&
                          pf.healthy_accuracy != null
                            ? pct(
                                (pf.leukemia_accuracy + pf.healthy_accuracy) /
                                  2,
                              )
                            : "—"}
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <Tag className={sev.cls}>{pct(gap)}</Tag>
                        </td>
                        <td className="py-3.5 px-4 text-right tabular-nums text-gray-700">
                          {pct(pf.leukemia_accuracy)}
                        </td>
                        <td className="py-3.5 px-4 text-right tabular-nums text-gray-700">
                          {pct(pf.healthy_accuracy)}
                        </td>
                        <td className="py-3.5 px-4 text-right tabular-nums text-gray-700">
                          {pct(pf.precision)}
                        </td>
                        <td className="py-3.5 px-4 text-right tabular-nums text-gray-700">
                          {pct(pf.recall)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ── Post-FL: Individual client ── */}
        {selectedClient !== "all" && selectedCM && (
          <section>
            <SectionHeader
              title="Post-FL evaluation"
              aside={
                <span className="text-xs text-gray-400">
                  {selectedInfo?.client_name} · {selectedInfo?.model_type}
                </span>
              }
            />

            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-3">
              <KpiCard
                label="Accuracy"
                value={pct(selectedCM.global.post_fl.accuracy)}
              />
              <KpiCard
                label="Balanced acc"
                value={
                  selectedCM.global.post_fl.leukemia_accuracy != null &&
                  selectedCM.global.post_fl.healthy_accuracy != null
                    ? pct(
                        (selectedCM.global.post_fl.leukemia_accuracy +
                          selectedCM.global.post_fl.healthy_accuracy) /
                          2,
                      )
                    : "—"
                }
                sub="Per-class avg"
                highlight="neutral"
              />
              <KpiCard
                label="Class gap"
                value={pct(selectedCM.global.post_fl.class_gap ?? 0)}
                sub={
                  gapSeverity(selectedCM.global.post_fl.class_gap ?? 0).label
                }
                highlight={
                  (selectedCM.global.post_fl.class_gap ?? 0) <= 0.1
                    ? "green"
                    : (selectedCM.global.post_fl.class_gap ?? 0) <= 0.2
                      ? "amber"
                      : "red"
                }
              />
              <KpiCard
                label="Leukemia acc"
                value={pct(selectedCM.global.post_fl.leukemia_accuracy)}
                sub="Sensitivity"
              />
              <KpiCard
                label="Healthy acc"
                value={pct(selectedCM.global.post_fl.healthy_accuracy)}
                sub="Specificity"
              />
            </div>

            <div className="grid grid-cols-3 gap-3 mb-6">
              <KpiCard
                label="Precision"
                value={pct(selectedCM.global.post_fl.precision)}
              />
              <KpiCard
                label="Recall"
                value={pct(selectedCM.global.post_fl.recall)}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Confusion matrix */}
              <div className="rounded-xl border border-gray-100 bg-white p-5">
                <p className="text-xs font-semibold text-gray-900 mb-5">
                  Confusion matrix
                </p>
                <ConfusionMatrixCard
                  cm={selectedCM.global.post_fl.confusion_matrix}
                  accuracy={selectedCM.global.post_fl.accuracy}
                />
              </div>

              {/* Data partition */}
              {selectedCM.data_heterogeneity && (
                <div className="rounded-xl border border-gray-100 bg-white p-5">
                  <p className="text-xs font-semibold text-gray-900 mb-5">
                    Data partition
                  </p>
                  <div className="space-y-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Total</span>
                      <span className="font-semibold text-gray-900">
                        {selectedCM.data_heterogeneity.total_samples.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Train / Val split</span>
                      <span className="font-medium text-gray-700">
                        {selectedCM.data_heterogeneity.train_samples.toLocaleString()}{" "}
                        /{" "}
                        {selectedCM.data_heterogeneity.val_samples.toLocaleString()}
                      </span>
                    </div>
                    <div className="pt-2 space-y-4 border-t border-gray-50">
                      <ClassBar
                        label="ALL (leukemia)"
                        value={
                          selectedCM.data_heterogeneity.class_distribution
                            .leukemia
                        }
                        pctVal={
                          selectedCM.data_heterogeneity.class_distribution
                            .leukemia_pct
                        }
                        color="rgba(220,38,38,0.75)"
                      />
                      <ClassBar
                        label="Healthy"
                        value={
                          selectedCM.data_heterogeneity.class_distribution
                            .healthy
                        }
                        pctVal={
                          selectedCM.data_heterogeneity.class_distribution
                            .healthy_pct
                        }
                        color="rgba(16,185,129,0.75)"
                      />
                    </div>
                    <div className="flex justify-between items-center pt-3 border-t border-gray-50">
                      <span className="text-xs text-gray-400">
                        Imbalance ratio
                      </span>
                      <Tag
                        className={
                          selectedCM.data_heterogeneity.imbalance_ratio <= 1.5
                            ? "text-emerald-700 bg-emerald-50 border-emerald-200"
                            : selectedCM.data_heterogeneity.imbalance_ratio <= 5
                              ? "text-amber-700 bg-amber-50 border-amber-200"
                              : "text-red-700 bg-red-50 border-red-200"
                        }
                      >
                        {selectedCM.data_heterogeneity.imbalance_ratio.toFixed(
                          1,
                        )}
                        :1
                      </Tag>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ── Data heterogeneity (all clients) ── */}
        {selectedClient === "all" && dataHetChart && (
          <section>
            <SectionHeader
              title="Data heterogeneity"
              aside={
                <>
                  <Tag className="text-gray-600 bg-gray-50 border-gray-200">
                    {totalSamples.toLocaleString()} samples
                  </Tag>
                  <Tag className="text-blue-700 bg-blue-50 border-blue-200">
                    α = {simulation?.configs?.dirichlet_alpha ?? "—"}
                  </Tag>
                </>
              }
            />
            <div className="rounded-xl border border-gray-100 bg-white p-5">
              <div
                style={{ height: `${Math.max(120, clients.length * 64)}px` }}
              >
                <Bar data={dataHetChart} options={dataHetOptions} />
              </div>
            </div>
          </section>
        )}

        {/* ── Training convergence ── */}
        <section>
          <SectionHeader title="Training convergence" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartCard title="Loss" sub="Validation vs training loss per round">
              <Line
                data={lossChartData}
                options={lineOptions(lossLimits, false, bestModelRound)}
              />
            </ChartCard>
            <ChartCard
              title="Accuracy"
              sub="Validation vs training accuracy per round"
            >
              <Line
                data={accChartData}
                options={lineOptions(accLimits, true, bestModelRound)}
              />
            </ChartCard>
            {/* <ChartCard title="Accuracy delta" sub="Round-to-round change">
              <Line
                data={accDeltaChartData}
                options={lineOptions(deltaLimits, true)}
              />
            </ChartCard> */}
          </div>
        </section>
      </div>
    </div>
  );
}
