"use client";

import { Users, Loader2, Clock, XCircle, AlertCircle } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import {
  FLSimulation,
  formatDuration,
  parseMetrics,
} from "@/types/fl-simulation";
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
} from "chart.js";
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
// Constants
// ---------------------------------------------------------------------------

const CLIENT_COLORS = [
  "#B80028",
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function smartYLimit(
  values: number[],
  padding: number = 0.15,
  allowNegative: boolean = false,
): { min: number; max: number } {
  const valid = (values ?? []).filter((v) => v != null && !isNaN(v));
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
  return {
    min: allowNegative ? lo - pad : Math.max(0, lo - pad),
    max: hi + pad,
  };
}

function gapBadge(gap: number) {
  if (gap <= 0.1) return "bg-emerald-50 text-emerald-700 border-emerald-200";
  if (gap <= 0.2) return "bg-amber-50 text-amber-700 border-amber-200";
  return "bg-red-50 text-red-700 border-red-200";
}

function gapLabel(gap: number) {
  if (gap <= 0.1) return "Balanced";
  if (gap <= 0.2) return "Moderate";
  return "High Gap";
}

function pct(v: number | undefined) {
  return v != null ? `${(v * 100).toFixed(1)}%` : "—";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ConfusionMatrixCard({
  confusionMatrix,
  accuracy,
}: {
  confusionMatrix?: ConfusionMatrix;
  accuracy?: number;
}) {
  if (!confusionMatrix) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
        No confusion matrix data
      </div>
    );
  }

  const { TP = 0, FN = 0, FP = 0, TN = 0 } = confusionMatrix;
  const maxVal = Math.max(TP, FN, FP, TN);
  const intensity = (v: number) =>
    maxVal === 0 ? 0 : (v / maxVal) * 0.8 + 0.2;
  const bg = (v: number) => {
    const l = 100 - intensity(v) * 50;
    return `hsl(347, 91%, ${l}%)`;
  };
  const fg = (v: number) =>
    intensity(v) > 0.6 ? "text-white" : "text-slate-900";

  const Cell = ({ value, label }: { value: number; label: string }) => (
    <div
      className={`h-20 flex flex-col items-center justify-center rounded-lg border border-slate-200 ${fg(value)}`}
      style={{ backgroundColor: bg(value) }}
    >
      <span className="text-2xl font-bold">{value}</span>
      <span className="text-[10px] font-medium opacity-80">{label}</span>
    </div>
  );

  return (
    <div>
      {/* Column labels */}
      <div className="grid grid-cols-2 gap-2 mb-1.5 ml-24">
        <div className="text-center text-xs font-medium text-slate-500">
          Pred Leukemia
        </div>
        <div className="text-center text-xs font-medium text-slate-500">
          Pred Healthy
        </div>
      </div>

      <div className="flex gap-2">
        {/* Row labels */}
        <div className="flex flex-col gap-2 justify-center w-20">
          <div className="h-20 flex items-center justify-end pr-2 text-xs font-medium text-slate-500 text-right leading-tight">
            Actual
            <br />
            Leukemia
          </div>
          <div className="h-20 flex items-center justify-end pr-2 text-xs font-medium text-slate-500 text-right leading-tight">
            Actual
            <br />
            Healthy
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 flex-1">
          <Cell value={TP} label="True Positive" />
          <Cell value={FN} label="False Negative" />
          <Cell value={FP} label="False Positive" />
          <Cell value={TN} label="True Negative" />
        </div>
      </div>

      {accuracy != null && (
        <div className="mt-3 text-center text-sm text-slate-500">
          Accuracy:{" "}
          <span className="font-semibold text-slate-900">{pct(accuracy)}</span>
        </div>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5">
      <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
        {label}
      </div>
      <div className={`text-2xl font-bold ${accent ?? "text-slate-900"}`}>
        {value}
      </div>
      {sub && (
        <div className="text-xs text-slate-400 mt-1 font-medium">{sub}</div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chart option builder
// ---------------------------------------------------------------------------

function makeLineOptions(
  limits: { min: number; max: number },
  isPercentage: boolean = false,
) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index" as const, intersect: false },
    plugins: {
      legend: {
        display: true,
        position: "top" as const,
        labels: { usePointStyle: true, padding: 16, font: { size: 11 } },
      },
      tooltip: {
        backgroundColor: "rgba(255,255,255,0.95)",
        titleColor: "#1e293b",
        bodyColor: "#1e293b",
        borderColor: "#e2e8f0",
        borderWidth: 1,
        padding: 10,
        callbacks: {
          label: (ctx: any) => {
            const v = ctx.raw;
            if (v == null) return "";
            return isPercentage
              ? `${ctx.dataset.label}: ${(v * 100).toFixed(1)}%`
              : `${ctx.dataset.label}: ${v.toFixed(4)}`;
          },
        },
      },
    },
    scales: {
      y: {
        min: limits.min,
        max: limits.max,
        grid: { color: "rgba(0,0,0,0.04)" },
        ticks: {
          font: { size: 11 },
          callback: (v: any) =>
            isPercentage
              ? `${(Number(v) * 100).toFixed(0)}%`
              : Number(v).toFixed(3),
        },
      },
      x: {
        grid: { color: "rgba(0,0,0,0.04)" },
        ticks: { font: { size: 11 } },
      },
    },
  };
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

  console.log(simulation);

  // ---- Fetch data ----
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

  // ---- Parse client metrics ----
  const clientMetricsMap = useMemo(() => {
    const map = new Map<number, ClientMetrics>();
    clients.forEach((c) => {
      try {
        map.set(c.id, JSON.parse(c.metrics) as ClientMetrics);
      } catch {
        /* skip */
      }
    });
    return map;
  }, [clients]);

  // ---- Training convergence data (multi-client loss + consensus weights) ----
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

  // Loss chart
  const lossChartData = useMemo(() => {
    if (selectedClient === "all") {
      const avgData = roundLabels.map((_, ri) => {
        const vals: number[] = [];
        clients.forEach((c) => {
          const v = clientMetricsMap.get(c.id)?.rounds?.[ri]?.training
            ?.val_loss;
          if (v != null) vals.push(v);
        });
        return vals.length > 0
          ? vals.reduce((a, b) => a + b, 0) / vals.length
          : null;
      });
      return {
        labels: roundLabels,
        datasets: [
          {
            label: "Avg Validation Loss",
            data: avgData,
            borderColor: CLIENT_COLORS[0],
            backgroundColor: "rgba(184, 0, 40, 0.08)",
            tension: 0.3,
            pointRadius: 4,
            borderWidth: 2.5,
            fill: true,
            spanGaps: true,
          },
        ],
      };
    } else {
      const cm = clientMetricsMap.get(parseInt(selectedClient));
      const rounds = cm?.rounds ?? [];
      const labels = rounds.map((r) => `${r.round}`);
      return {
        labels,
        datasets: [
          {
            label: "Val Loss",
            data: rounds.map((r) => r.training?.val_loss ?? null),
            borderColor: CLIENT_COLORS[0],
            backgroundColor: "rgba(184, 0, 40, 0.08)",
            tension: 0.3,
            pointRadius: 4,
            borderWidth: 2.5,
            fill: true,
            spanGaps: true,
          },
        ],
      };
    }
  }, [clients, clientMetricsMap, roundLabels, selectedClient]);

  const lossValues = useMemo(() => {
    const vals: number[] = [];
    lossChartData.datasets.forEach((ds) =>
      ds.data.forEach((v) => {
        if (v != null) vals.push(v);
      }),
    );
    return vals;
  }, [lossChartData]);

  const lossOptions = useMemo(
    () => makeLineOptions(smartYLimit(lossValues)),
    [lossValues],
  );

  // Accuracy chart
  const accChartData = useMemo(() => {
    if (selectedClient === "all") {
      const avgData = roundLabels.map((_, ri) => {
        const vals: number[] = [];
        clients.forEach((c) => {
          const rd = clientMetricsMap.get(c.id)?.rounds?.[ri];
          const v = rd?.validation?.accuracy ?? rd?.training?.train_accuracy;
          if (v != null) vals.push(v);
        });
        return vals.length > 0
          ? vals.reduce((a, b) => a + b, 0) / vals.length
          : null;
      });
      return {
        labels: roundLabels,
        datasets: [
          {
            label: "Avg Validation Accuracy",
            data: avgData,
            borderColor: CLIENT_COLORS[1],
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            tension: 0.3,
            pointRadius: 4,
            borderWidth: 2.5,
            fill: true,
            spanGaps: true,
          },
        ],
      };
    } else {
      const cm = clientMetricsMap.get(parseInt(selectedClient));
      const rounds = cm?.rounds ?? [];
      const labels = rounds.map((r) => `${r.round}`);
      const cInfo = clients.find((c) => c.id === parseInt(selectedClient));
      return {
        labels,
        datasets: [
          {
            label: cInfo?.client_name ?? "Val Accuracy",
            data: rounds.map((r) => r.validation?.accuracy ?? null),
            borderColor: CLIENT_COLORS[1],
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            tension: 0.3,
            pointRadius: 4,
            borderWidth: 2.5,
            fill: true,
            spanGaps: true,
          },
        ],
      };
    }
  }, [clients, clientMetricsMap, roundLabels, selectedClient]);

  const accValues = useMemo(() => {
    const vals: number[] = [];
    accChartData.datasets.forEach((ds) =>
      ds.data.forEach((v) => {
        if (v != null) vals.push(v as number);
      }),
    );
    return vals;
  }, [accChartData]);

  const accOptions = useMemo(
    () => makeLineOptions(smartYLimit(accValues, 0.15), true),
    [accValues],
  );

  // Accuracy difference (round-to-round delta) chart
  const accDeltaChartData = useMemo(() => {
    if (selectedClient === "all") {
      // Compute average accuracy per round first, then delta
      const avgAccs = roundLabels.map((_, ri) => {
        const vals: number[] = [];
        clients.forEach((c) => {
          const rd = clientMetricsMap.get(c.id)?.rounds?.[ri];
          const v = rd?.validation?.accuracy ?? rd?.training?.train_accuracy;
          if (v != null) vals.push(v);
        });
        return vals.length > 0
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
            label: "Avg Accuracy Delta",
            data: deltas,
            borderColor: CLIENT_COLORS[2],
            backgroundColor: "rgba(16, 185, 129, 0.08)",
            tension: 0.3,
            pointRadius: 4,
            borderWidth: 2.5,
            fill: true,
            spanGaps: true,
          },
        ],
      };
    } else {
      const cm = clientMetricsMap.get(parseInt(selectedClient));
      const rounds = cm?.rounds ?? [];
      const labels = rounds.slice(1).map((r) => `${r.round}`);
      const valAccs = rounds.map(
        (r) => r.validation?.accuracy ?? r.training?.train_accuracy ?? null,
      );
      const deltas = valAccs.slice(1).map((v, i) => {
        const prev = valAccs[i];
        return v != null && prev != null ? v - prev : null;
      });
      const cInfo = clients.find((c) => c.id === parseInt(selectedClient));
      return {
        labels,
        datasets: [
          {
            label: cInfo?.client_name ?? "Client",
            data: deltas,
            borderColor: CLIENT_COLORS[0],
            backgroundColor: CLIENT_COLORS[0],
            tension: 0.3,
            pointRadius: 3,
            borderWidth: 2,
            spanGaps: true,
          },
        ],
      };
    }
  }, [clients, clientMetricsMap, roundLabels, selectedClient]);

  const accDeltaValues = useMemo(() => {
    const vals: number[] = [];
    accDeltaChartData.datasets.forEach((ds) =>
      ds.data.forEach((v) => {
        if (v != null) vals.push(v as number);
      }),
    );
    return vals;
  }, [accDeltaChartData]);

  const accDeltaOptions = useMemo(
    () => makeLineOptions(smartYLimit(accDeltaValues, 0.25, true), true),
    [accDeltaValues],
  );

  // ---- Post-FL aggregate stats ----
  const postFlStats = useMemo(() => {
    const accs: number[] = [];
    const gaps: number[] = [];
    const f1s: number[] = [];
    let worst = { name: "", acc: 1 };

    clients.forEach((c) => {
      const cm = clientMetricsMap.get(c.id);
      const pf = cm?.global?.post_fl;
      if (!pf) return;
      accs.push(pf.accuracy);
      gaps.push(pf.class_gap ?? 0);
      f1s.push(pf.f1_score);
      if (pf.accuracy < worst.acc)
        worst = { name: c.client_name, acc: pf.accuracy };
    });

    const avg = (arr: number[]) =>
      arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    return {
      avgAcc: avg(accs),
      avgGap: avg(gaps),
      avgF1: avg(f1s),
      worst,
    };
  }, [clients, clientMetricsMap]);

  // ---- Data heterogeneity chart ----
  const dataHetChart = useMemo(() => {
    const items: { name: string; leukemia: number; healthy: number }[] = [];
    clients.forEach((c) => {
      const cm = clientMetricsMap.get(c.id);
      const dh = cm?.data_heterogeneity;
      if (dh)
        items.push({
          name: c.client_name,
          leukemia: dh.class_distribution?.leukemia ?? 0,
          healthy: dh.class_distribution?.healthy ?? 0,
        });
    });
    if (items.length === 0) return null;
    return {
      labels: items.map((i) => i.name),
      datasets: [
        {
          label: "ALL (Leukemia)",
          data: items.map((i) => i.leukemia),
          backgroundColor: "rgba(220, 38, 38, 0.75)",
          borderColor: "rgb(220, 38, 38)",
          borderWidth: 1,
        },
        {
          label: "Healthy",
          data: items.map((i) => i.healthy),
          backgroundColor: "rgba(34, 197, 94, 0.75)",
          borderColor: "rgb(34, 197, 94)",
          borderWidth: 1,
        },
      ],
    };
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
          labels: { usePointStyle: true, padding: 16, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: "rgba(255,255,255,0.95)",
          titleColor: "#1e293b",
          bodyColor: "#1e293b",
          borderColor: "#e2e8f0",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx: any) => {
              const v = ctx.raw;
              const di = ctx.dataIndex;
              const t =
                (dataHetChart?.datasets[0]?.data[di] ?? 0) +
                (dataHetChart?.datasets[1]?.data[di] ?? 0);
              const p = t > 0 ? ((v / t) * 100).toFixed(1) : "0";
              return `${ctx.dataset.label}: ${v.toLocaleString()} (${p}%)`;
            },
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          beginAtZero: true,
          grid: { color: "rgba(0,0,0,0.04)" },
          title: {
            display: true,
            text: "Samples",
            font: { weight: "bold" as const, size: 11 },
          },
        },
        y: {
          stacked: true,
          grid: { display: false },
        },
      },
    }),
    [dataHetChart],
  );

  // ---- Selected client helpers ----
  const selectedCM =
    selectedClient !== "all"
      ? clientMetricsMap.get(parseInt(selectedClient))
      : null;
  const selectedInfo =
    selectedClient !== "all"
      ? clients.find((c) => c.id === parseInt(selectedClient))
      : null;

  // ===========================================================================
  // Conditional rendering for loading / running / failed states
  // ===========================================================================

  return (
    <div className="p-8 w-full animate-in fade-in duration-500">
      {isLoading && !simulation ? (
        <Loading className="min-h-[400px]" text="Loading simulation data..." />
      ) : !simulation ? (
        <div className="p-8 flex flex-col items-center justify-center min-h-[400px] text-muted-foreground">
          <p>Simulation not found</p>
        </div>
      ) : (
        (() => {
          if (
            simulation.status === "running" ||
            simulation.status === "pending"
          ) {
            return (
              <div className="flex flex-col items-center justify-center p-16 bg-muted/30 border-2 border-dashed border-border rounded-lg text-center">
                <Loader2 className="h-16 w-16 mb-4 animate-spin text-primary" />
                <h2 className="text-xl font-semibold mb-2">
                  Simulation In Progress
                </h2>
                <p className="text-muted-foreground max-w-md">
                  The simulation is currently running. This may take several
                  minutes.
                </p>
              </div>
            );
          }
          if (simulation.status === "failed") {
            return (
              <div className="flex flex-col items-center justify-center p-16 bg-red-50/50 border-2 border-dashed border-red-200 rounded-lg text-center">
                <XCircle className="h-16 w-16 mb-4 text-red-500" />
                <h2 className="text-xl font-semibold mb-2 text-red-900">
                  Simulation Failed
                </h2>
                {simulation.error_message && (
                  <p className="mt-4 max-w-md text-sm text-red-700 bg-white rounded p-4">
                    {simulation.error_message}
                  </p>
                )}
              </div>
            );
          }

          const metrics = parseMetrics(simulation.aggregate_metrics);
          const globalMetrics = metrics?.aggregate?.post_fl;
          const improvement = metrics?.aggregate?.improvement;

          return (
            <div className="space-y-10">
              {/* ── Header ── */}
              <div className="flex items-center justify-between">
                <div className="text-sm text-slate-500">
                  <span className="font-mono bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">
                    #{simulationId}
                  </span>{" "}
                  &middot;{" "}
                  <span className="font-medium text-slate-700">
                    {clients.length} Clients
                  </span>{" "}
                  &middot;{" "}
                  <span className="font-medium text-emerald-600">
                    Completed
                  </span>
                  {simulation.duration != null && (
                    <>
                      {" "}
                      &middot;{" "}
                      <span className="font-medium text-slate-700">
                        {formatDuration(simulation.duration)}
                      </span>
                    </>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <Users size={16} className="text-slate-400" />
                  <Select
                    value={selectedClient}
                    onValueChange={setSelectedClient}
                  >
                    <SelectTrigger className="w-[260px] h-9 text-sm font-medium">
                      <SelectValue placeholder="Select client" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">
                        All Clients (Overview)
                      </SelectItem>
                      {clients.map((c, idx) => (
                        <SelectItem key={c.id} value={c.id.toString()}>
                          <div className="flex items-center gap-2">
                            <div
                              className="w-2 h-2 rounded-full"
                              style={{
                                backgroundColor:
                                  CLIENT_COLORS[idx % CLIENT_COLORS.length],
                              }}
                            />
                            {c.client_name}{" "}
                            <span className="text-xs text-slate-400">
                              ({c.model_type})
                            </span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* ── Simulation Config Summary ── */}
              {simulation.configs && (
                <section className="bg-white rounded-2xl border border-slate-200 p-5">
                  <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">
                    {simulation.heterogeneity_preset && (
                      <div className="flex items-center gap-2">
                        <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">
                          Preset
                        </span>
                        <span
                          className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                            simulation.heterogeneity_preset === "high"
                              ? "border-red-300 text-red-700 bg-red-50"
                              : simulation.heterogeneity_preset === "moderate"
                                ? "border-amber-300 text-amber-700 bg-amber-50"
                                : simulation.heterogeneity_preset === "low"
                                  ? "border-emerald-300 text-emerald-700 bg-emerald-50"
                                  : "border-purple-300 text-purple-700 bg-purple-50"
                          }`}
                        >
                          {simulation.heterogeneity_preset
                            .charAt(0)
                            .toUpperCase() +
                            simulation.heterogeneity_preset.slice(1)}
                        </span>
                      </div>
                    )}
                    <div className="h-4 w-px bg-slate-200" />
                    <div>
                      <span className="text-slate-400 text-xs">Rounds</span>{" "}
                      <span className="font-semibold text-slate-700">
                        {simulation.configs.num_server_rounds}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 text-xs">Epochs</span>{" "}
                      <span className="font-semibold text-slate-700">
                        {simulation.configs.local_epochs}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 text-xs">LR</span>{" "}
                      <span className="font-semibold text-slate-700">
                        {simulation.configs.lr}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 text-xs">LR Decay</span>{" "}
                      <span className="font-semibold text-slate-700">
                        {simulation.configs.lr_decay}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 text-xs">Batch</span>{" "}
                      <span className="font-semibold text-slate-700">
                        {simulation.configs.batch_size}
                      </span>
                    </div>
                    {simulation.configs.dirichlet_alpha != null && (
                      <div>
                        <span className="text-slate-400 text-xs">
                          Dirichlet &alpha;
                        </span>{" "}
                        <span className="font-semibold text-slate-700">
                          {simulation.configs.dirichlet_alpha}
                        </span>
                      </div>
                    )}
                    {simulation.configs.temperature != null && (
                      <div>
                        <span className="text-slate-400 text-xs">Temp</span>{" "}
                        <span className="font-semibold text-slate-700">
                          {simulation.configs.temperature}
                        </span>
                      </div>
                    )}
                    {simulation.duration != null && (
                      <>
                        <div className="h-4 w-px bg-slate-200" />
                        <div>
                          <span className="text-slate-400 text-xs">
                            Duration
                          </span>{" "}
                          <span className="font-semibold text-slate-700">
                            {formatDuration(simulation.duration)}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </section>
              )}

              {/* ── Section: Data Heterogeneity (all clients view) ── */}
              {selectedClient === "all" && dataHetChart && (
                <section>
                  <div className="flex items-center justify-between mb-5">
                    <h2 className="text-xl font-bold text-slate-900">
                      Data Heterogeneity
                    </h2>
                    <span className="text-xs font-medium bg-blue-50 text-blue-700 px-3 py-1 rounded-full border border-blue-200">
                      Dirichlet &alpha; ={" "}
                      {simulation?.heterogeneity_preset === "moderate"
                        ? "2.5"
                        : simulation?.heterogeneity_preset === "high"
                          ? "1.0"
                          : "5.0"}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Bar chart */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-200">
                      <h3 className="text-sm font-semibold text-slate-700 mb-3">
                        Class Distribution per Client
                      </h3>
                      <div className="h-72">
                        <Bar data={dataHetChart} options={dataHetOptions} />
                      </div>
                    </div>

                    {/* Table */}
                    <div className="bg-white p-6 rounded-2xl border border-slate-200">
                      <h3 className="text-sm font-semibold text-slate-700 mb-3">
                        Client Breakdown
                      </h3>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-200 text-left">
                            <th className="py-2.5 font-semibold text-slate-600">
                              Client
                            </th>
                            <th className="py-2.5 font-semibold text-slate-600 text-right">
                              Samples
                            </th>
                            <th className="py-2.5 font-semibold text-slate-600 text-right">
                              Ratio
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {clients.map((c, idx) => {
                            const cm = clientMetricsMap.get(c.id);
                            const dh = cm?.data_heterogeneity;
                            return (
                              <tr
                                key={c.id}
                                className="border-b border-slate-100 hover:bg-slate-50"
                              >
                                <td className="py-2.5 font-medium text-slate-900 flex items-center gap-2">
                                  <div
                                    className="w-2 h-2 rounded-full"
                                    style={{
                                      backgroundColor:
                                        CLIENT_COLORS[
                                          idx % CLIENT_COLORS.length
                                        ],
                                    }}
                                  />
                                  {c.client_name}
                                </td>
                                <td className="py-2.5 text-right text-slate-700">
                                  {dh?.total_samples?.toLocaleString() ?? "—"}
                                </td>
                                <td className="py-2.5 text-right">
                                  {dh ? (
                                    <span
                                      className={`text-xs font-medium px-2 py-0.5 rounded-full border ${
                                        dh.imbalance_ratio <= 1.5
                                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                          : dh.imbalance_ratio <= 5
                                            ? "bg-amber-50 text-amber-700 border-amber-200"
                                            : "bg-red-50 text-red-700 border-red-200"
                                      }`}
                                    >
                                      {dh.imbalance_ratio.toFixed(1)}:1
                                    </span>
                                  ) : (
                                    "—"
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </section>
              )}

              {/* ── Section: Training Convergence ── */}
              <section>
                <h2 className="text-xl font-bold text-slate-900 mb-5">
                  Training Convergence
                </h2>
                <div className="flex flex-wrap items-start gap-6">
                  <div className="bg-white p-6 rounded-2xl border border-slate-200 w-full lg:w-[calc(33.333%-1rem)]">
                    <h3 className="text-sm font-semibold text-slate-700 mb-1">
                      {selectedClient === "all"
                        ? "Avg Validation Loss"
                        : "Validation Loss"}
                    </h3>
                    <p className="text-xs text-slate-400 mb-3">
                      {selectedClient === "all"
                        ? "Average validation loss across all clients"
                        : "The loss value calculated on the validation set after local training rounds"}
                    </p>
                    <div className="h-72">
                      <Line data={lossChartData} options={lossOptions} />
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-2xl border border-slate-200 w-full lg:w-[calc(33.333%-1rem)]">
                    <h3 className="text-sm font-semibold text-slate-700 mb-1">
                      {selectedClient === "all"
                        ? "Avg Validation Accuracy"
                        : "Validation Accuracy"}
                    </h3>
                    <p className="text-xs text-slate-400 mb-3">
                      {selectedClient === "all"
                        ? "Average validation accuracy across all clients"
                        : "Validation accuracy each round"}
                    </p>
                    <div className="h-72">
                      <Line data={accChartData} options={accOptions} />
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-2xl border border-slate-200 w-full lg:w-[calc(33.333%-1rem)]">
                    <h3 className="text-sm font-semibold text-slate-700 mb-1">
                      {selectedClient === "all"
                        ? "Avg Accuracy Delta"
                        : "Accuracy Delta"}
                    </h3>
                    <p className="text-xs text-slate-400 mb-3">
                      Round-to-round accuracy change (positive = improving)
                    </p>
                    <div className="h-72">
                      <Line
                        data={accDeltaChartData}
                        options={accDeltaOptions}
                      />
                    </div>
                  </div>
                </div>
              </section>

              {/* ── Section: Post-FL Results — Overall (all clients view) ── */}
              {selectedClient === "all" && (
                <section>
                  <h2 className="text-xl font-bold text-slate-900 mb-5">
                    Post-FL Evaluation
                    <span className="text-xs font-normal text-slate-400 ml-2">
                      on balanced test set (50/50)
                    </span>
                  </h2>

                  {/* Summary cards */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <MetricCard
                      label="Avg Accuracy"
                      value={pct(postFlStats.avgAcc)}
                      accent="text-slate-900"
                    />
                    <MetricCard
                      label="Avg Class Gap"
                      value={pct(postFlStats.avgGap)}
                      sub={gapLabel(postFlStats.avgGap)}
                      accent={
                        postFlStats.avgGap <= 0.1
                          ? "text-emerald-700"
                          : postFlStats.avgGap <= 0.2
                            ? "text-amber-700"
                            : "text-red-700"
                      }
                    />
                    <MetricCard
                      label="Avg F1 Score"
                      value={postFlStats.avgF1.toFixed(3)}
                    />
                    <MetricCard
                      label="Weakest Client"
                      value={pct(postFlStats.worst.acc)}
                      sub={postFlStats.worst.name}
                    />
                  </div>

                  {/* Comparison table */}
                  <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200">
                          <th className="py-3 px-4 text-left font-semibold text-slate-600">
                            Client
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            Accuracy
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            Class Gap
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            Leukemia Acc
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            Healthy Acc
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            F1
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            Precision
                          </th>
                          <th className="py-3 px-4 text-right font-semibold text-slate-600">
                            Recall
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {clients.map((c, idx) => {
                          const pf = clientMetricsMap.get(c.id)?.global
                            ?.post_fl;
                          if (!pf) return null;
                          const gap = pf.class_gap ?? 0;
                          return (
                            <tr
                              key={c.id}
                              className="border-b border-slate-100 hover:bg-slate-50"
                            >
                              <td className="py-3 px-4 font-medium text-slate-900 flex items-center gap-2">
                                <div
                                  className="w-2 h-2 rounded-full flex-shrink-0"
                                  style={{
                                    backgroundColor:
                                      CLIENT_COLORS[idx % CLIENT_COLORS.length],
                                  }}
                                />
                                <span>{c.client_name}</span>
                                <span className="text-xs text-slate-400">
                                  {c.model_type}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-right font-semibold text-slate-900">
                                {pct(pf.accuracy)}
                              </td>
                              <td className="py-3 px-4 text-right">
                                <span
                                  className={`text-xs font-medium px-2 py-0.5 rounded-full border ${gapBadge(gap)}`}
                                >
                                  {pct(gap)}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-right text-slate-700">
                                {pct(pf.leukemia_accuracy)}
                              </td>
                              <td className="py-3 px-4 text-right text-slate-700">
                                {pct(pf.healthy_accuracy)}
                              </td>
                              <td className="py-3 px-4 text-right text-slate-700">
                                {pf.f1_score.toFixed(3)}
                              </td>
                              <td className="py-3 px-4 text-right text-slate-700">
                                {pct(pf.precision)}
                              </td>
                              <td className="py-3 px-4 text-right text-slate-700">
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

              {/* ── Section: Individual Client Detail ── */}
              {selectedClient !== "all" && selectedCM && (
                <section>
                  <div className="flex items-center justify-between mb-5">
                    <h2 className="text-xl font-bold text-slate-900">
                      Post-FL Evaluation
                      <span className="text-xs font-normal text-slate-400 ml-2">
                        on balanced test set (50/50)
                      </span>
                    </h2>
                    <span className="text-sm font-medium text-slate-500">
                      {selectedInfo?.client_name}{" "}
                      <span className="text-slate-400">|</span>{" "}
                      {selectedInfo?.model_type}
                    </span>
                  </div>

                  {/* Metric cards — row 1 */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <MetricCard
                      label="Accuracy"
                      value={pct(selectedCM.global.post_fl.accuracy)}
                      accent="text-slate-900"
                    />
                    <MetricCard
                      label="Class Gap"
                      value={pct(selectedCM.global.post_fl.class_gap ?? 0)}
                      sub={gapLabel(selectedCM.global.post_fl.class_gap ?? 0)}
                      accent={
                        (selectedCM.global.post_fl.class_gap ?? 0) <= 0.1
                          ? "text-emerald-700"
                          : (selectedCM.global.post_fl.class_gap ?? 0) <= 0.2
                            ? "text-amber-700"
                            : "text-red-700"
                      }
                    />
                    <MetricCard
                      label="Leukemia Accuracy"
                      value={pct(selectedCM.global.post_fl.leukemia_accuracy)}
                      sub="Sensitivity"
                    />
                    <MetricCard
                      label="Healthy Accuracy"
                      value={pct(selectedCM.global.post_fl.healthy_accuracy)}
                      sub="Specificity"
                    />
                  </div>

                  {/* Metric cards — row 2 */}
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <MetricCard
                      label="Precision"
                      value={pct(selectedCM.global.post_fl.precision)}
                    />
                    <MetricCard
                      label="Recall"
                      value={pct(selectedCM.global.post_fl.recall)}
                    />
                    <MetricCard
                      label="F1 Score"
                      value={selectedCM.global.post_fl.f1_score.toFixed(3)}
                    />
                  </div>

                  {/* Confusion matrix + data heterogeneity */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-white p-6 rounded-2xl border border-slate-200">
                      <h3 className="text-sm font-semibold text-slate-700 mb-4">
                        Confusion Matrix
                      </h3>
                      <ConfusionMatrixCard
                        confusionMatrix={
                          selectedCM.global.post_fl.confusion_matrix
                        }
                        accuracy={selectedCM.global.post_fl.accuracy}
                      />
                    </div>

                    {selectedCM.data_heterogeneity && (
                      <div className="bg-white p-6 rounded-2xl border border-slate-200">
                        <h3 className="text-sm font-semibold text-slate-700 mb-4">
                          Data Partition
                        </h3>
                        <div className="space-y-4">
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">
                              Total Samples
                            </span>
                            <span className="font-semibold text-slate-900">
                              {selectedCM.data_heterogeneity.total_samples.toLocaleString()}
                            </span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Train / Val</span>
                            <span className="text-slate-700">
                              {selectedCM.data_heterogeneity.train_samples.toLocaleString()}{" "}
                              /{" "}
                              {selectedCM.data_heterogeneity.val_samples.toLocaleString()}
                            </span>
                          </div>

                          {/* Class bars */}
                          <div className="pt-2 space-y-3">
                            <div>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-red-600 font-medium">
                                  Leukemia (ALL)
                                </span>
                                <span className="text-slate-500">
                                  {selectedCM.data_heterogeneity.class_distribution.leukemia.toLocaleString()}{" "}
                                  (
                                  {selectedCM.data_heterogeneity.class_distribution.leukemia_pct.toFixed(
                                    1,
                                  )}
                                  %)
                                </span>
                              </div>
                              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-red-500 rounded-full"
                                  style={{
                                    width: `${selectedCM.data_heterogeneity.class_distribution.leukemia_pct}%`,
                                  }}
                                />
                              </div>
                            </div>
                            <div>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-emerald-600 font-medium">
                                  Healthy
                                </span>
                                <span className="text-slate-500">
                                  {selectedCM.data_heterogeneity.class_distribution.healthy.toLocaleString()}{" "}
                                  (
                                  {selectedCM.data_heterogeneity.class_distribution.healthy_pct.toFixed(
                                    1,
                                  )}
                                  %)
                                </span>
                              </div>
                              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-emerald-500 rounded-full"
                                  style={{
                                    width: `${selectedCM.data_heterogeneity.class_distribution.healthy_pct}%`,
                                  }}
                                />
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                            <span className="text-sm text-slate-500">
                              Imbalance Ratio
                            </span>
                            <span
                              className={`text-xs font-medium px-2.5 py-0.5 rounded-full border ${
                                selectedCM.data_heterogeneity.imbalance_ratio <=
                                1.5
                                  ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                  : selectedCM.data_heterogeneity
                                        .imbalance_ratio <= 5
                                    ? "bg-amber-50 text-amber-700 border-amber-200"
                                    : "bg-red-50 text-red-700 border-red-200"
                              }`}
                            >
                              {selectedCM.data_heterogeneity.imbalance_ratio.toFixed(
                                1,
                              )}
                              :1
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </section>
              )}
            </div>
          );
        })()
      )}
    </div>
  );
}
