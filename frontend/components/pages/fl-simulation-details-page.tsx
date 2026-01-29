"use client";

import { Users, Loader2, Clock, XCircle, AlertCircle } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import { FLSimulation, parseMetrics } from "@/types/fl-simulation";
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
import EvalCard from "@/components/ui/eval-card";
import { Loading } from "../ui/loading";

// Register Chart.js components
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
    pre_fl: any;
    post_fl: any;
    improvement: any;
  };
  rounds: Array<{
    round: number;
    validation: {
      loss: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      evaluated_at: string;
    };
  }>;
}

interface ConfusionMatrix {
  TP: number;
  FN: number;
  FP: number;
  TN: number;
}

// Color palette for different clients
const CLIENT_COLORS = [
  "#B80028", // Primary red
  "#3b82f6", // Blue
  "#10b981", // Green
  "#f59e0b", // Orange
  "#8b5cf6", // Purple
  "#ec4899", // Pink
  "#14b8a6", // Teal
  "#f97316", // Orange-red
];

/**
 * CONFUSION MATRIX COMPONENT
 * ==========================
 *
 * Renders a 2x2 confusion matrix with TP, FN, FP, TN values.
 * Similar to the Python implementation in fl_evaluation.py.
 */
interface ConfusionMatrixCardProps {
  title: string;
  confusionMatrix?: ConfusionMatrix;
  accuracy?: number;
}

function ConfusionMatrixCard({
  title,
  confusionMatrix,
  accuracy,
}: ConfusionMatrixCardProps) {
  if (!confusionMatrix) {
    return (
      <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">{title}</h3>
        <div className="flex items-center justify-center h-48 text-slate-400">
          No confusion matrix data available
        </div>
      </div>
    );
  }

  const { TP = 0, FN = 0, FP = 0, TN = 0 } = confusionMatrix;
  const maxVal = Math.max(TP, FN, FP, TN);

  // Calculate color intensity based on value
  const getColorIntensity = (value: number) => {
    if (maxVal === 0) return 0;
    return (value / maxVal) * 0.8 + 0.2; // 0.2 to 1.0 range
  };

  const getCellColor = (value: number) => {
    const intensity = getColorIntensity(value);
    const lightness = 100 - intensity * 50; // 50% to 100% lightness
    return `hsl(347, 91%, ${lightness}%)`;
  };

  const getTextColor = (value: number) => {
    const intensity = getColorIntensity(value);
    return intensity > 0.6 ? "text-white" : "text-slate-900";
  };

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-900 mb-4">{title}</h3>

      {/* Confusion Matrix Grid */}
      <div className="relative">
        {/* Column Labels */}
        <div className="grid grid-cols-2 gap-2 mb-2 ml-32">
          <div className="text-center text-sm font-medium text-slate-600">
            Predicted
            <br />
            Leukemia
          </div>
          <div className="text-center text-sm font-medium text-slate-600">
            Predicted
            <br />
            Healthy
          </div>
        </div>

        {/* Matrix with Row Labels */}
        <div className="flex gap-2">
          {/* Row Labels */}
          <div className="flex flex-col gap-2 justify-center w-28">
            <div className="h-24 flex items-center justify-end pr-3 text-sm font-medium text-slate-600 text-right">
              Actual
              <br />
              Leukemia
            </div>
            <div className="h-24 flex items-center justify-end pr-3 text-sm font-medium text-slate-600 text-right">
              Actual
              <br />
              Healthy
            </div>
          </div>

          {/* Matrix Cells */}
          <div className="grid grid-cols-2 gap-2 flex-1">
            {/* TP */}
            <div
              className={`h-24 flex flex-col items-center justify-center rounded-lg border border-slate-300 ${getTextColor(TP)}`}
              style={{ backgroundColor: getCellColor(TP) }}
            >
              <div className="text-3xl font-bold">{TP}</div>
              <div className="text-xs font-medium opacity-80">
                True Positive
              </div>
            </div>

            {/* FN */}
            <div
              className={`h-24 flex flex-col items-center justify-center rounded-lg border border-slate-300 ${getTextColor(FN)}`}
              style={{ backgroundColor: getCellColor(FN) }}
            >
              <div className="text-3xl font-bold">{FN}</div>
              <div className="text-xs font-medium opacity-80">
                False Negative
              </div>
            </div>

            {/* FP */}
            <div
              className={`h-24 flex flex-col items-center justify-center rounded-lg border border-slate-300 ${getTextColor(FP)}`}
              style={{ backgroundColor: getCellColor(FP) }}
            >
              <div className="text-3xl font-bold">{FP}</div>
              <div className="text-xs font-medium opacity-80">
                False Positive
              </div>
            </div>

            {/* TN */}
            <div
              className={`h-24 flex flex-col items-center justify-center rounded-lg border border-slate-300 ${getTextColor(TN)}`}
              style={{ backgroundColor: getCellColor(TN) }}
            >
              <div className="text-3xl font-bold">{TN}</div>
              <div className="text-xs font-medium opacity-80">
                True Negative
              </div>
            </div>
          </div>
        </div>

        {/* Accuracy Display */}
        {accuracy !== undefined && (
          <div className="mt-4 text-center">
            <span className="text-sm font-medium text-slate-600">
              Accuracy:{" "}
              <span className="text-slate-900 font-bold">
                {(accuracy * 100).toFixed(1)}%
              </span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * SMART Y-AXIS SCALING IMPLEMENTATION
 * ====================================
 *
 * This implementation mirrors the Python fl_evaluation.py smart scaling logic:
 *
 * Key Features:
 * 1. Dynamic range adjustment based on actual data values
 * 2. 15% padding for visual clarity (configurable)
 * 3. Metric-specific optimization:
 *    - Bounded metrics (accuracy, precision, recall, f1): 0-1 range with intelligent scaling
 *    - Unbounded metrics (loss): Open range with padding
 * 4. Minimum range enforcement to prevent flat graphs
 * 5. Pre-FL baseline included as Round 0 (matches Python's extract_validation_series)
 *
 * Changes from original implementation:
 * - Fixed round labeling: Now shows "Round 0, Round 1, ..." instead of "Round -1, Round 0, ..."
 * - Added Pre-FL baseline as Round 0 for all metrics
 * - Replaced fixed y-axis (0-1) with smart dynamic scaling
 * - Per-metric chart options instead of one-size-fits-all
 *
 * Similar to Weights & Biases auto-scaling behavior.
 */

// Smart y-axis scaling function (based on fl_evaluation.py)
function smartYLimit(
  values: number[],
  metricType: "bounded" | "unbounded",
  padding: number = 0.15,
): { min: number; max: number } {
  if (!values || values.length === 0) {
    return { min: 0, max: 1 };
  }

  // Filter out null/undefined/NaN values
  const validValues = values.filter((v) => v != null && !isNaN(v));
  if (validValues.length === 0) {
    return { min: 0, max: 1 };
  }

  const minVal = Math.min(...validValues);
  const maxVal = Math.max(...validValues);
  const valueRange = maxVal - minVal;

  if (metricType === "bounded") {
    // For metrics bounded between 0 and 1 (accuracy, precision, recall, f1)

    // If range is very small, ensure minimum visibility
    if (valueRange < 0.05) {
      const center = (maxVal + minVal) / 2;
      const yMin = Math.max(0, center - 0.05);
      const yMax = Math.min(1.0, center + 0.05);
      return { min: yMin, max: yMax };
    }

    // Add padding to the range
    const padAmount = valueRange * padding;

    // For high values (>0.6), we can start higher than 0
    let yMin: number;
    if (minVal > 0.6) {
      yMin = Math.max(0, minVal - padAmount);
    } else if (minVal > 0.3) {
      yMin = Math.max(0, minVal - padAmount * 1.5);
    } else {
      yMin = 0;
    }

    // Cap at 1.0 but add padding
    const yMax = Math.min(1.0, maxVal + padAmount);

    // Ensure we have at least 10% range for clarity
    if (yMax - yMin < 0.1) {
      const center = (yMax + yMin) / 2;
      return {
        min: Math.max(0, center - 0.05),
        max: Math.min(1.0, center + 0.05),
      };
    }

    return { min: yMin, max: yMax };
  } else {
    // For unbounded metrics (loss)
    if (valueRange < 0.01) {
      const center = (maxVal + minVal) / 2;
      return {
        min: Math.max(0, center - 0.01),
        max: center + 0.01,
      };
    }

    const padAmount = valueRange * padding;
    return {
      min: Math.max(0, minVal - padAmount),
      max: maxVal + padAmount,
    };
  }
}

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

  // Fetch simulation and client data once
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setLoadError(null);

        // Fetch simulation details
        const simResponse = await fetch(
          `${API_BASE_PATH}/fl_simulations/${simulationId}`,
        );
        if (!simResponse.ok) throw new Error("Failed to fetch simulation");

        const simData: FLSimulation = await simResponse.json();
        setSimulation(simData);

        // Only fetch client metrics if simulation is completed
        if (simData.status === "completed") {
          await fetchClientMetrics();
        }

        setIsLoading(false);
      } catch (error) {
        console.error("Error loading simulation:", error);
        setLoadError("Failed to load simulation data");
        toast.error("Failed to load simulation data");
        setIsLoading(false);
      }
    };

    const fetchClientMetrics = async () => {
      try {
        // Fetch client metrics from client_simulation_metrics table
        const metricsResponse = await fetch(
          `${API_BASE_PATH}/client-simulation-metrics/simulation/${simulationId}`,
        );

        if (!metricsResponse.ok) {
          throw new Error("Failed to fetch client metrics");
        }

        const clientMetricsData = await metricsResponse.json();

        // Transform data to match Client interface
        const clientsData: Client[] = clientMetricsData.map((record: any) => ({
          id: record.client_id,
          client_name:
            record.clients?.client_name || `Client ${record.client_id}`,
          model_type: record.clients?.model_type || "unknown",
          metrics: JSON.stringify(record.metrics),
        }));

        setClients(clientsData);
      } catch (error) {
        console.error("Error loading client metrics:", error);
        toast.error("Failed to load client metrics");
      }
    };

    fetchData();
  }, [simulationId]);

  // Parse simulation metrics
  const simulationMetrics = simulation
    ? parseMetrics(simulation.aggregate_metrics)
    : null;

  // Debug logging (can be removed in production)
  useEffect(() => {
    if (simulation) {
      console.log("Simulation status:", simulation.status);
      console.log("Raw aggregate_metrics:", simulation.aggregate_metrics);
      console.log("Parsed simulationMetrics:", simulationMetrics);
    }
  }, [simulation, simulationMetrics]);

  // (Conditional rendering moved to bottom to fix Rules of Hooks - see below)

  // If we reach here, simulation is completed - continue with existing rendering logic
  // Parse simulation metrics (already done above)

  // Continue with the rest of the component for completed simulations...

  // Parse client metrics
  // Parse client metrics
  const clientMetricsMap = new Map<number, ClientMetrics>();
  clients.forEach((client) => {
    try {
      const metrics = JSON.parse(client.metrics) as ClientMetrics;
      clientMetricsMap.set(client.id, metrics);
    } catch (e) {
      console.error(`Failed to parse metrics for client ${client.id}`);
    }
  });

  // Helper function to collect all values for a metric (for smart scaling)
  const collectAllValues = (
    metricKey: keyof ClientMetrics["rounds"][0]["validation"],
  ): number[] => {
    const allValues: number[] = [];

    if (selectedClient === "all") {
      // Collect from all clients
      clients.forEach((client) => {
        const metrics = clientMetricsMap.get(client.id);
        if (metrics) {
          // Include pre-FL baseline
          const preFl = metrics.global?.pre_fl?.[metricKey];
          if (preFl != null) {
            allValues.push(preFl as number);
          }

          // Include validation rounds
          metrics.rounds?.forEach((round) => {
            const value = round.validation[metricKey];
            if (value != null) {
              allValues.push(value as number);
            }
          });

          // Include post-FL
          const postFl = metrics.global?.post_fl?.[metricKey];
          if (postFl != null) {
            allValues.push(postFl as number);
          }
        }
      });
    } else {
      // Collect from selected client
      const clientId = parseInt(selectedClient);
      const metrics = clientMetricsMap.get(clientId);
      if (metrics) {
        const preFl = metrics.global?.pre_fl?.[metricKey];
        if (preFl != null) {
          allValues.push(preFl as number);
        }

        metrics.rounds?.forEach((round) => {
          const value = round.validation[metricKey];
          if (value != null) {
            allValues.push(value as number);
          }
        });

        const postFl = metrics.global?.post_fl?.[metricKey];
        if (postFl != null) {
          allValues.push(postFl as number);
        }
      }
    }

    return allValues;
  };

  // Prepare data for charts (includes Pre-FL baseline as round 0)
  const prepareChartData = () => {
    if (selectedClient === "all") {
      // Show aggregate data from simulation metrics
      if (!simulationMetrics?.rounds) return [];

      const data: any[] = [];

      // Add Pre-FL baseline as round 0 (aggregate across all clients)
      if (simulationMetrics.aggregate?.pre_fl) {
        data.push({
          round: 0,
          accuracy: simulationMetrics.aggregate.pre_fl.avg_accuracy,
          precision: simulationMetrics.aggregate.pre_fl.avg_precision,
          recall: simulationMetrics.aggregate.pre_fl.avg_recall,
          f1: simulationMetrics.aggregate.pre_fl.avg_f1,
          loss: simulationMetrics.aggregate.pre_fl.avg_loss,
        });
      }

      // Add validation rounds
      simulationMetrics.rounds.forEach((round) => {
        data.push({
          round: round.round,
          accuracy: round.avg_accuracy,
          precision: round.avg_precision,
          recall: round.avg_recall,
          f1: round.avg_f1,
          loss: round.avg_loss,
        });
      });

      return data;
    } else {
      // Show individual client data
      const clientId = parseInt(selectedClient);
      const clientMetrics = clientMetricsMap.get(clientId);
      if (!clientMetrics) return [];

      const data: any[] = [];

      // Add Pre-FL baseline as round 0
      if (clientMetrics.global?.pre_fl) {
        data.push({
          round: 0,
          accuracy: clientMetrics.global.pre_fl.accuracy,
          precision: clientMetrics.global.pre_fl.precision,
          recall: clientMetrics.global.pre_fl.recall,
          f1: clientMetrics.global.pre_fl.f1_score,
          loss: clientMetrics.global.pre_fl.loss,
        });
      }

      // Add validation rounds
      clientMetrics.rounds?.forEach((round) => {
        data.push({
          round: round.round,
          accuracy: round.validation.accuracy,
          precision: round.validation.precision,
          recall: round.validation.recall,
          f1: round.validation.f1_score,
          loss: round.validation.loss,
        });
      });

      return data;
    }
  };

  // Prepare multi-client data (when "all" is selected, show individual lines)
  const prepareMultiClientData = () => {
    if (selectedClient !== "all") return null;

    // Get max rounds from any client
    let maxRounds = 0;
    clients.forEach((client) => {
      const metrics = clientMetricsMap.get(client.id);
      if (metrics?.rounds) {
        maxRounds = Math.max(maxRounds, metrics.rounds.length);
      }
    });

    // Build data with one entry per round (including round 0 for Pre-FL)
    const data: any[] = [];

    // Round 0: Pre-FL baseline
    const round0Data: any = { round: 0 };
    clients.forEach((client) => {
      const metrics = clientMetricsMap.get(client.id);
      if (metrics?.global?.pre_fl) {
        round0Data[`accuracy_${client.id}`] = metrics.global.pre_fl.accuracy;
        round0Data[`precision_${client.id}`] = metrics.global.pre_fl.precision;
        round0Data[`recall_${client.id}`] = metrics.global.pre_fl.recall;
        round0Data[`f1_${client.id}`] = metrics.global.pre_fl.f1_score;
        round0Data[`loss_${client.id}`] = metrics.global.pre_fl.loss;
      }
    });
    data.push(round0Data);

    // Validation rounds
    for (let roundNum = 1; roundNum <= maxRounds; roundNum++) {
      const roundData: any = { round: roundNum };

      clients.forEach((client) => {
        const metrics = clientMetricsMap.get(client.id);
        const roundMetrics = metrics?.rounds.find((r) => r.round === roundNum);

        if (roundMetrics) {
          roundData[`accuracy_${client.id}`] = roundMetrics.validation.accuracy;
          roundData[`precision_${client.id}`] =
            roundMetrics.validation.precision;
          roundData[`recall_${client.id}`] = roundMetrics.validation.recall;
          roundData[`f1_${client.id}`] = roundMetrics.validation.f1_score;
          roundData[`loss_${client.id}`] = roundMetrics.validation.loss;
        }
      });

      data.push(roundData);
    }

    return data;
  };

  // Prepare Pre FL vs Post FL comparison
  const prepareComparisonData = () => {
    return clients.map((client) => {
      const metrics = clientMetricsMap.get(client.id);
      return {
        name: client.client_name,
        preFl: metrics?.global?.pre_fl?.accuracy
          ? metrics.global.pre_fl.accuracy
          : 0,
        postFl: metrics?.global?.post_fl?.accuracy
          ? metrics.global.post_fl.accuracy
          : 0,
      };
    });
  };

  const chartData = prepareChartData();
  const multiClientData = prepareMultiClientData();
  const comparisonData = prepareComparisonData();

  // Collect all values for smart scaling
  const allAccuracyValues = collectAllValues("accuracy");
  const allPrecisionValues = collectAllValues("precision");
  const allRecallValues = collectAllValues("recall");
  const allF1Values = collectAllValues("f1_score");
  const allLossValues = collectAllValues("loss");

  // Calculate smart y-axis limits
  const accuracyLimits = smartYLimit(allAccuracyValues, "bounded");
  const precisionLimits = smartYLimit(allPrecisionValues, "bounded");
  const recallLimits = smartYLimit(allRecallValues, "bounded");
  const f1Limits = smartYLimit(allF1Values, "bounded");
  const lossLimits = smartYLimit(allLossValues, "unbounded");

  // Calculate improvement values for smart scaling
  const allImprovementValues: number[] = [];
  if (selectedClient === "all" && multiClientData) {
    clients.forEach((client) => {
      const metrics = clientMetricsMap.get(client.id);
      const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

      multiClientData.forEach((d: any) => {
        const val = d[`accuracy_${client.id}`];
        if (typeof val === "number") {
          allImprovementValues.push(val - preFlAccuracy);
        }
      });
    });
  } else if (selectedClient !== "all") {
    // For single client, calculate improvement values
    const clientId = parseInt(selectedClient);
    const metrics = clientMetricsMap.get(clientId);
    const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

    chartData?.forEach((d: any) => {
      const val = d.accuracy;
      if (typeof val === "number") {
        allImprovementValues.push(val - preFlAccuracy);
      }
    });
  }
  const improvementLimits = smartYLimit(allImprovementValues, "unbounded", 0.1);

  // Calculate round-to-round delta values for smart scaling
  const allRoundDeltaValues: number[] = [];
  if (selectedClient === "all" && multiClientData) {
    const roundsData = multiClientData.slice(1); // Skip Round 0
    clients.forEach((client) => {
      const metrics = clientMetricsMap.get(client.id);
      const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

      roundsData.forEach((d: any, index: number) => {
        const currentAcc = d[`accuracy_${client.id}`];
        if (typeof currentAcc !== "number") return;

        if (index === 0) {
          // Round 1 vs Pre-FL
          allRoundDeltaValues.push(currentAcc - preFlAccuracy);
        } else {
          // Round N vs Round N-1
          const prevAcc = roundsData[index - 1][`accuracy_${client.id}`];
          if (typeof prevAcc === "number") {
            allRoundDeltaValues.push(currentAcc - prevAcc);
          }
        }
      });
    });
  } else if (selectedClient !== "all" && chartData) {
    const roundsData = chartData.slice(1); // Skip Round 0
    const clientId = parseInt(selectedClient);
    const metrics = clientMetricsMap.get(clientId);
    const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

    roundsData.forEach((d: any, index: number) => {
      const currentAcc = d.accuracy;
      if (typeof currentAcc !== "number") return;

      if (index === 0) {
        // Round 1 vs Pre-FL
        allRoundDeltaValues.push(currentAcc - preFlAccuracy);
      } else {
        // Round N vs Round N-1
        const prevAcc = roundsData[index - 1].accuracy;
        if (typeof prevAcc === "number") {
          allRoundDeltaValues.push(currentAcc - prevAcc);
        }
      }
    });
  }

  // Calculate limits allowing negative values (for deltas)
  const roundDeltaLimits = (() => {
    if (allRoundDeltaValues.length === 0) {
      return { min: -0.1, max: 0.1 };
    }

    const minVal = Math.min(...allRoundDeltaValues);
    const maxVal = Math.max(...allRoundDeltaValues);
    const valueRange = maxVal - minVal;

    if (valueRange < 0.01) {
      const center = (maxVal + minVal) / 2;
      return {
        min: center - 0.05,
        max: center + 0.05,
      };
    }

    const padding = 0.1;
    const padAmount = valueRange * padding;
    return {
      min: minVal - padAmount, // Allow negative values
      max: maxVal + padAmount,
    };
  })();

  // Charts configuration (with smart scaling dependencies)
  const accuracyChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`accuracy_${client.id}`]) ||
                [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
              borderWidth: 2,
            }))
          : [
              {
                label: "Accuracy",
                data: (chartData || []).map((d: any) => d.accuracy),
                borderColor: "#B80028",
                backgroundColor: "#B80028",
                tension: 0.3,
                pointRadius: 4,
                borderWidth: 2,
              },
            ],
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [chartData, multiClientData, selectedClient, clients],
  );

  const lossChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`loss_${client.id}`]) || [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
              borderWidth: 2,
            }))
          : [
              {
                label: "Loss",
                data: (chartData || []).map((d: any) => d.loss),
                borderColor: "#3b82f6",
                backgroundColor: "rgba(59, 130, 246, 0.1)",
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                borderWidth: 2,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const precisionChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`precision_${client.id}`]) ||
                [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
              borderWidth: 2,
            }))
          : [
              {
                label: "Precision",
                data: (chartData || []).map((d: any) => d.precision),
                borderColor: "#10b981",
                backgroundColor: "#10b981",
                tension: 0.3,
                pointRadius: 4,
                borderWidth: 2,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const recallChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`recall_${client.id}`]) ||
                [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
              borderWidth: 2,
            }))
          : [
              {
                label: "Recall",
                data: (chartData || []).map((d: any) => d.recall),
                borderColor: "#f59e0b",
                backgroundColor: "#f59e0b",
                tension: 0.3,
                pointRadius: 4,
                borderWidth: 2,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const f1ChartConfig = useMemo(
    () => ({
      labels: (multiClientData || chartData || []).map(
        (d: any) => `Round ${d.round}`,
      ),
      datasets:
        selectedClient === "all"
          ? clients.map((client, idx) => ({
              label: client.client_name,
              data:
                multiClientData?.map((d: any) => d[`f1_${client.id}`]) || [],
              borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
              tension: 0.3,
              pointRadius: 3,
              borderWidth: 2,
            }))
          : [
              {
                label: "F1 Score",
                data: (chartData || []).map((d: any) => d.f1),
                borderColor: "#8b5cf6",
                backgroundColor: "#8b5cf6",
                tension: 0.3,
                pointRadius: 4,
                borderWidth: 2,
              },
            ],
    }),
    [chartData, multiClientData, selectedClient, clients],
  );

  const comparisonChartConfig = useMemo(
    () => ({
      labels: (comparisonData || []).map((d) => d.name),
      datasets: [
        {
          label: "Post-FL Accuracy",
          data: (comparisonData || []).map((d) => d.postFl),
          backgroundColor: "rgba(184, 0, 40, 0.7)",
          borderColor: "#B80028",
          borderWidth: 1,
        },
        {
          label: "Pre-FL Accuracy",
          data: (comparisonData || []).map((d) => d.preFl),
          backgroundColor: "rgba(148, 163, 184, 0.6)",
          borderColor: "#94a3b8",
          borderWidth: 1,
        },
      ],
    }),
    [comparisonData],
  );

  const perClientImprovementChartConfig = useMemo(
    () => ({
      labels: (multiClientData || []).map((d: any) => `Round ${d.round}`),
      datasets: clients.map((client, idx) => {
        const metrics = clientMetricsMap.get(client.id);
        const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;
        return {
          label: client.client_name,
          data:
            multiClientData?.map((d: any) => {
              const val = d[`accuracy_${client.id}`];
              return typeof val === "number" ? val - preFlAccuracy : null;
            }) || [],
          borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
          backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
          tension: 0.3,
          pointRadius: 3,
          borderWidth: 2,
        };
      }),
    }),
    [multiClientData, clients, clientMetricsMap],
  );

  const singleClientImprovementChartConfig = useMemo(() => {
    if (selectedClient === "all") return null;

    const clientId = parseInt(selectedClient);
    const metrics = clientMetricsMap.get(clientId);
    const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

    return {
      labels: (chartData || []).map((d: any) => `Round ${d.round}`),
      datasets: [
        {
          label: "Accuracy Improvement",
          data: (chartData || []).map((d: any) => {
            const val = d.accuracy;
            return typeof val === "number" ? val - preFlAccuracy : null;
          }),
          borderColor: "#10b981", // Green for improvement
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          borderWidth: 2,
        },
      ],
    };
  }, [chartData, selectedClient, clientMetricsMap]);

  // Round-to-Round Improvement Chart (Delta from Previous Round)
  const perClientRoundDeltaChartConfig = useMemo(() => {
    if (selectedClient !== "all" || !multiClientData) return null;

    // Skip Round 0 (Pre-FL) - start from Round 1
    const roundsData = multiClientData.slice(1);

    return {
      labels: roundsData.map((d: any) => `Round ${d.round}`),
      datasets: clients.map((client, idx) => {
        const metrics = clientMetricsMap.get(client.id);
        const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

        // Calculate round-to-round deltas (starting from Round 1)
        const deltas = roundsData.map((d: any, index: number) => {
          const currentAcc = d[`accuracy_${client.id}`];
          if (typeof currentAcc !== "number") return null;

          if (index === 0) {
            // Round 1 - compare to Pre-FL
            return currentAcc - preFlAccuracy;
          } else {
            // Round 2+ - compare to previous round
            const prevAcc = roundsData[index - 1][`accuracy_${client.id}`];
            if (typeof prevAcc !== "number") return null;
            return currentAcc - prevAcc;
          }
        });

        return {
          label: client.client_name,
          data: deltas,
          borderColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
          backgroundColor: CLIENT_COLORS[idx % CLIENT_COLORS.length],
          tension: 0.3,
          pointRadius: 3,
          borderWidth: 2,
        };
      }),
    };
  }, [multiClientData, clients, clientMetricsMap, selectedClient]);

  const singleClientRoundDeltaChartConfig = useMemo(() => {
    if (selectedClient === "all" || !chartData) return null;

    const clientId = parseInt(selectedClient);
    const metrics = clientMetricsMap.get(clientId);
    const preFlAccuracy = metrics?.global?.pre_fl?.accuracy || 0;

    // Skip Round 0 (Pre-FL) - start from Round 1
    const roundsData = chartData.slice(1);

    // Calculate round-to-round deltas (starting from Round 1)
    const deltas = roundsData.map((d: any, index: number) => {
      const currentAcc = d.accuracy;
      if (typeof currentAcc !== "number") return null;

      if (index === 0) {
        // Round 1 - compare to Pre-FL
        return currentAcc - preFlAccuracy;
      } else {
        // Round 2+ - compare to previous round
        const prevAcc = roundsData[index - 1].accuracy;
        if (typeof prevAcc !== "number") return null;
        return currentAcc - prevAcc;
      }
    });

    return {
      labels: roundsData.map((d: any) => `Round ${d.round}`),
      datasets: [
        {
          label: "Round-to-Round Improvement",
          data: deltas,
          borderColor: "#3b82f6", // Blue for delta
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          tension: 0.3,
          fill: true,
          pointRadius: 4,
          borderWidth: 2,
        },
      ],
    };
  }, [chartData, selectedClient, clientMetricsMap]);

  // Get selected client metrics for individual view
  const selectedClientMetrics =
    selectedClient !== "all"
      ? clientMetricsMap.get(parseInt(selectedClient))
      : null;

  // Get selected client info
  const selectedClientInfo =
    selectedClient !== "all"
      ? clients.find((c) => c.id === parseInt(selectedClient))
      : null;

  // Base chart options
  const getChartOptions = (
    metricLimits: { min: number; max: number },
    isPercentage: boolean = false,
    isRoundBased: boolean = true,
  ) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: selectedClient === "all",
        position: "top" as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          font: { size: 11 },
        },
      },
      tooltip: {
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        titleColor: "#1e293b",
        bodyColor: "#1e293b",
        borderColor: "#e2e8f0",
        borderWidth: 1,
        padding: 12,
        boxPadding: 4,
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || "";
            const value = context.raw;
            if (value == null) return label;
            if (isPercentage) {
              return `${label}: ${(value * 100).toFixed(1)}%`;
            }
            return `${label}: ${value.toFixed(4)}`;
          },
          title: (context: any) => {
            const roundNum = context[0]?.label || "";
            // Add context for round 0
            if (roundNum === "Round 0") {
              return `${roundNum} (Pre-FL Baseline)`;
            }
            return roundNum;
          },
        },
      },
      annotation: {
        annotations:
          selectedClient !== "all" && selectedClientMetrics?.global?.post_fl
            ? {
                preFlLine: {
                  type: "line",
                  yMin: 0,
                  yMax: 0,
                  borderColor: "#1e293b",
                  borderWidth: 2.5,
                  borderDash: [5, 5],
                  label: {
                    display: true,
                    content: "Pre-FL",
                    position: "start",
                  },
                },
              }
            : {},
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        min: metricLimits.min,
        max: metricLimits.max,
        grid: {
          display: true,
          color: (context: any) =>
            Math.abs(context.tick.value) < 0.0001
              ? "rgba(30, 41, 59, 1)"
              : "rgba(0,0,0,0.05)",
          lineWidth: (context: any) =>
            Math.abs(context.tick.value) < 0.0001 ? 2 : 1,
        },
        ticks: {
          font: { size: 11 },
          callback: (v: any) => {
            if (isPercentage) {
              if (Math.abs(Number(v) - 1) < 0.0001) return "100%";
              return Number(v).toFixed(2);
            }
            return Number(v).toFixed(3);
          },
        },
      },
      x: {
        offset: false,
        grid: {
          display: isRoundBased, // Show vertical grid lines for round-based charts
          color: "rgba(0,0,0,0.05)",
        },
        ticks: {
          font: { size: 11 },
          // Custom ticking for rounds: 0, 2, 4, ...
          callback: isRoundBased
            ? function (this: any, val: any, index: any) {
                const label = this.getLabelForValue(val) as string;
                const roundNum = parseInt(label.replace("Round ", ""));

                if (!isNaN(roundNum)) {
                  return roundNum; // Show every round number
                }
                return label; // Fallback for other labels
              }
            : undefined,
        },
      },
    },
  });

  // Metric-specific chart options with smart scaling
  const accuracyChartOptions = getChartOptions(accuracyLimits, true);
  const precisionChartOptions = getChartOptions(precisionLimits, true);
  const recallChartOptions = getChartOptions(recallLimits, true);
  const f1ChartOptions = getChartOptions(f1Limits, true);
  const lossChartOptions = getChartOptions(lossLimits, false);
  const comparisonChartOptions = getChartOptions(accuracyLimits, true, false);

  // Improvement chart options with baseline reference line
  const improvementChartOptions = useMemo(() => {
    const baseOptions = getChartOptions(improvementLimits, true, true);
    return {
      ...baseOptions,
      plugins: {
        ...baseOptions.plugins,
        annotation: {
          annotations: {
            baselineLine: {
              type: "line" as const,
              yMin: 0,
              yMax: 0,
              borderColor: "#1e293b",
              borderWidth: 2.5,
              borderDash: [8, 4],
              label: {
                display: true,
                content: "Pre-FL Baseline (0% improvement)",
                position: "end" as const,
                backgroundColor: "#1e293b",
                color: "white",
                font: {
                  size: 10,
                  weight: "bold" as const,
                },
              },
            },
          },
        },
      },
    };
  }, [improvementLimits]);

  // Round-to-Round Delta chart options with zero reference line
  const roundDeltaChartOptions = useMemo(() => {
    const baseOptions = getChartOptions(roundDeltaLimits, true, true);
    return {
      ...baseOptions,
      plugins: {
        ...baseOptions.plugins,
        annotation: {
          annotations: {
            zeroLine: {
              type: "line" as const,
              yMin: 0,
              yMax: 0,
              borderColor: "#1e293b",
              borderWidth: 2.5,
              borderDash: [8, 4],
              label: {
                display: true,
                content: "No Change (0% delta)",
                position: "end" as const,
                backgroundColor: "#1e293b",
                color: "white",
                font: {
                  size: 10,
                  weight: "bold" as const,
                },
              },
            },
          },
        },
      },
    };
  }, [roundDeltaLimits]);

  // =====================================================================================
  // CONDITIONAL RENDERING (Moved here to ensure all Hooks run before early returns)
  // =====================================================================================

  // Render loading/running state
  if (isLoading && !simulation) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <Loader2 className="h-12 w-12 animate-spin text-primary" />
        <p className="mt-4 text-lg text-gray-600">Loading simulation...</p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <AlertCircle className="h-12 w-12 text-red-500" />
        <p className="mt-4 text-lg text-gray-600">{loadError}</p>
      </div>
    );
  }

  // Render running/pending status
  if (
    simulation &&
    (simulation.status === "running" || simulation.status === "pending")
  ) {
    return (
      <div className="p-8">
        <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 py-16">
          <div className="relative">
            <Loader2 className="h-16 w-16 animate-spin text-primary" />
            <Clock className="absolute right-0 top-0 h-6 w-6 text-blue-500" />
          </div>
          <h2 className="mt-6 text-2xl font-semibold text-gray-900">
            FL Simulation{" "}
            {simulation.status === "pending" ? "Starting" : "In Progress"}
          </h2>
          <p className="mt-2 text-center text-gray-600 max-w-md">
            {simulation.status === "pending"
              ? "The federated learning simulation is being initialized. This may take a moment..."
              : "The federated learning simulation is currently running. This may take several minutes depending on the configuration."}
          </p>

          <div className="mt-6 rounded-lg bg-blue-50 border border-blue-200 p-4 max-w-lg">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-blue-800">
                <p className="font-medium">
                  Metrics will be available once training completes
                </p>
                <p className="mt-1 text-blue-700">
                  Please refresh this page after the simulation finishes to view
                  the results. The simulation runs in the background and may
                  take 10-30 minutes depending on your configuration.
                </p>
              </div>
            </div>
          </div>

          {simulation.configs && (
            <div className="mt-8 w-full max-w-md">
              <h3 className="text-sm font-medium text-gray-700">
                Configuration:
              </h3>
              <dl className="mt-2 grid grid-cols-2 gap-2 text-sm">
                <div>
                  <dt className="text-gray-500">Rounds:</dt>
                  <dd className="font-medium">
                    {simulation.configs.num_server_rounds}
                  </dd>
                </div>
                <div>
                  <dt className="text-gray-500">Local Epochs:</dt>
                  <dd className="font-medium">
                    {simulation.configs.local_epochs}
                  </dd>
                </div>
                <div>
                  <dt className="text-gray-500">Learning Rate:</dt>
                  <dd className="font-medium">{simulation.configs.lr}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Batch Size:</dt>
                  <dd className="font-medium">
                    {simulation.configs.batch_size}
                  </dd>
                </div>
              </dl>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Render failed status
  if (simulation && simulation.status === "failed") {
    return (
      <div className="p-8">
        <div className="flex flex-col items-center justify-center rounded-lg border-2 border-red-200 bg-red-50 py-16">
          <XCircle className="h-16 w-16 text-red-500" />
          <h2 className="mt-6 text-2xl font-semibold text-gray-900">
            Simulation Failed
          </h2>
          <p className="mt-2 text-center text-gray-600 max-w-md">
            The federated learning simulation encountered an error and could not
            complete.
          </p>
          {simulation.error_message && (
            <div className="mt-4 max-w-md rounded bg-white p-4 text-sm text-gray-700">
              <p className="font-medium">Error details:</p>
              <p className="mt-1">{simulation.error_message}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (isLoading) {
    return <Loading fullScreen text="Synchronizing client database..." />;
  }

  if (!simulation) {
    return (
      <div className="p-8">
        <div className="text-center text-muted-foreground">
          Simulation not found
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <div className="mb-8 flex items-start justify-between">
        <div>
          <p className="text-slate-500 text-sm">
            Simulation ID:{" "}
            <span className="font-mono text-slate-700 bg-slate-100 px-1 py-0.5 rounded">
              {simulationId}
            </span>{" "}
            •{" "}
            <span className="font-medium text-slate-700">{clients.length}</span>{" "}
            Clients •{" "}
            <span
              className={`font-medium ${simulation?.status === "completed" ? "text-green-600" : "text-slate-700"}`}
            >
              {simulation?.status
                ? simulation.status.charAt(0).toUpperCase() +
                  simulation.status.slice(1)
                : "Unknown"}
            </span>
          </p>
        </div>

        {/* Client Selector */}
        <div className="flex items-center gap-3">
          <Users size={18} className="text-muted-foreground" />
          <Select value={selectedClient} onValueChange={setSelectedClient}>
            <SelectTrigger className="w-[280px] h-10 font-medium">
              <SelectValue placeholder="Select client" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="font-medium">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500"></div>
                  <span>All Clients (Multi-Line View)</span>
                </div>
              </SelectItem>
              <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Individual Clients
              </div>
              {clients.map((client, idx) => (
                <SelectItem
                  key={client.id}
                  value={client.id.toString()}
                  className="pl-6"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor:
                          CLIENT_COLORS[idx % CLIENT_COLORS.length],
                      }}
                    ></div>
                    <span>
                      {client.client_name}{" "}
                      <span className="text-xs text-muted-foreground">
                        ({client.model_type})
                      </span>
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Individual Client Summary (when specific client selected) */}
      {selectedClient !== "all" && selectedClientMetrics && (
        <div className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
              Client Performance Summary
            </h2>
            <div className="px-4 py-1.5 bg-slate-100 rounded-full text-sm font-medium text-slate-600">
              {selectedClientInfo?.client_name}{" "}
              <span className="text-slate-400">|</span>{" "}
              <span className="text-slate-500">
                {selectedClientInfo?.model_type}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
            {/* Pre-FL Accuracy */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Pre-FL Baseline
              </div>
              <div className="text-3xl font-bold text-slate-700">
                {(
                  (selectedClientMetrics.global?.pre_fl?.accuracy || 0) * 100
                ).toFixed(1)}
                <span className="text-lg text-slate-400 ml-1">%</span>
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                Initial Accuracy
              </div>
            </div>

            {/* Post-FL Accuracy */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow ring-1 ring-green-100">
              <div className="text-xs font-semibold text-green-600 uppercase tracking-wider mb-2">
                Post-FL Result
              </div>
              <div className="text-3xl font-bold text-slate-900">
                {(
                  (selectedClientMetrics.global?.post_fl?.accuracy || 0) * 100
                ).toFixed(1)}
                <span className="text-lg text-slate-400 ml-1">%</span>
              </div>
              <div className="text-sm text-green-600 mt-2 font-medium">
                Final Accuracy
              </div>
            </div>

            {/* Best Round */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Peak Performance
              </div>
              <div className="text-3xl font-bold text-purple-600">
                Round{" "}
                {selectedClientMetrics.rounds?.reduce(
                  (best, round) =>
                    round.validation.accuracy > (best?.validation.accuracy || 0)
                      ? round
                      : best,
                  selectedClientMetrics.rounds[0],
                )?.round || "N/A"}
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                {(
                  (selectedClientMetrics.rounds?.reduce(
                    (best, round) =>
                      round.validation.accuracy >
                      (best?.validation.accuracy || 0)
                        ? round
                        : best,
                    selectedClientMetrics.rounds[0],
                  )?.validation.accuracy || 0) * 100
                ).toFixed(1)}
                % Accuracy
              </div>
            </div>

            {/* Total Rounds */}
            <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Duration
              </div>
              <div className="text-3xl font-bold text-slate-700">
                {selectedClientMetrics.rounds?.length || 0}
              </div>
              <div className="text-sm text-slate-400 mt-2 font-medium">
                Rounds Completed
              </div>
            </div>
          </div>

          {/* Additional Metrics Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-6">
            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  F1 Score
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(
                    selectedClientMetrics.global?.post_fl?.f1_score || 0
                  ).toFixed(3)}
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.f1_score || 0) >=
                  0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.f1_score || 0) >= 0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.f1_score || 0) *
                  100
                ).toFixed(1)}
                %
              </div>
            </div>

            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  Precision
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(
                    (selectedClientMetrics.global?.post_fl?.precision || 0) *
                    100
                  ).toFixed(1)}
                  %
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.precision || 0) >=
                  0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.precision || 0) >=
                0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.precision || 0) *
                  100
                ).toFixed(1)}
                %
              </div>
            </div>

            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  Recall
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(
                    (selectedClientMetrics.global?.post_fl?.recall || 0) * 100
                  ).toFixed(1)}
                  %
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.recall || 0) >= 0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.recall || 0) >= 0
                  ? "+"
                  : ""}
                {(
                  (selectedClientMetrics.global?.improvement?.recall || 0) * 100
                ).toFixed(1)}
                %
              </div>
            </div>

            <div className="bg-slate-50/50 p-4 rounded-xl border border-slate-100 flex items-center justify-between">
              <div>
                <div className="text-xs font-semibold text-slate-500 mb-1 uppercase tracking-wide">
                  Loss
                </div>
                <div className="text-xl font-bold text-slate-900">
                  {(selectedClientMetrics.global?.post_fl?.loss || 0).toFixed(
                    4,
                  )}
                </div>
              </div>
              <div
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  (selectedClientMetrics.global?.improvement?.loss || 0) <= 0
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {(selectedClientMetrics.global?.improvement?.loss || 0).toFixed(
                  4,
                )}
              </div>
            </div>
          </div>

          {/* Confusion Matrix Analysis */}
          <div className="mt-8 border-t border-slate-100 pt-8">
            <h3 className="text-lg font-bold text-slate-900 mb-6">
              Confusion Matrix Analysis
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <ConfusionMatrixCard
                title="Pre-FL Baseline"
                confusionMatrix={
                  selectedClientMetrics.global?.pre_fl?.confusion_matrix
                }
                accuracy={selectedClientMetrics.global?.pre_fl?.accuracy}
              />
              <ConfusionMatrixCard
                title="Post-FL Result"
                confusionMatrix={
                  selectedClientMetrics.global?.post_fl?.confusion_matrix
                }
                accuracy={selectedClientMetrics.global?.post_fl?.accuracy}
              />
            </div>
          </div>
        </div>
      )}

      {/* View Mode Indicator */}
      <div className="mb-4 flex items-center gap-2 text-sm text-slate-500">
        {selectedClient === "all" ? (
          <>
            <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500"></div>
            <span>
              Viewing{" "}
              <strong className="text-slate-900 font-semibold">
                all clients
              </strong>{" "}
              with individual trend lines per client
            </span>
          </>
        ) : (
          <>
            <div
              className="w-2 h-2 rounded-full"
              style={{
                backgroundColor:
                  CLIENT_COLORS[
                    clients.findIndex(
                      (c) => c.id === parseInt(selectedClient),
                    ) % CLIENT_COLORS.length
                  ],
              }}
            ></div>
            <span>
              Viewing{" "}
              <strong className="text-slate-900 font-semibold">
                {selectedClientInfo?.client_name}
              </strong>{" "}
              individual performance
              {" • "}Round 0 = Pre-FL baseline
            </span>
          </>
        )}
      </div>

      {/* Statistics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-6">
        {/* Accuracy Progression Chart */}
        <EvalCard
          title="Accuracy"
          description={
            selectedClient === "all"
              ? "Multi-client accuracy progression (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} accuracy over time`
          }
          content={
            <Line data={accuracyChartConfig} options={accuracyChartOptions} />
          }
        />

        {/* Loss Progression Chart */}
        <EvalCard
          title="Loss"
          description={
            selectedClient === "all"
              ? "Multi-client loss convergence (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} loss convergence`
          }
          content={<Line data={lossChartConfig} options={lossChartOptions} />}
        />

        {/* Precision Chart */}
        <EvalCard
          title="Precision"
          description={
            selectedClient === "all"
              ? "Multi-client precision trends (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} precision progression`
          }
          content={
            <Line data={precisionChartConfig} options={precisionChartOptions} />
          }
        />

        {/* Recall Chart */}
        <EvalCard
          title="Recall"
          description={
            selectedClient === "all"
              ? "Multi-client recall/sensitivity (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} recall progression`
          }
          content={
            <Line data={recallChartConfig} options={recallChartOptions} />
          }
        />

        {/* F1 Score Chart */}
        <EvalCard
          title="F1 Score"
          description={
            selectedClient === "all"
              ? "Multi-client F1 score trends (Round 0 = Pre-FL)"
              : `${selectedClientInfo?.client_name} F1 score progression`
          }
          content={<Line data={f1ChartConfig} options={f1ChartOptions} />}
        />

        {/* Round-to-Round Improvement Delta */}
        {selectedClient === "all" ? (
          <EvalCard
            title="Round-to-Round Improvement"
            description="Accuracy gain from previous round (shows diminishing returns and convergence)"
            content={
              perClientRoundDeltaChartConfig ? (
                <Line
                  data={perClientRoundDeltaChartConfig}
                  options={roundDeltaChartOptions}
                />
              ) : (
                <div className="flex items-center justify-center h-48 text-slate-400">
                  No delta data available
                </div>
              )
            }
          />
        ) : (
          <EvalCard
            title="Round-to-Round Improvement"
            description={`${selectedClientInfo?.client_name} accuracy gain from previous round`}
            content={
              singleClientRoundDeltaChartConfig ? (
                <Line
                  data={singleClientRoundDeltaChartConfig}
                  options={roundDeltaChartOptions}
                />
              ) : (
                <div className="flex items-center justify-center h-48 text-slate-400">
                  No delta data available
                </div>
              )
            }
          />
        )}

        {/* Pre FL vs Post FL Comparison */}
        <EvalCard
          title="FL Impact"
          description={
            selectedClient === "all"
              ? "Pre-FL vs Post-FL comparison across all clients"
              : "Pre-FL baseline vs Post-FL final performance"
          }
          content={
            <Bar
              data={comparisonChartConfig}
              options={comparisonChartOptions}
            />
          }
        />
      </div>
    </div>
  );
}
