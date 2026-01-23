"use client";

import { ArrowLeft, Users } from "lucide-react";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import {
  FLSimulation,
  parseMetrics,
  FLSimulationMetrics,
} from "@/types/fl-simulation";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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

export default function FLSimulationDetailsPage({
  simulationId,
  simulationName,
  onBack,
}: FLSimulationDetailsPageProps) {
  const [simulation, setSimulation] = useState<FLSimulation | null>(null);
  const [clients, setClients] = useState<Client[]>([]);
  const [selectedClient, setSelectedClient] = useState<string>("all");
  const [isLoading, setIsLoading] = useState(true);

  // Fetch simulation and client data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);

        // Fetch simulation
        const simResponse = await fetch(
          `${API_BASE_PATH}/fl_simulations/${simulationId}`,
        );
        if (!simResponse.ok) throw new Error("Failed to fetch simulation");
        const simData: FLSimulation = await simResponse.json();
        setSimulation(simData);

        // Fetch clients data
        const clientIds = simData.client_ids;
        const clientPromises = clientIds.map((id) =>
          fetch(`${API_BASE_PATH}/clients/${id}`).then((res) => res.json()),
        );
        const clientsData = await Promise.all(clientPromises);
        setClients(clientsData);
      } catch (error) {
        console.error("Error fetching data:", error);
        toast.error("Failed to load simulation data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [simulationId]);

  // Parse simulation metrics
  const simulationMetrics = simulation
    ? parseMetrics(simulation.metrics)
    : null;

  // Debug logging
  useEffect(() => {
    if (simulation) {
      console.log("Raw metrics string:", simulation.metrics);
      console.log("Parsed simulationMetrics:", simulationMetrics);
      console.log("Has rounds?", simulationMetrics?.rounds?.length);
    }
  }, [simulation, simulationMetrics]);

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

  // Prepare data for charts
  const prepareChartData = () => {
    if (selectedClient === "all") {
      // Show aggregate data from simulation metrics
      if (!simulationMetrics?.rounds) return [];

      return simulationMetrics.rounds.map((round) => ({
        round: round.round,
        accuracy: round.avg_accuracy * 100,
        precision: round.avg_precision * 100,
        recall: round.avg_recall * 100,
        f1: round.avg_f1 * 100,
        loss: round.avg_loss,
      }));
    } else {
      // Show individual client data
      const clientId = parseInt(selectedClient);
      const clientMetrics = clientMetricsMap.get(clientId);
      if (!clientMetrics?.rounds) return [];

      return clientMetrics.rounds.map((round) => ({
        round: round.round,
        accuracy: round.validation.accuracy * 100,
        precision: round.validation.precision * 100,
        recall: round.validation.recall * 100,
        f1: round.validation.f1_score * 100,
        loss: round.validation.loss,
      }));
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

    // Build data with one entry per round
    const data: any[] = [];
    for (let roundNum = 1; roundNum <= maxRounds; roundNum++) {
      const roundData: any = { round: roundNum };

      clients.forEach((client, idx) => {
        const metrics = clientMetricsMap.get(client.id);
        const roundMetrics = metrics?.rounds.find((r) => r.round === roundNum);

        if (roundMetrics) {
          roundData[`accuracy_${client.id}`] =
            roundMetrics.validation.accuracy * 100;
          roundData[`precision_${client.id}`] =
            roundMetrics.validation.precision * 100;
          roundData[`recall_${client.id}`] =
            roundMetrics.validation.recall * 100;
          roundData[`f1_${client.id}`] = roundMetrics.validation.f1_score * 100;
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
          ? metrics.global.pre_fl.accuracy * 100
          : 0,
        postFl: metrics?.global?.post_fl?.accuracy
          ? metrics.global.post_fl.accuracy * 100
          : 0,
      };
    });
  };

  const chartData = prepareChartData();
  const multiClientData = prepareMultiClientData();
  const comparisonData = prepareComparisonData();

  if (isLoading) {
    return (
      <div className="p-8">
        <div className="text-center text-muted-foreground">
          Loading simulation data...
        </div>
      </div>
    );
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

  // Check if metrics are empty
  const hasMetrics = simulationMetrics && simulationMetrics.rounds && simulationMetrics.rounds.length > 0;

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          Back to History
        </button>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-semibold text-foreground">
              {simulationName}
            </h1>
            <p className="text-muted-foreground mt-2">
              Simulation ID: {simulationId} • {clients.length} Clients •{" "}
              {simulation.status.charAt(0).toUpperCase() +
                simulation.status.slice(1)}
            </p>
          </div>

          {/* Client Selector */}
          <div className="flex items-center gap-3">
            <Users size={18} className="text-muted-foreground" />
            <Select value={selectedClient} onValueChange={setSelectedClient}>
              <SelectTrigger className="w-[250px]">
                <SelectValue placeholder="Select client" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Clients (Aggregate)</SelectItem>
                {clients.map((client) => (
                  <SelectItem key={client.id} value={client.id.toString()}>
                    {client.client_name} ({client.model_type})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      {/* Empty State Warning */}
      {!hasMetrics && (
        <div className="mb-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <div className="text-yellow-600">⚠️</div>
            <div>
              <h3 className="font-semibold text-yellow-800 mb-1">
                No Metrics Available
              </h3>
              <p className="text-sm text-yellow-700">
                This simulation doesn't have metrics data yet. Metrics are
                populated after the FL simulation completes. Current status:{" "}
                <span className="font-medium">
                  {simulation.status.charAt(0).toUpperCase() +
                    simulation.status.slice(1)}
                </span>
              </p>
              <p className="text-xs text-yellow-600 mt-2">
                Check the console for raw metrics data to debug.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Grid - 3x2 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Accuracy Chart */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            Accuracy Over Rounds
            {selectedClient === "all" && " (Per Client)"}
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={multiClientData || chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 12 }}
                label={{ value: "Round", position: "insideBottom", offset: -5 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={[0, 100]}
                label={{ value: "%", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              {selectedClient === "all" ? (
                <>
                  <Legend />
                  {clients.map((client, idx) => (
                    <Line
                      key={client.id}
                      type="monotone"
                      dataKey={`accuracy_${client.id}`}
                      name={client.client_name}
                      stroke={CLIENT_COLORS[idx % CLIENT_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#B80028"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Precision Chart */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            Precision Over Rounds
            {selectedClient === "all" && " (Per Client)"}
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={multiClientData || chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 12 }}
                label={{ value: "Round", position: "insideBottom", offset: -5 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={[0, 100]}
                label={{ value: "%", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              {selectedClient === "all" ? (
                <>
                  <Legend />
                  {clients.map((client, idx) => (
                    <Line
                      key={client.id}
                      type="monotone"
                      dataKey={`precision_${client.id}`}
                      name={client.client_name}
                      stroke={CLIENT_COLORS[idx % CLIENT_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="precision"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* F1 Score Chart */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            F1 Score Over Rounds
            {selectedClient === "all" && " (Per Client)"}
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={multiClientData || chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 12 }}
                label={{ value: "Round", position: "insideBottom", offset: -5 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={[0, 100]}
                label={{ value: "%", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              {selectedClient === "all" ? (
                <>
                  <Legend />
                  {clients.map((client, idx) => (
                    <Line
                      key={client.id}
                      type="monotone"
                      dataKey={`f1_${client.id}`}
                      name={client.client_name}
                      stroke={CLIENT_COLORS[idx % CLIENT_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="f1"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recall Chart */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            Recall Over Rounds
            {selectedClient === "all" && " (Per Client)"}
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={multiClientData || chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 12 }}
                label={{ value: "Round", position: "insideBottom", offset: -5 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={[0, 100]}
                label={{ value: "%", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              {selectedClient === "all" ? (
                <>
                  <Legend />
                  {clients.map((client, idx) => (
                    <Line
                      key={client.id}
                      type="monotone"
                      dataKey={`recall_${client.id}`}
                      name={client.client_name}
                      stroke={CLIENT_COLORS[idx % CLIENT_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="recall"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Loss Chart */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            Loss Over Rounds
            {selectedClient === "all" && " (Per Client)"}
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={multiClientData || chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="round"
                tick={{ fontSize: 12 }}
                label={{ value: "Round", position: "insideBottom", offset: -5 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={[0, 1]}
                label={{ value: "Loss", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              {selectedClient === "all" ? (
                <>
                  <Legend />
                  {clients.map((client, idx) => (
                    <Line
                      key={client.id}
                      type="monotone"
                      dataKey={`loss_${client.id}`}
                      name={client.client_name}
                      stroke={CLIENT_COLORS[idx % CLIENT_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Pre FL vs Post FL Comparison */}
        <div className="bg-card rounded-lg shadow-sm border border-border p-4">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            Pre FL vs Post FL Accuracy
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis
                tick={{ fontSize: 12 }}
                domain={[0, 100]}
                label={{ value: "%", angle: -90, position: "insideLeft" }}
              />
              <Tooltip />
              <Legend />
              <Bar dataKey="preFl" fill="#94a3b8" name="Pre FL" />
              <Bar dataKey="postFl" fill="#B80028" name="Post FL" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
