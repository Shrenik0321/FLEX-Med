// FL Simulation Types matching backend schema

export type SimulationStatus = "pending" | "running" | "completed" | "failed";

export interface FLConfig {
  num_server_rounds: number;
  fraction_train: number;
  fraction_evaluate: number;
  local_epochs: number;
  lr: number;
  lr_decay: number;
  distill_lr: number;
  distill_epochs: number;
  temperature: number;
  batch_size: number;
}

export interface RoundMetrics {
  round: number;
  avg_accuracy: number;
  avg_loss: number;
  avg_f1: number;
  avg_precision: number;
  avg_recall: number;
  num_clients_trained: number;
  timestamp: string;
}

export interface AggregateMetrics {
  avg_accuracy: number;
  avg_loss: number;
  avg_precision: number;
  avg_recall: number;
  avg_f1: number;
  std_accuracy: number;
  num_clients: number;
}

export interface FLSimulationMetrics {
  aggregate: {
    pre_fl: AggregateMetrics;
    post_fl: AggregateMetrics;
    improvement: {
      avg_accuracy: number;
      avg_loss: number;
      avg_precision: number;
      avg_recall: number;
      avg_f1: number;
    };
  };
  rounds: RoundMetrics[];
  best_round: {
    round: number;
    avg_accuracy: number;
  };
  total_rounds_completed: number;
  total_clients: number;
}

export interface FLSimulation {
  id: number;
  client_ids: number[];
  configs: FLConfig;
  metrics: string; // JSON string (parse with parseMetrics helper)
  status: SimulationStatus;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration?: number | null; // Duration in seconds
}

// Helper to parse metrics JSON string
export function parseMetrics(metricsStr: string): FLSimulationMetrics | null {
  if (!metricsStr || metricsStr.trim() === "" || metricsStr === "{}") {
    return null;
  }
  try {
    return JSON.parse(metricsStr) as FLSimulationMetrics;
  } catch (error) {
    console.error("Failed to parse metrics:", error);
    return null;
  }
}

// Helper to format duration
export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

// Helper to format date
export function formatDateTime(dateString: string): string {
  const date = new Date(dateString);
  const day = String(date.getDate()).padStart(2, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const year = String(date.getFullYear()).slice(-2);
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${day}/${month}/${year} ${hours}:${minutes}`;
}

// API response type for start_fl
export interface StartFLResponse {
  message: string;
  simulation_id: number;
  orchestrator_response: any;
}
