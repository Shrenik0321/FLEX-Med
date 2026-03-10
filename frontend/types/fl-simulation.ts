// FL Simulation Types matching backend schema

export type SimulationStatus = "pending" | "running" | "completed" | "failed";

export interface TrainingConfig {
  dirichlet_alpha?: number;
  dirichlet_seed?: number;
  dirichlet_min_partition_size?: number;
  minority_boost?: number;
  focal_alpha?: number;
  focal_gamma?: number;
  consensus_momentum?: number;
  distill_weight_base?: number;
  distill_decay_rate?: number;
  train_loss_weight?: number;
  distill_loss_weight?: number;
}

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
  // Dirichlet Partitioning Configuration
  dirichlet_alpha?: number;
  dirichlet_seed?: number;
  dirichlet_min_partition_size?: number;
  // Training strategy snapshot (stored for reference)
  training_config?: TrainingConfig;
}

export interface RoundMetrics {
  round: number;
  avg_accuracy: number;
  avg_loss: number;
  avg_f1: number;
  avg_precision: number;
  avg_recall: number;
  avg_train_loss?: number; // Training loss from local training phase
  avg_val_loss?: number; // Validation loss from local training phase
  num_clients_trained: number;
  timestamp: string;
}

export interface AggregateMetrics {
  avg_accuracy: number;
  avg_loss: number;
  avg_precision: number;
  avg_recall: number;
  avg_f1: number;
  avg_specificity?: number;
  avg_roc_auc?: number;
  std_accuracy: number;
  num_clients: number;
}

export interface FLSimulationMetrics {
  aggregate: {
    // Note: pre_fl removed as evaluating untrained models gives meaningless ~50% accuracy
    post_fl: AggregateMetrics;
    improvement: {
      // Now calculated as Round 1 vs Final Round progression
      avg_accuracy: number;
      avg_loss: number;
      avg_precision?: number;
      avg_recall?: number;
      avg_f1?: number;
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
  configs: FLConfig;
  aggregate_metrics: FLSimulationMetrics; // JSONB stored as object (not string)
  status: SimulationStatus;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration?: number | null; // Duration in seconds
}

// Helper to parse metrics (handles both object and JSON string for backwards compatibility)
export function parseMetrics(
  metrics: FLSimulationMetrics | string | any,
): FLSimulationMetrics | null {
  // If already an object with expected structure
  if (metrics && typeof metrics === "object" && !Array.isArray(metrics)) {
    return metrics as FLSimulationMetrics;
  }

  // If it's a JSON string (backwards compatibility)
  if (typeof metrics === "string") {
    if (!metrics || metrics.trim() === "" || metrics === "{}") {
      return null;
    }
    try {
      return JSON.parse(metrics) as FLSimulationMetrics;
    } catch (error) {
      console.error("Failed to parse metrics:", error);
      return null;
    }
  }

  return null;
}

// Helper to format duration
export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
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
