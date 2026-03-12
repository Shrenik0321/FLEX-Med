// Simulation types and helpers
export interface FLSimulation {
  id: number;
  run_name?: string; // Optional as it might not be in the backend yet
  configs: any;
  aggregate_metrics: any;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  started_at?: string;
  completed_at?: string;
  duration?: number;
  error_message?: string;
}

export interface StartFLResponse {
  message: string;
  simulation_id: number;
  orchestrator_response?: any;
}

/**
 * Safely parse metrics from JSON string or return the object if already parsed
 */
export function parseMetrics(metrics: any) {
  if (!metrics) return null;
  if (typeof metrics === "string") {
    try {
      return JSON.parse(metrics);
    } catch (e) {
      console.error("Failed to parse metrics string:", e);
      return null;
    }
  }
  return metrics;
}

/**
 * Format duration in seconds to human readable string (e.g., 2m 30s)
 */
export function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return "0s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

/**
 * Format ISO datetime string to a more readable format
 */
export function formatDateTime(dateStr: string): string {
  if (!dateStr) return "—";
  try {
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  } catch (e) {
    return dateStr;
  }
}
