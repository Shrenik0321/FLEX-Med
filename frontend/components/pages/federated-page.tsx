"use client";

import { useState } from "react";
import { Brain, Activity, Clock, Play, Shield } from "lucide-react";

export default function FederatedPage() {
  const [isStarting, setIsStarting] = useState(false);
  const [lastRun, setLastRun] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "error">("idle");

  const handleStart = async () => {
    try {
      setIsStarting(true);
      setStatus("running");

      // We trigger the API which internal fetches clients from DB
      await fetch("http://localhost:8000/api/start_fl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      setLastRun(new Date().toLocaleString());
    } catch (error) {
      console.error("Failed to start federated learning:", error);
      setStatus("error");
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <h1 className="text-3xl font-semibold text-foreground flex items-center gap-3">
            <Brain className="text-primary" />
            Federated Learning Orchestrator
          </h1>
          <p className="text-muted-foreground mt-2 max-w-2xl">
            Coordinate privacy-preserving training across your hospital clients.
            Monitor the global model status and trigger new federated rounds.
          </p>
        </div>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card rounded-lg p-5 shadow-sm border border-border">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-2">
              <span className="flex-red-dot" />
              Orchestrator Status
            </p>
            <Activity className="text-primary" size={18} />
          </div>
          <p className="text-xl font-semibold text-foreground capitalize">
            {status}
          </p>
          {lastRun && (
            <p className="text-xs text-muted-foreground mt-1">Last run: {lastRun}</p>
          )}
        </div>

        <div className="bg-card rounded-lg p-5 shadow-sm border border-border">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-2 mb-2">
            <span className="flex-red-dot" />
            Participating Sites
          </p>
          <p className="text-xl font-semibold text-foreground">3 Hospitals</p>
          <p className="text-xs text-muted-foreground mt-1">
            Configured via FLEX-Med clients table.
          </p>
        </div>

        <div className="bg-card rounded-lg p-5 shadow-sm border border-border">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide flex items-center gap-2 mb-2">
            <span className="flex-red-dot" />
            Compliance
          </p>
          <div className="flex items-center gap-2">
            <Shield className="text-emerald-500" size={18} />
            <p className="text-sm font-medium text-foreground">
              Data stays on-premise
            </p>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Only model updates leave local sites.
          </p>
        </div>
      </div>

      {/* Control + timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control panel */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border lg:col-span-1">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Play size={18} className="text-primary" />
            Control Panel
          </h2>
          <p className="text-sm text-muted-foreground mb-4">
            Trigger a new federated learning round using the current client
            configuration.
          </p>
          <button
            onClick={handleStart}
            disabled={isStarting}
            className="px-5 py-2 bg-card border border-primary text-primary rounded-lg text-sm font-medium hover:bg-primary/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Play size={18} />
            {isStarting ? "Starting round..." : "Start Federated Learning"}
          </button>
          {status === "error" && (
            <p className="mt-3 text-xs text-red-600">
              Something went wrong starting the run. Check the backend service
              and try again.
            </p>
          )}
        </div>

        {/* Timeline / logs */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <Clock size={18} className="text-primary" />
              Recent Federated Rounds
            </h2>
          </div>
          <div className="space-y-3">
            {[
              {
                round: "Round #5",
                status: "Completed",
                detail: "Global AUC improved to 0.94",
                time: "Today, 10:12",
              },
              {
                round: "Round #4",
                status: "Completed",
                detail: "3/3 clients contributed updates",
                time: "Yesterday, 18:45",
              },
              {
                round: "Round #3",
                status: "Degraded",
                detail: "One client dropped due to connectivity",
                time: "Yesterday, 09:30",
              },
            ].map((item, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3 pb-3 border-b border-border last:border-0"
              >
                <div
                  className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                    item.status === "Completed"
                      ? "bg-emerald-500"
                      : item.status === "Degraded"
                      ? "bg-amber-500"
                      : "bg-primary"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground flex items-center gap-2">
                    {item.round}
                    <span className="text-xs font-normal text-muted-foreground">
                      {item.status}
                    </span>
                  </p>
                  <p className="text-xs text-muted-foreground">{item.detail}</p>
                  <p className="text-[11px] text-muted-foreground/70 mt-1">{item.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
