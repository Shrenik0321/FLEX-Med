"use client";

import { useState } from "react";
import { Brain, Activity, Clock, Play, Shield } from "lucide-react";

export default function FederatedPage() {
  const [isStarting, setIsStarting] = useState(false);
  const [lastRun, setLastRun] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "running" | "error">("idle");

  // Static DB Data to simulate the database fetch
  const activeClients = [
    {
      client_name: "Clinic - A",
      client_email: "s.shrenikdeep@gmail.com",
      status: "Training",
      model_type: "resnet18",
      id: 0, // Changed to 0 to match Flower's 0-indexed partitions
    },
    {
      client_name: "Clinic - B",
      client_email: "s.shrenikdeep@gmail.com",
      status: "Active",
      model_type: "mobilenet_v2",
      id: 1,
    },
    {
      client_name: "Clinic - C (Free Rider)",
      client_email: "freerider@clinic.com",
      status: "Active",
      model_type: "densenet121",
      id: 2,
    },
  ];

  const handleStart = async () => {
    try {
      setIsStarting(true);
      setStatus("running");

      // We pass the activeClients list to the API
      await fetch("https://426a2dbb35c1.ngrok-free.app/start_fl", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(activeClients),
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
          <h1 className="text-3xl font-semibold text-gray-900 flex items-center gap-3">
            <Brain className="text-[#B80028]" />
            Federated Learning Orchestrator
          </h1>
          <p className="text-[#718096] mt-2 max-w-2xl">
            Coordinate privacy-preserving training across your hospital clients.
            Monitor the global model status and trigger new federated rounds.
          </p>
        </div>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg p-5 flex-card-shadow">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-medium text-[#718096] uppercase tracking-wide flex items-center gap-2">
              <span className="flex-red-dot" />
              Orchestrator Status
            </p>
            <Activity className="text-[#B80028]" size={18} />
          </div>
          <p className="text-xl font-semibold text-gray-900 capitalize">
            {status}
          </p>
          {lastRun && (
            <p className="text-xs text-[#718096] mt-1">Last run: {lastRun}</p>
          )}
        </div>

        <div className="bg-white rounded-lg p-5 flex-card-shadow">
          <p className="text-xs font-medium text-[#718096] uppercase tracking-wide flex items-center gap-2 mb-2">
            <span className="flex-red-dot" />
            Participating Sites
          </p>
          <p className="text-xl font-semibold text-gray-900">3 Hospitals</p>
          <p className="text-xs text-[#718096] mt-1">
            Configured via FLEX-Med clients table.
          </p>
        </div>

        <div className="bg-white rounded-lg p-5 flex-card-shadow">
          <p className="text-xs font-medium text-[#718096] uppercase tracking-wide flex items-center gap-2 mb-2">
            <span className="flex-red-dot" />
            Compliance
          </p>
          <div className="flex items-center gap-2">
            <Shield className="text-emerald-500" size={18} />
            <p className="text-sm font-medium text-gray-900">
              Data stays on-premise
            </p>
          </div>
          <p className="text-xs text-[#718096] mt-1">
            Only model updates leave local sites.
          </p>
        </div>
      </div>

      {/* Control + timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control panel */}
        <div className="bg-white rounded-lg p-6 flex-card-shadow lg:col-span-1">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Play size={18} className="text-[#B80028]" />
            Control Panel
          </h2>
          <p className="text-sm text-[#718096] mb-4">
            Trigger a new federated learning round using the current client
            configuration.
          </p>
          <button
            onClick={handleStart}
            disabled={isStarting}
            className="px-5 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg text-sm font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
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
        <div className="bg-white rounded-lg p-6 flex-card-shadow lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <Clock size={18} className="text-[#B80028]" />
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
                className="flex items-start gap-3 pb-3 border-b border-[#E2E8F0] last:border-0"
              >
                <div
                  className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                    item.status === "Completed"
                      ? "bg-emerald-500"
                      : item.status === "Degraded"
                      ? "bg-amber-500"
                      : "bg-[#B80028]"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 flex items-center gap-2">
                    {item.round}
                    <span className="text-xs font-normal text-[#718096]">
                      {item.status}
                    </span>
                  </p>
                  <p className="text-xs text-[#718096]">{item.detail}</p>
                  <p className="text-[11px] text-[#A0AEC0] mt-1">{item.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
