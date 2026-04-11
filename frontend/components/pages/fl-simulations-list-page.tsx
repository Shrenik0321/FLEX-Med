"use client";

import { MoreVertical, Eye, Trash2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import {
  FLSimulation,
  parseMetrics,
  formatDateTime,
} from "@/types/fl-simulation";
import { Loading } from "../ui/loading";
import { EmptyState } from "../ui/empty-state";

interface FLHistoryPageProps {
  onStartClick: () => void;
  onSelectSimulation: (simulation: FLSimulation) => void;
}

interface ActionDropdownProps {
  simulation: FLSimulation;
  onView: (simulation: FLSimulation) => void;
  onDelete: (simulation: FLSimulation) => void;
}

function ActionDropdown({ simulation, onView, onDelete }: ActionDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    if (isOpen) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="p-1.5 rounded-md hover:bg-gray-100 transition-colors"
        title="Actions"
      >
        <MoreVertical className="h-3.5 w-3.5 text-gray-400" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-1 w-40 bg-white rounded-lg border border-gray-100 shadow-lg py-1 z-50">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onView(simulation);
              setIsOpen(false);
            }}
            className="w-full px-3 py-2 text-left text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors"
          >
            <Eye size={13} className="text-blue-500" />
            View Details
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(simulation);
              setIsOpen(false);
            }}
            className="w-full px-3 py-2 text-left text-xs text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
          >
            <Trash2 size={13} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

function StatusTag({ status }: { status: string }) {
  const map: Record<string, string> = {
    completed: "text-emerald-700 bg-emerald-50 border-emerald-200",
    running: "text-blue-700 bg-blue-50 border-blue-200",
    pending: "text-amber-700 bg-amber-50 border-amber-200",
    failed: "text-red-700 bg-red-50 border-red-200",
  };
  const cls = map[status] ?? "text-gray-500 bg-gray-50 border-gray-200";
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold border ${cls}`}
    >
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

const tagPill =
  "inline-flex items-center justify-center min-w-[2.25rem] px-2 py-0.5 rounded-md text-xs font-semibold border tabular-nums";

function alphaHighlightClass(alpha: number): string | null {
  if (Math.abs(alpha - 0.5) < 1e-6) {
    return "text-violet-700 bg-violet-50 border-violet-200";
  }
  if (Math.abs(alpha - 1.0) < 1e-6) {
    return "text-teal-700 bg-teal-50 border-teal-200";
  }
  if (Math.abs(alpha - 1.5) < 1e-6) {
    return "text-orange-700 bg-orange-50 border-orange-200";
  }
  return null;
}

function AlphaTag({ alpha }: { alpha: number }) {
  const cls = alphaHighlightClass(alpha);
  const label = String(alpha);
  if (cls) {
    return <span className={`${tagPill} ${cls}`}>{label}</span>;
  }
  return <span className="text-gray-600 tabular-nums">{label}</span>;
}

const SEED_STYLES: Record<number, string> = {
  7: "text-sky-700 bg-sky-50 border-sky-200",
  42: "text-indigo-700 bg-indigo-50 border-indigo-200",
  123: "text-fuchsia-700 bg-fuchsia-50 border-fuchsia-200",
};

function SeedTag({ seed }: { seed: number }) {
  const cls = SEED_STYLES[seed];
  if (cls) {
    return <span className={`${tagPill} ${cls}`}>{seed}</span>;
  }
  return <span className="text-gray-500 tabular-nums">{seed}</span>;
}

export default function FLHistoryPage({
  onStartClick,
  onSelectSimulation,
}: FLHistoryPageProps) {
  const [simulations, setSimulations] = useState<FLSimulation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSimulations = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_PATH}/fl_simulations`);
        if (!response.ok) throw new Error("Failed to fetch simulations");
        const data: FLSimulation[] = await response.json();
        setSimulations(data);
      } catch {
        toast.error("Failed to load simulation history");
      } finally {
        setLoading(false);
      }
    };
    fetchSimulations();
  }, []);

  const handleView = (simulation: FLSimulation) => {
    onSelectSimulation(simulation);
  };

  const handleDelete = async (simulation: FLSimulation) => {
    toast.success(`Deleted simulation #${simulation.id}`);
    setSimulations((prev) => prev.filter((s) => s.id !== simulation.id));
  };

  return (
    <div className="px-6 py-8">
      {loading ? (
        <Loading className="min-h-[400px]" text="Loading simulations…" />
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          {simulations.length === 0 ? (
            <EmptyState
              title="No simulations found"
              description="Get started by running your first federated learning simulation."
              action={{ label: "Start Simulation", onClick: onStartClick }}
            />
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50/60">
                  {[
                    "ID",
                    "Status",
                    "Alpha",
                    "Seed",
                    "Rounds",
                    "Clients",
                    "Avg Accuracy",
                    "Started",
                    "",
                  ].map((h) => (
                    <th
                      key={h}
                      className={`py-3 px-4 font-semibold uppercase tracking-wider text-xs text-gray-400 ${
                        h === "ID" || h === "Status"
                          ? "text-left"
                          : h === ""
                            ? "text-right"
                            : "text-right"
                      }`}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {simulations.map((simulation) => {
                  const metrics = parseMetrics(simulation.aggregate_metrics);
                  const avgAccuracy = metrics?.aggregate?.post_fl?.avg_accuracy;
                  const numClients = metrics?.total_clients || 0;
                  const alpha = simulation.configs?.dirichlet_alpha;
                  const seed =
                    simulation.configs?.training_config?.dirichlet_seed;

                  return (
                    <tr
                      key={simulation.id}
                      onClick={() => onSelectSimulation(simulation)}
                      className="cursor-pointer border-b border-gray-100 last:border-0 hover:bg-gray-100 transition-colors"
                    >
                      <td className="py-3.5 px-4 font-semibold text-gray-900 tabular-nums">
                        #{simulation.id}
                      </td>
                      <td className="py-3.5 px-4">
                        <StatusTag status={simulation.status} />
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        {alpha != null ? (
                          <AlphaTag alpha={Number(alpha)} />
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        {seed != null ? (
                          <SeedTag seed={Number(seed)} />
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right tabular-nums text-gray-600">
                        {metrics?.total_rounds_completed ?? (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right tabular-nums text-gray-600">
                        {numClients || <span className="text-gray-300">—</span>}
                      </td>
                      <td className="py-3.5 px-4 text-right tabular-nums font-medium text-gray-900">
                        {avgAccuracy ? (
                          `${(avgAccuracy * 100).toFixed(1)}%`
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right tabular-nums text-gray-400">
                        {simulation.started_at ? (
                          formatDateTime(simulation.started_at)
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <ActionDropdown
                          simulation={simulation}
                          onView={handleView}
                          onDelete={handleDelete}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
