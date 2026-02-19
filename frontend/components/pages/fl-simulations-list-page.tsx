"use client";

import { Search, MoreVertical, Eye, Trash2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import { Button } from "@/components/ui/button";
import {
  FLSimulation,
  parseMetrics,
  formatDuration,
  formatDateTime,
} from "@/types/fl-simulation";
import { Loading } from "../ui/loading";

interface FLHistoryPageProps {
  onStartClick: () => void;
  onSelectSimulation: (simulation: FLSimulation) => void;
}

// Action Dropdown Component
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

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className="relative" ref={dropdownRef}>
      <Button
        variant="ghost"
        size="icon"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="h-8 w-8 hover:bg-muted"
        title="Actions"
      >
        <MoreVertical className="h-4 w-4 text-muted-foreground" />
      </Button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-card rounded-lg shadow-lg border border-border py-1 z-50">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onView(simulation);
              setIsOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors"
          >
            <Eye size={16} className="text-blue-600" />
            View Details
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(simulation);
              setIsOpen(false);
            }}
            className="w-full px-4 py-2 text-left text-sm text-primary hover:bg-red-50 flex items-center gap-2 transition-colors"
          >
            <Trash2 size={16} />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

export default function FLHistoryPage({
  onStartClick,
  onSelectSimulation,
}: FLHistoryPageProps) {
  const [simulations, setSimulations] = useState<FLSimulation[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(true);

  // Fetch simulations from API
  useEffect(() => {
    const fetchSimulations = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_PATH}/fl_simulations`);

        if (!response.ok) {
          throw new Error("Failed to fetch simulations");
        }

        const data: FLSimulation[] = await response.json();
        setSimulations(data);
      } catch (error) {
        console.error("Error fetching simulations:", error);
        toast.error("Failed to load simulation history");
      } finally {
        setLoading(false);
      }
    };

    fetchSimulations();
  }, []);

  const handleView = (simulation: FLSimulation) => {
    toast.info(`Viewing simulation: ID ${simulation.id}`);
    onSelectSimulation(simulation);
  };

  const handleDelete = async (simulation: FLSimulation) => {
    // TODO: Implement delete API endpoint
    toast.success(`Deleted simulation: ID ${simulation.id}`);
    // Refresh simulations after delete
    setSimulations((prev) => prev.filter((s) => s.id !== simulation.id));
  };

  const filteredSimulations = simulations.filter((sim) => {
    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    return (
      sim.id.toString().includes(term) ||
      sim.status.toLowerCase().includes(term)
    );
  });

  if (loading) {
    return <Loading fullScreen text="Synchronizing client database..." />;
  }

  return (
    <div className="p-8">
      {/* Search */}
      <div className="mb-6 max-w-md">
        <div className="relative">
          <Search
            className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
            size={18}
          />
          <input
            type="text"
            placeholder="Search simulations..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
          />
        </div>
      </div>

      {/* Simulations Table */}
      <div className="bg-card rounded-lg shadow-sm border border-border">
        {filteredSimulations.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground">
            No simulations found.{" "}
            <Button
              variant="link"
              onClick={onStartClick}
              className="p-0 h-auto font-medium"
            >
              Start your first simulation
            </Button>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Simulation ID
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Heterogeneity
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Rounds
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Clients
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Avg Accuracy
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Duration
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Started At
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredSimulations.map((simulation) => {
                const metrics = parseMetrics(simulation.aggregate_metrics);
                const avgAccuracy = metrics?.aggregate?.post_fl?.avg_accuracy;
                const numClients = metrics?.total_clients || 0;
                const preset = simulation.heterogeneity_preset ?? simulation.configs?.heterogeneity_preset;
                const alpha = simulation.configs?.dirichlet_alpha;

                return (
                  <tr
                    key={simulation.id}
                    onClick={() => onSelectSimulation(simulation)}
                    className="cursor-pointer border-b border-border hover:bg-[rgba(184,0,40,0.03)] transition-colors"
                  >
                    <td className="px-6 py-4 text-sm font-medium text-foreground">
                      Simulation #{simulation.id}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span
                        className={`inline-block px-3 py-1 rounded-full text-xs font-medium border ${
                          simulation.status === "completed"
                            ? "border-green-500 text-green-700 bg-green-50"
                            : simulation.status === "running"
                              ? "border-blue-500 text-blue-700 bg-blue-50"
                              : simulation.status === "pending"
                                ? "border-yellow-500 text-yellow-700 bg-yellow-50"
                                : "border-red-500 text-red-700 bg-red-50"
                        }`}
                      >
                        {simulation.status.charAt(0).toUpperCase() +
                          simulation.status.slice(1)}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm">
                      {preset ? (
                        <div className="flex flex-col">
                          <span
                            className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium border w-fit ${
                              preset === "high"
                                ? "border-red-300 text-red-700 bg-red-50"
                                : preset === "moderate"
                                  ? "border-amber-300 text-amber-700 bg-amber-50"
                                  : preset === "low"
                                    ? "border-emerald-300 text-emerald-700 bg-emerald-50"
                                    : "border-purple-300 text-purple-700 bg-purple-50"
                            }`}
                          >
                            {preset.charAt(0).toUpperCase() + preset.slice(1)}
                          </span>
                          {alpha != null && (
                            <span className="text-[10px] text-muted-foreground mt-0.5">
                              &alpha;={alpha}
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {metrics?.total_rounds_completed ?? "?"}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {numClients || "—"}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {avgAccuracy ? `${(avgAccuracy * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {simulation.duration
                        ? formatDuration(simulation.duration)
                        : simulation.started_at
                          ? "In progress..."
                          : "—"}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {simulation.started_at
                        ? formatDateTime(simulation.started_at)
                        : "Not started"}
                    </td>
                    <td className="px-6 py-4 text-sm">
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
    </div>
  );
}
