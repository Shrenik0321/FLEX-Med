"use client";

import { Search, MoreVertical, Eye, Trash2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { toast } from "sonner";
import { API_BASE_PATH } from "@/utils";
import {
  FLSimulation,
  parseMetrics,
  formatDuration,
  formatDateTime,
} from "@/types/fl-simulation";

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

function ActionDropdown({
  simulation,
  onView,
  onDelete,
}: ActionDropdownProps) {
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
      <button
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="p-1 hover:bg-gray-100 rounded transition-colors"
        title="Actions"
      >
        <MoreVertical size={18} className="text-gray-600" />
      </button>

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
  const [isLoading, setIsLoading] = useState(true);

  // Fetch simulations from API
  useEffect(() => {
    const fetchSimulations = async () => {
      try {
        setIsLoading(true);
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
        setIsLoading(false);
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

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">
            FL Simulation History
          </h1>
          <p className="text-muted-foreground mt-2">
            View and manage all federated learning simulation runs
          </p>
        </div>
        <button
          onClick={onStartClick}
          className="px-4 py-2 bg-card border border-primary text-primary rounded-lg font-medium hover:bg-primary/10 transition-colors"
        >
          Start New Simulation
        </button>
      </div>

      {/* Search */}
      <div className="mb-6">
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
        {isLoading ? (
          <div className="p-8 text-center text-muted-foreground">
            Loading simulations...
          </div>
        ) : filteredSimulations.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground">
            No simulations found.{" "}
            <button
              onClick={onStartClick}
              className="text-primary hover:underline"
            >
              Start your first simulation
            </button>
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
                const metrics = parseMetrics(simulation.metrics);
                const avgAccuracy = metrics?.aggregate?.post_fl?.avg_accuracy;
                const numRounds = simulation.configs.num_server_rounds;
                const numClients = simulation.client_ids.length;

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
                    <td className="px-6 py-4 text-sm text-foreground">
                      {metrics?.total_rounds_completed || 0}/{numRounds}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {numClients}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {avgAccuracy
                        ? `${(avgAccuracy * 100).toFixed(1)}%`
                        : "N/A"}
                    </td>
                    <td className="px-6 py-4 text-sm text-foreground">
                      {simulation.duration
                        ? formatDuration(simulation.duration)
                        : simulation.started_at
                        ? "In progress..."
                        : "N/A"}
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
