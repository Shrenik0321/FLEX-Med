"use client";

import { useState, useEffect, useRef } from "react";
import { Play, Users, X, Loader2 } from "lucide-react";
import { API_BASE_PATH } from "@/utils";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { Loading } from "@/components/ui/loading";
import { StartFLResponse } from "@/types/fl-simulation";
import { Client } from "@/types/client";
import { Checkbox } from "@/components/ui/checkbox";

export default function StartFLPage() {
  const router = useRouter();
  const [isStarting, setIsStarting] = useState(false);
  const [clients, setClients] = useState<Client[]>([]);
  const [isLoadingClients, setIsLoadingClients] = useState(true);
  const [selectedClients, setSelectedClients] = useState<number[]>([]);
  const [showClientDropdown, setShowClientDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchClients = async () => {
      try {
        setIsLoadingClients(true);
        const response = await fetch(`${API_BASE_PATH}/clients`);
        if (!response.ok) throw new Error("Failed to fetch clients");
        const data: Client[] = await response.json();
        setClients(data);
        setSelectedClients(data.map((c) => c.id));
        toast.success(`Loaded ${data.length} clients`);
      } catch (error) {
        console.error("Error fetching clients:", error);
        toast.error("Failed to load clients. Please check backend connection.");
      } finally {
        setIsLoadingClients(false);
      }
    };
    fetchClients();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowClientDropdown(false);
      }
    };
    if (showClientDropdown)
      document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showClientDropdown]);

  const handleClientToggle = (clientId: number) => {
    setSelectedClients((prev) =>
      prev.includes(clientId)
        ? prev.filter((id) => id !== clientId)
        : [...prev, clientId],
    );
  };

  const handleStart = async () => {
    if (selectedClients.length === 0) {
      toast.error("Please select at least one client");
      return;
    }
    try {
      setIsStarting(true);
      const response = await fetch(`${API_BASE_PATH}/start_fl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_ids: selectedClients }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to start simulation");
      }
      const data: StartFLResponse = await response.json();
      toast.success(
        `Federated learning simulation started! (ID: ${data.simulation_id})`,
      );
      setTimeout(() => {
        router.push(`/federated/${data.simulation_id}`);
      }, 1000);
    } catch (error) {
      console.error("Failed to start federated learning:", error);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to start simulation. Check backend service.",
      );
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="px-6 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Client Selection */}
        <div className="lg:col-span-2">
          <div className="rounded-xl border border-gray-100 bg-white p-5">
            <div className="flex items-center gap-2 mb-5">
              <Users size={13} className="text-gray-400" />
              <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                Participating Clients
              </p>
            </div>

            <div className="space-y-3" ref={dropdownRef}>
              <div className="relative">
                <button
                  onClick={() => setShowClientDropdown(!showClientDropdown)}
                  className="w-full flex items-center justify-between rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs text-gray-500 hover:border-gray-300 transition-colors focus:outline-none focus:ring-1 focus:ring-gray-300"
                >
                  <span>
                    {selectedClients.length === 0
                      ? "Select clients…"
                      : `${selectedClients.length} client${selectedClients.length > 1 ? "s" : ""} selected`}
                  </span>
                  <span className="text-gray-300 text-[10px]">▼</span>
                </button>

                {showClientDropdown && (
                  <div className="absolute z-50 mt-1 w-full rounded-lg border border-gray-100 bg-white shadow-lg py-1">
                    {isLoadingClients ? (
                      <Loading size="sm" text="Loading clients…" className="py-4" />
                    ) : clients.length === 0 ? (
                      <div className="text-xs text-gray-400 text-center py-4">
                        No clients found. Please add clients first.
                      </div>
                    ) : (
                      clients.map((client) => (
                        <div
                          key={client.id}
                          className="flex items-center gap-2.5 px-3 py-2 hover:bg-gray-50 cursor-pointer transition-colors"
                          onClick={() => handleClientToggle(client.id)}
                        >
                          <Checkbox
                            checked={selectedClients.includes(client.id)}
                            onCheckedChange={() => handleClientToggle(client.id)}
                          />
                          <span className="text-xs text-gray-700">
                            {client.client_name}
                          </span>
                          <span className="text-[10px] text-gray-400">
                            {client.model_type}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>

              {selectedClients.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {clients
                    .filter((c) => selectedClients.includes(c.id))
                    .map((client) => (
                      <div
                        key={client.id}
                        className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-50 border border-gray-100 rounded-md text-[10px] font-medium text-gray-600"
                      >
                        {client.client_name}
                        <button
                          onClick={() => handleClientToggle(client.id)}
                          className="ml-0.5 hover:text-gray-900 transition-colors"
                        >
                          <X size={10} />
                        </button>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Summary + Action */}
        <div className="lg:col-span-1 space-y-4">
          <div className="rounded-xl border border-gray-100 bg-white p-5">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400 mb-4">
              Summary
            </p>
            <div className="space-y-3">
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-gray-400">Clients</span>
                <span className="text-xs font-semibold text-gray-900 tabular-nums">
                  {selectedClients.length || "—"}
                </span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-xs text-gray-400">Dirichlet α</span>
                <span className="text-[10px] text-gray-400">
                  Configured in Settings
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-gray-100 bg-white p-5">
            <button
              onClick={handleStart}
              disabled={isStarting}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-[#b80028] text-white text-xs font-medium hover:bg-[#9a0022] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isStarting ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Starting…
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  Start FL Simulation
                </>
              )}
            </button>
            <p className="text-[10px] text-gray-400 mt-3 text-center leading-relaxed">
              Initiates federated learning with the selected configuration
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
