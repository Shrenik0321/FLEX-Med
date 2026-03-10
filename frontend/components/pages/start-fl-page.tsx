"use client";

import { useState, useEffect, useRef } from "react";
import { Play, Users, X, Loader2 } from "lucide-react";
import { API_BASE_PATH } from "@/utils";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Loading } from "@/components/ui/loading";
import { StartFLResponse } from "@/types/fl-simulation";
import { Client } from "@/types/client";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

export default function StartFLPage() {
  const router = useRouter();
  const [isStarting, setIsStarting] = useState(false);
  const [clients, setClients] = useState<Client[]>([]);
  const [isLoadingClients, setIsLoadingClients] = useState(true);
  const [selectedClients, setSelectedClients] = useState<number[]>([]);

  const [showClientDropdown, setShowClientDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch clients from backend on mount
  useEffect(() => {
    const fetchClients = async () => {
      try {
        setIsLoadingClients(true);
        const response = await fetch(`${API_BASE_PATH}/clients`);

        if (!response.ok) {
          throw new Error("Failed to fetch clients");
        }

        const data: Client[] = await response.json();
        setClients(data);

        // Select all clients by default
        const allClientIds = data.map((client) => client.id);
        setSelectedClients(allClientIds);

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

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setShowClientDropdown(false);
      }
    };

    if (showClientDropdown) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showClientDropdown]);

  const handleClientToggle = (clientId: number) => {
    setSelectedClients((prev) =>
      prev.includes(clientId)
        ? prev.filter((id) => id !== clientId)
        : [...prev, clientId],
    );
  };

  const handleStart = async () => {
    // Validation
    if (selectedClients.length === 0) {
      toast.error("Please select at least one client");
      return;
    }
    // if (!selectedDataset) {
    //   toast.error("Please select a dataset");
    //   return;
    // }

    try {
      setIsStarting(true);

      // Call the API to start FL simulation with selected client IDs
      // Training config is managed in Settings page; heterogeneity preset is selected here
      const response = await fetch(`${API_BASE_PATH}/start_fl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_ids: selectedClients,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to start simulation");
      }

      const data: StartFLResponse = await response.json();

      toast.success(
        `Federated learning simulation started! (ID: ${data.simulation_id})`,
      );

      // Redirect to the simulation details page after 1 second
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

  const selectedClientNames = clients
    .filter((c) => selectedClients.includes(c.id))
    .map((c) => `${c.client_name} - ${c.model_type}`);

  return (
    <div className="p-8 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Client Selection */}
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Users size={18} className="text-primary" />
              Select Participating Clients
            </h2>
            <div className="space-y-3">
              <Label>Clients (Multi-select)</Label>
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setShowClientDropdown(!showClientDropdown)}
                  className="w-full flex items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                >
                  <span className="text-muted-foreground">
                    {selectedClients.length === 0
                      ? "Select clients..."
                      : `${selectedClients.length} client(s) selected`}
                  </span>
                  <span className="text-muted-foreground">▼</span>
                </button>

                {showClientDropdown && (
                  <div className="absolute z-50 mt-2 w-full rounded-md border bg-popover p-2 shadow-md">
                    {isLoadingClients ? (
                      <Loading
                        size="sm"
                        text="Loading clients..."
                        className="py-4"
                      />
                    ) : clients.length === 0 ? (
                      <div className="text-sm text-muted-foreground text-center py-4">
                        No clients found. Please add clients first.
                      </div>
                    ) : (
                      clients.map((client) => (
                        <div
                          key={client.id}
                          className="flex items-center space-x-2 rounded-sm px-2 py-2 hover:bg-accent hover:text-accent-foreground cursor-pointer"
                          onClick={() => handleClientToggle(client.id)}
                        >
                          <Checkbox
                            checked={selectedClients.includes(client.id)}
                            onCheckedChange={() =>
                              handleClientToggle(client.id)
                            }
                          />
                          <label className="text-sm cursor-pointer flex-1">
                            {client.client_name} - {client.model_type}
                          </label>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>

              {/* Selected clients display */}
              {selectedClients.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {clients
                    .filter((c) => selectedClients.includes(c.id))
                    .map((client) => (
                      <div
                        key={client.id}
                        className="inline-flex items-center gap-1 px-3 py-1 bg-primary/10 text-primary rounded-full text-xs font-medium border border-primary/20"
                      >
                        {client.client_name} - {client.model_type}
                        <button
                          onClick={() => handleClientToggle(client.id)}
                          className="ml-1 hover:bg-primary/20 rounded-full p-0.5"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>

        </div>

        {/* Summary & Actions */}
        <div className="lg:col-span-1 space-y-6">
          {/* Configuration Summary */}
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">
              Configuration Summary
            </h3>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-muted-foreground">Clients</p>
                <p className="font-medium text-foreground">
                  {selectedClients.length || "None"}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Dirichlet Alpha</p>
                <p className="font-medium text-foreground text-xs text-muted-foreground">
                  Configured in Settings
                </p>
              </div>
            </div>
          </div>

          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <Button
              onClick={handleStart}
              disabled={isStarting}
              className="w-full text-base py-6"
            >
              {isStarting ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Starting Simulation...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5 fill-current" />
                  Start FL Simulation
                </>
              )}
            </Button>
            <p className="text-xs text-muted-foreground mt-3 text-center">
              This will initiate federated learning with the selected
              configuration
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
