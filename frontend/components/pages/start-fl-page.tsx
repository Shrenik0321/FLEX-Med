"use client";

import { useState, useEffect, useRef } from "react";
import { Brain, Play, Settings, Users, Database, X } from "lucide-react";
import { API_BASE_PATH } from "@/utils";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { StartFLResponse } from "@/types/fl-simulation";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";

// Dummy client data
const dummyClients = [
  { id: 1, name: "Hospital A - ResNet18" },
  { id: 2, name: "Hospital B - MobileNetV2" },
  { id: 3, name: "Hospital C - DenseNet121" },
  { id: 4, name: "Hospital D - VGG16" },
  { id: 5, name: "Hospital E - EfficientNet" },
];

// Dataset options
const datasets = [
  { value: "cnmc", label: "CNMC 2019 Dataset" },
  { value: "cifar", label: "CIFAR Dataset" },
];

export default function StartFLPage() {
  const router = useRouter();
  const [isStarting, setIsStarting] = useState(false);
  const [selectedClients, setSelectedClients] = useState<number[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [showClientDropdown, setShowClientDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Configuration state
  const [config, setConfig] = useState({
    numRounds: 10,
    localEpochs: 5,
    learningRate: 0.001,
    batchSize: 32,
    distillEpochs: 3,
    distillLearningRate: 0.001,
    temperature: 3.0,
  });

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
    // if (selectedClients.length === 0) {
    //   toast.error("Please select at least one client");
    //   return;
    // }
    // if (!selectedDataset) {
    //   toast.error("Please select a dataset");
    //   return;
    // }

    try {
      setIsStarting(true);

      // Log configuration (in future, this will write to pyproject.toml)
      console.log("FL Configuration:", {
        selectedClients,
        selectedDataset,
        config,
      });

      // Call the API to start FL simulation
      const response = await fetch(`${API_BASE_PATH}/start_fl_simulation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // In future, send the configuration in the body
        // body: JSON.stringify({ selectedClients, selectedDataset, config }),
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

  const selectedClientNames = dummyClients
    .filter((c) => selectedClients.includes(c.id))
    .map((c) => c.name);

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-foreground flex items-center gap-3">
          <Brain className="text-primary" />
          Start FL Simulation
        </h1>
        <p className="text-muted-foreground mt-2">
          Configure and launch a new federated learning simulation across
          selected clients
        </p>
      </div>

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
                    {dummyClients.map((client) => (
                      <div
                        key={client.id}
                        className="flex items-center space-x-2 rounded-sm px-2 py-2 hover:bg-accent cursor-pointer"
                        onClick={() => handleClientToggle(client.id)}
                      >
                        <Checkbox
                          checked={selectedClients.includes(client.id)}
                          onCheckedChange={() => handleClientToggle(client.id)}
                        />
                        <label className="text-sm cursor-pointer flex-1">
                          {client.name}
                        </label>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Selected clients display */}
              {selectedClients.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {selectedClientNames.map((name, idx) => (
                    <div
                      key={idx}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-primary/10 text-primary rounded-full text-xs font-medium border border-primary/20"
                    >
                      {name}
                      <button
                        onClick={() =>
                          handleClientToggle(
                            dummyClients.find((c) => c.name === name)!.id,
                          )
                        }
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

          {/* Dataset Selection */}
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Database size={18} className="text-primary" />
              Select Dataset
            </h2>
            <div className="space-y-3">
              <Label>Public Dataset</Label>
              <Select
                value={selectedDataset}
                onValueChange={setSelectedDataset}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a dataset..." />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((dataset) => (
                    <SelectItem key={dataset.value} value={dataset.value}>
                      {dataset.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* FL Configuration */}
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Settings size={18} className="text-primary" />
              FL Configuration
            </h2>
            <p className="text-xs text-muted-foreground mb-4">
              These settings will be written to pyproject.toml (UI only for now)
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Number of Rounds */}
              <div className="space-y-2">
                <Label>Number of Federated Rounds</Label>
                <Input
                  type="number"
                  value={config.numRounds}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      numRounds: parseInt(e.target.value),
                    })
                  }
                  min="1"
                />
              </div>

              {/* Local Epochs */}
              <div className="space-y-2">
                <Label>Local Epochs per Client</Label>
                <Input
                  type="number"
                  value={config.localEpochs}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      localEpochs: parseInt(e.target.value),
                    })
                  }
                  min="1"
                />
              </div>

              {/* Learning Rate */}
              <div className="space-y-2">
                <Label>Learning Rate</Label>
                <Input
                  type="number"
                  step="0.0001"
                  value={config.learningRate}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      learningRate: parseFloat(e.target.value),
                    })
                  }
                />
              </div>

              {/* Batch Size */}
              <div className="space-y-2">
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  value={config.batchSize}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      batchSize: parseInt(e.target.value),
                    })
                  }
                  min="1"
                />
              </div>

              {/* Distillation Epochs */}
              <div className="space-y-2">
                <Label>Distillation Epochs</Label>
                <Input
                  type="number"
                  value={config.distillEpochs}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      distillEpochs: parseInt(e.target.value),
                    })
                  }
                  min="1"
                />
              </div>

              {/* Distillation Learning Rate */}
              <div className="space-y-2">
                <Label>Distillation Learning Rate</Label>
                <Input
                  type="number"
                  step="0.0001"
                  value={config.distillLearningRate}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      distillLearningRate: parseFloat(e.target.value),
                    })
                  }
                />
              </div>

              {/* Temperature */}
              <div className="space-y-2">
                <Label>Temperature</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={config.temperature}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      temperature: parseFloat(e.target.value),
                    })
                  }
                />
              </div>
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
                <p className="text-muted-foreground">Dataset</p>
                <p className="font-medium text-foreground">
                  {datasets.find((d) => d.value === selectedDataset)?.label ||
                    "None"}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Federated Rounds</p>
                <p className="font-medium text-foreground">
                  {config.numRounds}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Local Epochs</p>
                <p className="font-medium text-foreground">
                  {config.localEpochs}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Learning Rate</p>
                <p className="font-medium text-foreground">
                  {config.learningRate}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Batch Size</p>
                <p className="font-medium text-foreground">
                  {config.batchSize}
                </p>
              </div>
            </div>
          </div>

          {/* Start Button */}
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <button
              onClick={handleStart}
              disabled={isStarting}
              className="w-full px-5 py-3 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <Play size={18} />
              {isStarting ? "Starting Simulation..." : "Start FL Simulation"}
            </button>
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
