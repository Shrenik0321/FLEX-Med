"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Loading } from "@/components/ui/loading";

interface SystemConfig {
  // Dirichlet Alpha (data heterogeneity control)
  dirichlet_alpha: number;

  // Training Strategy Parameters (global)
  minority_boost: number;
  focal_alpha: number;
  focal_gamma: number;
  consensus_momentum: number;
  distill_weight_base: number;
  distill_decay_rate: number;
  train_loss_weight: number;
  distill_loss_weight: number;
  lr_decay: number;
  learning_rate: number;

  // FL Training Configuration (global)
  num_rounds: number;
  local_epochs: number;
  batch_size: number;

  // Knowledge Distillation Configuration (global)
  distill_lr: number;
  distill_epochs: number;
  temperature: number;

  // Dirichlet Partitioning (global)
  dirichlet_seed: number;
  dirichlet_min_partition_size: number;

  // Dataset Paths (global)
  public_anchor_dataset_path: string;
  public_test_dataset_path: string;
  local_train_dataset_path: string;

  // System Configuration (global)
  ngrok_url: string;
}

export default function SettingsPage() {
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Fetch configuration from backend on mount
  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    setLoading(true);
    setFetchError(null);
    try {
      const response = await fetch("http://localhost:7860/api/system_config");

      if (!response.ok) {
        throw new Error(`Failed to fetch config: ${response.statusText}`);
      }

      const data = await response.json();
      setConfig(data.config);
    } catch (error) {
      console.error("Error fetching config:", error);
      const message = error instanceof Error ? error.message : "Unknown error";
      setFetchError(`Failed to load configuration: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (
    field: keyof SystemConfig,
    value: string | number,
  ) => {
    setConfig((prev) => {
      if (!prev) return prev;
      return { ...prev, [field]: value };
    });
  };

  const handleSave = async () => {
    setSaving(true);
    const toastId = toast.loading("Saving settings...");

    try {
      const response = await fetch("http://localhost:7860/api/system_config", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          config: config,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to save configuration");
      }

      const data = await response.json();
      setConfig(data.config);

      toast.success("Settings saved successfully!", { id: toastId });
    } catch (error) {
      console.error("Error saving config:", error);
      toast.error(
        `Failed to save settings: ${error instanceof Error ? error.message : "Unknown error"}`,
        { id: toastId },
      );
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    if (
      !confirm(
        "Are you sure you want to reset all settings to defaults? This will reload from the database.",
      )
    ) {
      return;
    }

    setLoading(true);
    const toastId = toast.loading("Reloading settings...");
    await fetchConfig();
    toast.success("Settings reloaded from database", { id: toastId });
  };

  // If config hasn't loaded yet, show loading/error state
  if (!config) {
    return (
      <div className="p-8">
        <h1 className="text-3xl font-semibold text-foreground mb-2">
          System Settings
        </h1>
        {loading && (
          <Loading className="mt-12" text="Loading configuration..." />
        )}
        {fetchError && (
          <div className="mt-6 p-4 rounded-lg border border-red-200 bg-red-50 text-red-800">
            <p>{fetchError}</p>
            <Button onClick={fetchConfig} className="mt-3">
              Retry
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="p-8 w-full">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-foreground mb-1">
            System Settings
          </h1>
          <p className="text-muted-foreground">
            Configure federated learning parameters, heterogeneity, and dataset
            paths
          </p>
        </div>

        <div className="flex gap-2">
          <Button onClick={handleSave} disabled={saving}>
            {saving ? "Saving..." : "Save Changes"}
          </Button>
          <Button variant="outline" onClick={handleReset} disabled={loading}>
            Reload
          </Button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Top Section: Heterogeneity Configuration */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex-red-dot" />
              Heterogeneity Configuration
            </CardTitle>
            <CardDescription>
              Configure Dirichlet partitioning parameters for data heterogeneity.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <Label className="mb-2 block">Dirichlet Alpha</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={config.dirichlet_alpha}
                  onChange={(e) =>
                    handleInputChange(
                      "dirichlet_alpha",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0.01"
                  max="100"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Controls data heterogeneity. Lower = more skewed (e.g. 0.1 extreme, 1.5 mild).
                </p>
              </div>
              <div>
                <Label className="mb-2 block">Min Partition Size</Label>
                <Input
                  type="number"
                  value={config.dirichlet_min_partition_size}
                  onChange={(e) =>
                    handleInputChange(
                      "dirichlet_min_partition_size",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  min="50"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Minimum samples per client partition
                </p>
              </div>
              <div>
                <Label className="mb-2 block">Dirichlet Seed</Label>
                <Input
                  type="number"
                  value={config.dirichlet_seed}
                  onChange={(e) =>
                    handleInputChange(
                      "dirichlet_seed",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  min="0"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Seed for deterministic partitioning
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Global FL Config */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex-red-dot" />
              Global FL Parameters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <Label className="mb-2 block">Number of Rounds</Label>
                <Input
                  type="number"
                  value={config.num_rounds}
                  onChange={(e) =>
                    handleInputChange(
                      "num_rounds",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  min="1"
                  max="100"
                />
              </div>
              <div>
                <Label className="mb-2 block">Local Epochs</Label>
                <Input
                  type="number"
                  value={config.local_epochs}
                  onChange={(e) =>
                    handleInputChange(
                      "local_epochs",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  min="1"
                  max="20"
                />
              </div>
              <div>
                <Label className="mb-2 block">Batch Size</Label>
                <Input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) =>
                    handleInputChange(
                      "batch_size",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  min="1"
                  max="256"
                />
              </div>
              <div>
                <Label className="mb-2 block">Learning Rate</Label>
                <Input
                  type="number"
                  step="0.0001"
                  value={config.learning_rate}
                  onChange={(e) =>
                    handleInputChange(
                      "learning_rate",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0.00001"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">Learning Rate Decay</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={config.lr_decay}
                  onChange={(e) =>
                    handleInputChange(
                      "lr_decay",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0.8"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">NGROK URL</Label>
                <Input
                  type="text"
                  value={config.ngrok_url}
                  onChange={(e) =>
                    handleInputChange("ngrok_url", e.target.value)
                  }
                  placeholder="https://your-ngrok-url.ngrok-free.app"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Advanced Training Configuration */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex-red-dot" />
              Advanced Training Configuration
            </CardTitle>
            <CardDescription>
              Fine-tune loss weights, focal loss parameters, and distillation
              settings.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div>
                <Label className="mb-2 block">Minority Boost</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.minority_boost}
                  onChange={(e) =>
                    handleInputChange(
                      "minority_boost",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="2"
                />
              </div>
              <div>
                <Label className="mb-2 block">Focal Alpha</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.focal_alpha}
                  onChange={(e) =>
                    handleInputChange(
                      "focal_alpha",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">Focal Gamma</Label>
                <Input
                  type="number"
                  step="0.1"
                  value={config.focal_gamma}
                  onChange={(e) =>
                    handleInputChange(
                      "focal_gamma",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="5"
                />
              </div>
              <div>
                <Label className="mb-2 block">Consensus Momentum</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.consensus_momentum}
                  onChange={(e) =>
                    handleInputChange(
                      "consensus_momentum",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">Distill Weight Base</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.distill_weight_base}
                  onChange={(e) =>
                    handleInputChange(
                      "distill_weight_base",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">Distill Decay Rate</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.distill_decay_rate}
                  onChange={(e) =>
                    handleInputChange(
                      "distill_decay_rate",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">Train Loss Weight</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.train_loss_weight}
                  onChange={(e) =>
                    handleInputChange(
                      "train_loss_weight",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="1"
                />
              </div>
              <div>
                <Label className="mb-2 block">Distill Loss Weight</Label>
                <Input
                  type="number"
                  step="0.05"
                  value={config.distill_loss_weight}
                  onChange={(e) =>
                    handleInputChange(
                      "distill_loss_weight",
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  min="0"
                  max="1"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Knowledge Distillation & Datasets */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="flex-red-dot" />
              Knowledge Distillation & Datasets
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <Label className="font-semibold text-base">Configuration</Label>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="mb-2 block">Distill LR</Label>
                    <Input
                      type="number"
                      step="0.0001"
                      value={config.distill_lr}
                      onChange={(e) =>
                        handleInputChange(
                          "distill_lr",
                          parseFloat(e.target.value) || 0,
                        )
                      }
                      min="0.00001"
                      max="1"
                    />
                  </div>
                  <div>
                    <Label className="mb-2 block">Distill Epochs</Label>
                    <Input
                      type="number"
                      value={config.distill_epochs}
                      onChange={(e) =>
                        handleInputChange(
                          "distill_epochs",
                          parseInt(e.target.value) || 0,
                        )
                      }
                      min="1"
                      max="20"
                    />
                  </div>
                  <div className="col-span-2">
                    <Label className="mb-2 block">Temperature (Softmax)</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={config.temperature}
                      onChange={(e) =>
                        handleInputChange(
                          "temperature",
                          parseFloat(e.target.value) || 0,
                        )
                      }
                      min="1"
                      max="10"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <Label className="font-semibold text-base">Paths</Label>
                <div>
                  <Label className="mb-2 block">Public Anchor</Label>
                  <Input
                    type="text"
                    value={config.public_anchor_dataset_path}
                    onChange={(e) =>
                      handleInputChange(
                        "public_anchor_dataset_path",
                        e.target.value,
                      )
                    }
                    placeholder="/path/to/anchor"
                  />
                </div>
                <div>
                  <Label className="mb-2 block">Public Test</Label>
                  <Input
                    type="text"
                    value={config.public_test_dataset_path}
                    onChange={(e) =>
                      handleInputChange(
                        "public_test_dataset_path",
                        e.target.value,
                      )
                    }
                    placeholder="/path/to/test"
                  />
                </div>
                <div>
                  <Label className="mb-2 block">Local Train</Label>
                  <Input
                    type="text"
                    value={config.local_train_dataset_path}
                    onChange={(e) =>
                      handleInputChange(
                        "local_train_dataset_path",
                        e.target.value,
                      )
                    }
                    placeholder="/path/to/local/train"
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
