"use client";

import { useState, useEffect } from "react";

interface SystemConfig {
  // FL Training Configuration
  num_rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  lr_decay: number;

  // Knowledge Distillation Configuration
  distill_lr: number;
  distill_epochs: number;
  temperature: number;

  // Data Heterogeneity Configuration
  dirichlet_alpha: number;
  dirichlet_seed: number;
  dirichlet_min_partition_size: number;

  // Dataset Paths
  public_anchor_dataset_path: string;
  public_test_dataset_path: string;
  local_train_dataset_path: string;

  // System Configuration
  ngrok_url: string;
}

const defaultConfig: SystemConfig = {
  num_rounds: 10,
  local_epochs: 5,
  batch_size: 32,
  learning_rate: 0.0001,
  lr_decay: 0.99,
  distill_lr: 0.001,
  distill_epochs: 2,
  temperature: 3.0,
  dirichlet_alpha: 1.0,
  dirichlet_seed: 42,
  dirichlet_min_partition_size: 400,
  public_anchor_dataset_path: "/content/datasets/cnmc/cnmc_public_anchor",
  public_test_dataset_path: "/content/datasets/cnmc/cnmc_public_test",
  local_train_dataset_path: "/content/datasets/cnmc/cnmc_local_train",
  ngrok_url: "https://eb474f08357f.ngrok-free.app",
};

export default function SettingsPage() {
  const [config, setConfig] = useState<SystemConfig>(defaultConfig);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  // Fetch configuration from backend on mount
  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/system_config");

      if (!response.ok) {
        throw new Error(`Failed to fetch config: ${response.statusText}`);
      }

      const data = await response.json();
      setConfig(data.config);
    } catch (error) {
      console.error("Error fetching config:", error);
      setSaveMessage({
        type: "error",
        text: `Failed to load configuration: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
      // Keep default config on error
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (
    field: keyof SystemConfig,
    value: string | number,
  ) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMessage(null);

    try {
      const response = await fetch("http://localhost:8000/api/system_config", {
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

      setSaveMessage({
        type: "success",
        text: "Settings saved successfully!",
      });

      // Clear message after 5 seconds
      setTimeout(() => setSaveMessage(null), 5000);
    } catch (error) {
      console.error("Error saving config:", error);
      setSaveMessage({
        type: "error",
        text: `Failed to save settings: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
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
    await fetchConfig();
    setSaveMessage({
      type: "success",
      text: "Settings reloaded from database",
    });
    setTimeout(() => setSaveMessage(null), 3000);
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-semibold text-foreground mb-2">
        System Settings
      </h1>
      <p className="text-muted-foreground mb-8">
        Configure federated learning parameters, data heterogeneity, and system
        paths
      </p>

      {/* Loading State */}
      {loading && (
        <div className="mb-6 p-4 rounded-lg border border-border bg-card">
          <p className="text-muted-foreground">Loading configuration...</p>
        </div>
      )}

      {/* Save Message Banner */}
      {saveMessage && (
        <div
          className={`mb-6 p-4 rounded-lg border ${
            saveMessage.type === "success"
              ? "bg-green-50 border-green-200 text-green-800"
              : "bg-red-50 border-red-200 text-red-800"
          }`}
        >
          {saveMessage.text}
        </div>
      )}

      <div className="max-w-4xl space-y-6">
        {/* Federated Learning Configuration */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            Federated Learning Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Number of Rounds
                <span className="text-muted-foreground font-normal ml-2">
                  (Total FL rounds)
                </span>
              </label>
              <input
                type="number"
                value={config.num_rounds}
                onChange={(e) =>
                  handleInputChange("num_rounds", parseInt(e.target.value) || 0)
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="1"
                max="100"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Local Epochs
                <span className="text-muted-foreground font-normal ml-2">
                  (Per round)
                </span>
              </label>
              <input
                type="number"
                value={config.local_epochs}
                onChange={(e) =>
                  handleInputChange(
                    "local_epochs",
                    parseInt(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="1"
                max="20"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={config.batch_size}
                onChange={(e) =>
                  handleInputChange("batch_size", parseInt(e.target.value) || 0)
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="1"
                max="256"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={config.learning_rate}
                onChange={(e) =>
                  handleInputChange(
                    "learning_rate",
                    parseFloat(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="0.00001"
                max="1"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Learning Rate Decay
              </label>
              <input
                type="number"
                step="0.01"
                value={config.lr_decay}
                onChange={(e) =>
                  handleInputChange("lr_decay", parseFloat(e.target.value) || 0)
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="0.8"
                max="1"
              />
            </div>
          </div>
        </div>

        {/* Knowledge Distillation Configuration */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            Knowledge Distillation Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Distillation Learning Rate
              </label>
              <input
                type="number"
                step="0.0001"
                value={config.distill_lr}
                onChange={(e) =>
                  handleInputChange(
                    "distill_lr",
                    parseFloat(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="0.00001"
                max="1"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Distillation Epochs
              </label>
              <input
                type="number"
                value={config.distill_epochs}
                onChange={(e) =>
                  handleInputChange(
                    "distill_epochs",
                    parseInt(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="1"
                max="20"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Temperature
                <span className="text-muted-foreground font-normal ml-2">
                  (Softmax temp)
                </span>
              </label>
              <input
                type="number"
                step="0.1"
                value={config.temperature}
                onChange={(e) =>
                  handleInputChange(
                    "temperature",
                    parseFloat(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="1"
                max="10"
              />
            </div>
          </div>
        </div>

        {/* Data Heterogeneity Configuration */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            Data Heterogeneity Configuration
          </h2>
          <p className="text-sm text-muted-foreground mb-4">
            Controls Dirichlet partitioning for simulating label and quantity
            skew across clients
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Dirichlet Alpha
              </label>
              <input
                type="number"
                step="0.1"
                value={config.dirichlet_alpha}
                onChange={(e) =>
                  handleInputChange(
                    "dirichlet_alpha",
                    parseFloat(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="0.1"
                max="10"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Lower = more heterogeneity. Recommended: 1.0
              </p>
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Dirichlet Seed
              </label>
              <input
                type="number"
                value={config.dirichlet_seed}
                onChange={(e) =>
                  handleInputChange(
                    "dirichlet_seed",
                    parseInt(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="0"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Random seed for reproducibility
              </p>
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Min Partition Size
              </label>
              <input
                type="number"
                value={config.dirichlet_min_partition_size}
                onChange={(e) =>
                  handleInputChange(
                    "dirichlet_min_partition_size",
                    parseInt(e.target.value) || 0,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                min="50"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Minimum samples per client
              </p>
            </div>
          </div>
        </div>

        {/* Dataset Paths */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            Dataset Paths
          </h2>
          <p className="text-sm text-muted-foreground mb-4">
            Configure dataset locations for the Colab orchestrator
          </p>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Public Anchor Dataset Path
                <span className="text-muted-foreground font-normal ml-2">
                  (For knowledge distillation)
                </span>
              </label>
              <input
                type="text"
                value={config.public_anchor_dataset_path}
                onChange={(e) =>
                  handleInputChange(
                    "public_anchor_dataset_path",
                    e.target.value,
                  )
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card font-mono"
                placeholder="/content/datasets/cnmc/cnmc_public_anchor"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Public Test Dataset Path
                <span className="text-muted-foreground font-normal ml-2">
                  (For evaluation)
                </span>
              </label>
              <input
                type="text"
                value={config.public_test_dataset_path}
                onChange={(e) =>
                  handleInputChange("public_test_dataset_path", e.target.value)
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card font-mono"
                placeholder="/content/datasets/cnmc/cnmc_public_test"
              />
            </div>

            <div>
              <label className="text-sm font-medium text-foreground block mb-2">
                Local Train Dataset Path
                <span className="text-muted-foreground font-normal ml-2">
                  (For Dirichlet partitioning)
                </span>
              </label>
              <input
                type="text"
                value={config.local_train_dataset_path}
                onChange={(e) =>
                  handleInputChange("local_train_dataset_path", e.target.value)
                }
                className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card font-mono"
                placeholder="/content/datasets/cnmc/cnmc_local_train"
              />
            </div>
          </div>
        </div>

        {/* System Configuration */}
        <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <span className="flex-red-dot" />
            System Configuration
          </h2>
          <div>
            <label className="text-sm font-medium text-foreground block mb-2">
              NGROK URL
              <span className="text-muted-foreground font-normal ml-2">
                (Federated orchestrator endpoint)
              </span>
            </label>
            <input
              type="text"
              value={config.ngrok_url}
              onChange={(e) => handleInputChange("ngrok_url", e.target.value)}
              className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card font-mono"
              placeholder="https://your-ngrok-url.ngrok-free.app"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Update this when your ngrok tunnel URL changes
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-6 py-2 bg-primary text-white rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? "Saving..." : "Save Changes"}
          </button>
          <button
            onClick={handleReset}
            disabled={loading}
            className="px-6 py-2 bg-card border border-border text-foreground rounded-lg font-medium hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Reload from Database
          </button>
        </div>
      </div>
    </div>
  );
}
