"use client";

import { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Loading } from "@/components/ui/loading";

interface SystemConfig {
  dirichlet_alpha: number;
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
  num_rounds: number;
  local_epochs: number;
  batch_size: number;
  distill_lr: number;
  distill_epochs: number;
  temperature: number;
  dirichlet_seed: number;
  dirichlet_min_partition_size: number;
  public_anchor_dataset_path: string;
  public_test_dataset_path: string;
  local_train_dataset_path: string;
  ngrok_url: string;
}

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

function SectionHeader({ title, description }: { title: string; description?: string }) {
  return (
    <div className="mb-5">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">{title}</p>
      {description && <p className="text-sm text-gray-400 mt-1">{description}</p>}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1.5">{label}</label>
      {children}
      {hint && <p className="text-xs text-gray-400 mt-1.5">{hint}</p>}
    </div>
  );
}

const inputCls =
  "w-full px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-[#b80028]/20 bg-white";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    setLoading(true);
    setFetchError(null);
    try {
      const response = await fetch("http://localhost:7860/api/system_config");
      if (!response.ok) throw new Error(`Failed to fetch config: ${response.statusText}`);
      const data = await response.json();
      setConfig(data.config);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setFetchError(`Failed to load configuration: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const set = (field: keyof SystemConfig, value: string | number) =>
    setConfig((prev) => (prev ? { ...prev, [field]: value } : prev));

  const handleSave = async () => {
    setSaving(true);
    const toastId = toast.loading("Saving settings…");
    try {
      const response = await fetch("http://localhost:7860/api/system_config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to save");
      }
      const data = await response.json();
      setConfig(data.config);
      toast.success("Settings saved", { id: toastId });
    } catch (error) {
      toast.error(
        `Failed to save: ${error instanceof Error ? error.message : "Unknown error"}`,
        { id: toastId },
      );
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    if (!confirm("Reset all settings to defaults?")) return;
    setLoading(true);
    const toastId = toast.loading("Reloading settings…");
    await fetchConfig();
    toast.success("Settings reloaded", { id: toastId });
  };

  // ---- Loading / error states ----
  if (!config) {
    return (
      <div className="px-6 py-8">
        {loading && <Loading className="min-h-[400px]" text="Loading configuration…" />}
        {fetchError && (
          <div className="rounded-xl border border-red-100 bg-red-50 p-5">
            <p className="text-sm text-red-700 mb-4">{fetchError}</p>
            <button
              onClick={fetchConfig}
              className="inline-flex items-center px-4 py-1.5 rounded-lg bg-[#b80028] text-white text-sm font-medium hover:bg-[#9b0022] transition-colors"
            >
              Retry
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="px-6 py-8 space-y-5">
      {/* Page actions */}
      <div className="flex items-center justify-end gap-2">
        <button
          onClick={handleReset}
          disabled={loading}
          className="px-3 py-1.5 rounded-lg text-sm font-medium text-gray-500 hover:bg-gray-100 disabled:opacity-40 transition-colors"
        >
          Reload
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-[#b80028] text-white text-sm font-medium hover:bg-[#9b0022] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {saving ? "Saving…" : "Save Changes"}
        </button>
      </div>

      {/* ── Heterogeneity ── */}
      <div className="rounded-xl border border-gray-100 bg-white p-5">
        <SectionHeader
          title="Heterogeneity Configuration"
          description="Dirichlet partitioning parameters for non-IID data distribution across clients."
        />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Field
            label="Dirichlet Alpha"
            hint="Lower = more skewed. 0.1 extreme → 1.5 mild → 10 near-IID."
          >
            <input
              type="number"
              step="0.1"
              min="0.01"
              max="100"
              value={config.dirichlet_alpha}
              onChange={(e) => set("dirichlet_alpha", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Min Partition Size" hint="Minimum samples guaranteed per client.">
            <input
              type="number"
              min="50"
              value={config.dirichlet_min_partition_size}
              onChange={(e) => set("dirichlet_min_partition_size", parseInt(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Dirichlet Seed" hint="Fixed seed for reproducible partitions.">
            <input
              type="number"
              min="0"
              value={config.dirichlet_seed}
              onChange={(e) => set("dirichlet_seed", parseInt(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
        </div>
      </div>

      {/* ── FL Parameters ── */}
      <div className="rounded-xl border border-gray-100 bg-white p-5">
        <SectionHeader title="Global FL Parameters" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Field label="Number of Rounds">
            <input
              type="number"
              min="1"
              max="100"
              value={config.num_rounds}
              onChange={(e) => set("num_rounds", parseInt(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Local Epochs">
            <input
              type="number"
              min="1"
              max="20"
              value={config.local_epochs}
              onChange={(e) => set("local_epochs", parseInt(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Batch Size">
            <input
              type="number"
              min="1"
              max="256"
              value={config.batch_size}
              onChange={(e) => set("batch_size", parseInt(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Learning Rate">
            <input
              type="number"
              step="0.0001"
              min="0.00001"
              max="1"
              value={config.learning_rate}
              onChange={(e) => set("learning_rate", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="LR Decay">
            <input
              type="number"
              step="0.01"
              min="0.8"
              max="1"
              value={config.lr_decay}
              onChange={(e) => set("lr_decay", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="NGROK URL">
            <input
              type="text"
              value={config.ngrok_url}
              onChange={(e) => set("ngrok_url", e.target.value)}
              placeholder="https://…ngrok-free.app"
              className={inputCls}
            />
          </Field>
        </div>
      </div>

      {/* ── Advanced Training ── */}
      <div className="rounded-xl border border-gray-100 bg-white p-5">
        <SectionHeader
          title="Advanced Training Configuration"
          description="Loss weights, focal loss parameters, and consensus distillation settings."
        />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
          <Field label="Minority Boost">
            <input
              type="number"
              step="0.05"
              min="0"
              max="2"
              value={config.minority_boost}
              onChange={(e) => set("minority_boost", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Focal Alpha">
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.focal_alpha}
              onChange={(e) => set("focal_alpha", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Focal Gamma">
            <input
              type="number"
              step="0.1"
              min="0"
              max="5"
              value={config.focal_gamma}
              onChange={(e) => set("focal_gamma", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Consensus Momentum">
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.consensus_momentum}
              onChange={(e) => set("consensus_momentum", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Distill Weight Base">
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.distill_weight_base}
              onChange={(e) => set("distill_weight_base", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Distill Decay Rate">
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.distill_decay_rate}
              onChange={(e) => set("distill_decay_rate", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Train Loss Weight">
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.train_loss_weight}
              onChange={(e) => set("train_loss_weight", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
          <Field label="Distill Loss Weight">
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.distill_loss_weight}
              onChange={(e) => set("distill_loss_weight", parseFloat(e.target.value) || 0)}
              className={inputCls}
            />
          </Field>
        </div>
      </div>

      {/* ── Knowledge Distillation & Datasets ── */}
      <div className="rounded-xl border border-gray-100 bg-white p-5">
        <SectionHeader title="Knowledge Distillation & Datasets" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-5">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">Distillation</p>
            <div className="grid grid-cols-2 gap-5">
              <Field label="Distill LR">
                <input
                  type="number"
                  step="0.0001"
                  min="0.00001"
                  max="1"
                  value={config.distill_lr}
                  onChange={(e) => set("distill_lr", parseFloat(e.target.value) || 0)}
                  className={inputCls}
                />
              </Field>
              <Field label="Distill Epochs">
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={config.distill_epochs}
                  onChange={(e) => set("distill_epochs", parseInt(e.target.value) || 0)}
                  className={inputCls}
                />
              </Field>
            </div>
            <Field label="Temperature (Softmax)">
              <input
                type="number"
                step="0.1"
                min="1"
                max="10"
                value={config.temperature}
                onChange={(e) => set("temperature", parseFloat(e.target.value) || 0)}
                className={inputCls}
              />
            </Field>
          </div>

          <div className="space-y-5">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">Dataset Paths</p>
            <Field label="Public Anchor">
              <input
                type="text"
                value={config.public_anchor_dataset_path}
                onChange={(e) => set("public_anchor_dataset_path", e.target.value)}
                placeholder="/path/to/anchor"
                className={inputCls}
              />
            </Field>
            <Field label="Public Test">
              <input
                type="text"
                value={config.public_test_dataset_path}
                onChange={(e) => set("public_test_dataset_path", e.target.value)}
                placeholder="/path/to/test"
                className={inputCls}
              />
            </Field>
            <Field label="Local Train">
              <input
                type="text"
                value={config.local_train_dataset_path}
                onChange={(e) => set("local_train_dataset_path", e.target.value)}
                placeholder="/path/to/local/train"
                className={inputCls}
              />
            </Field>
          </div>
        </div>
      </div>
    </div>
  );
}
