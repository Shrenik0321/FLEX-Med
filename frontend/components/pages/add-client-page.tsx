"use client";

import { API_BASE_PATH } from "@/utils";
import { Loader2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

interface AddClientPageProps {
  onBack: () => void;
}

const inputCls =
  "w-full px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-800 placeholder:text-gray-400 focus:outline-none focus:ring-1 focus:ring-[#b80028]/20 bg-white";

export default function AddClientPage({ onBack }: AddClientPageProps) {
  const [clientName, setClientName] = useState("");
  const [modelType, setModelType] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    if (!clientName || !modelType) {
      toast.error("Please fill in all required fields");
      return;
    }

    setIsLoading(true);
    const toastId = toast.loading("Registering client…");

    try {
      const response = await fetch(`${API_BASE_PATH}/clients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_name: clientName, model_type: modelType }),
      });

      if (!response.ok) throw new Error("Failed to register client");

      toast.success("Client registered successfully!", { id: toastId });
      onBack();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "An error occurred", {
        id: toastId,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const modelInsights: Record<string, string> = {
    efficientnet_b0:
      "Baseline compound-scaled network — ~5.3M params, 224×224 input. Best efficiency-accuracy tradeoff for small medical datasets.",
    efficientnet_b1:
      "Scaled-up B0 — ~7.8M params, 240×240 input. Moderately more expressive with slightly higher compute cost.",
    efficientnet_b2:
      "Further scaled variant — ~9.1M params, 260×260 input. Higher resolution captures finer morphological details in blood smear images.",
    mobilenet_v2:
      "Lightweight inverted-residuals network — ~3.4M params, 224×224 input. Ideal for resource-constrained client nodes.",
    densenet121:
      "Dense connectivity network — ~8M params, 224×224 input. Strong feature reuse, good for heterogeneous FL scenarios.",
    resnet18:
      "Residual network baseline — ~11M params, 224×224 input. Reliable general-purpose backbone with proven medical imaging performance.",
  };

  return (
    <div className="px-6 py-8">
      <div className="max-w-lg space-y-4">
        {/* Organization Info */}
        <div className="rounded-xl border border-gray-100 bg-white p-5 space-y-4">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
            Organization Information
          </p>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              Organization Name
            </label>
            <input
              type="text"
              placeholder="e.g., City Medical Center"
              value={clientName}
              onChange={(e) => setClientName(e.target.value)}
              className={inputCls}
            />
          </div>
        </div>

        {/* Model Selection */}
        <div className="rounded-xl border border-gray-100 bg-white p-5 space-y-4">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
            Model Architecture
          </p>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1.5">
              CNN Model
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className={inputCls}
            >
              <option value="">Select a model…</option>
              <optgroup label="EfficientNet">
                <option value="efficientnet_b0">EfficientNet-B0 · 5.3M params</option>
                <option value="efficientnet_b1">EfficientNet-B1 · 7.8M params</option>
                <option value="efficientnet_b2">EfficientNet-B2 · 9.1M params</option>
              </optgroup>
              <optgroup label="Lightweight">
                <option value="mobilenet_v2">MobileNetV2 · 3.4M params</option>
              </optgroup>
              <optgroup label="Classic">
                <option value="resnet18">ResNet-18 · 11M params</option>
                <option value="densenet121">DenseNet-121 · 8M params</option>
              </optgroup>
            </select>
          </div>

          {modelType && modelInsights[modelType] && (
            <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
              <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400 mb-1.5">
                Architectural Insight
              </p>
              <p className="text-sm text-gray-500 leading-relaxed">
                {modelInsights[modelType]}
              </p>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={onBack}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg text-sm font-medium text-gray-500 hover:bg-gray-100 disabled:opacity-40 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-[#b80028] text-white text-sm font-medium hover:bg-[#9b0022] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Registering…
              </>
            ) : (
              "Register Client"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
