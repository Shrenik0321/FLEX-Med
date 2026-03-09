"use client";

import { API_BASE_PATH } from "@/utils";
import { Upload, Check, X, FileJson, Loader2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

interface AddClientPageProps {
  onBack: () => void;
}

import { toast } from "sonner";
// ...

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
    const toastId = toast.loading("Registering client...");

    try {
      const response = await fetch(`${API_BASE_PATH}/clients`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          client_name: clientName,
          model_type: modelType,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to register client");
      }

      toast.success("Client registered successfully!", { id: toastId });

      // Navigate back to clients list after successful registration
      onBack();
    } catch (err) {
      console.error(err);
      toast.error(err instanceof Error ? err.message : "An error occurred", {
        id: toastId,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-8">
      {/* Form */}
      <div className="w-full">
        <div className="bg-card rounded-lg p-8 shadow-sm border border-border space-y-6">
          {/* Organization Info */}
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Organization Information
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Organization Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., City Medical Center"
                  value={clientName}
                  onChange={(e) => setClientName(e.target.value)}
                  className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                />
              </div>
            </div>
          </div>

          {/* Model Selection */}
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Select Model
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  CNN Model Architecture
                </label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                >
                  <option value="">Select a PyTorch CNN model...</option>
                  <option value="efficientnet_b0">
                    EfficientNet-B0 (Lightweight EfficientNet)
                  </option>
                  <option value="mobilenet_v2">
                    MobileNet-V2 (Mobile Optimized)
                  </option>
                  <option value="densenet121">
                    DenseNet-121 (High Dense Connections)
                  </option>
                </select>
              </div>

              {/* Architectural Insight Note */}
              {modelType && (
                <div className="p-4 bg-muted/50 rounded-lg border border-border">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-muted-foreground mb-2">
                    Architectural Insight
                  </h4>
                  {modelType === "efficientnet_b0" && (
                    <p className="text-sm text-foreground">
                      <strong>Lightweight EfficientNet:</strong> Uses compound
                      scaling to balance depth, width, and resolution. Provides
                      high accuracy with very few parameters, excellent for
                      preventing overfitting on small medical datasets.
                    </p>
                  )}
                  {modelType === "mobilenet_v2" && (
                    <p className="text-sm text-foreground">
                      <strong>Mobile Optimized:</strong> Uses depthwise
                      separable convolutions to reduce parameter count. Ideal
                      for clinical point-of-care devices with limited compute
                      power.
                    </p>
                  )}
                  {modelType === "densenet121" && (
                    <p className="text-sm text-foreground">
                      <strong>High Dense Connections:</strong> Connects every
                      layer to every other layer in a dense block. Superior at
                      feature reuse for capturing subtle textures in blood smear
                      images.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <Button variant="ghost" onClick={onBack} disabled={isLoading}>
              Cancel
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={isLoading}
              className="px-8"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Registering...
                </>
              ) : (
                "Register Client"
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
