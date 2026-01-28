"use client";

import { API_BASE_PATH } from "@/utils";
import { ArrowLeft, Upload, Check, X, FileJson, Loader2 } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

interface AddClientPageProps {
  onBack: () => void;
}

export default function AddClientPage({ onBack }: AddClientPageProps) {
  const [clientName, setClientName] = useState("");
  const [modelType, setModelType] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasDataset, setHasDataset] = useState(false);

  const handleSubmit = async () => {
    if (!clientName || !modelType) {
      setError("Please fill in all required fields");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_PATH}/clients`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          client_name: clientName,
          model_type: modelType,
          has_local_data: hasDataset,
          dataset_path: hasDataset ? undefined : null,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to register client");
      }

      const data = await response.json();
      console.log("Client registered:", data);

      // Navigate back to clients list after successful registration
      onBack();
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      {/* Form */}
      <div className="max-w-2xl">
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
                <optgroup label="ResNet">
                  <option value="resnet18">
                    ResNet-18 - Lightweight residual network
                  </option>
                  <option value="resnet34">
                    ResNet-34 - Deeper residual network
                  </option>
                  <option value="resnet50">
                    ResNet-50 - Standard residual network
                  </option>
                  <option value="resnet101">
                    ResNet-101 - Deep residual network
                  </option>
                  <option value="resnet152">
                    ResNet-152 - Very deep residual network
                  </option>
                </optgroup>
                <optgroup label="VGG">
                  <option value="vgg11">
                    VGG-11 - Lightweight VGG architecture
                  </option>
                  <option value="vgg13">
                    VGG-13 - Medium VGG architecture
                  </option>
                  <option value="vgg16">
                    VGG-16 - Standard VGG architecture
                  </option>
                  <option value="vgg19">
                    VGG-19 - Deepest VGG architecture
                  </option>
                </optgroup>
                <optgroup label="DenseNet">
                  <option value="densenet121">
                    DenseNet-121 - Efficient dense connections
                  </option>
                  <option value="densenet161">
                    DenseNet-161 - Wide dense network
                  </option>
                  <option value="densenet169">
                    DenseNet-169 - Deep dense network
                  </option>
                  <option value="densenet201">
                    DenseNet-201 - Very deep dense network
                  </option>
                </optgroup>
                <optgroup label="EfficientNet">
                  <option value="efficientnet_b0">
                    EfficientNet-B0 - Smallest efficient model
                  </option>
                  <option value="efficientnet_b1">
                    EfficientNet-B1 - Balanced efficiency
                  </option>
                  <option value="efficientnet_b2">
                    EfficientNet-B2 - Enhanced efficiency
                  </option>
                  <option value="efficientnet_b3">
                    EfficientNet-B3 - Advanced efficiency
                  </option>
                  <option value="efficientnet_b4">
                    EfficientNet-B4 - High efficiency
                  </option>
                  <option value="efficientnet_b5">
                    EfficientNet-B5 - Very high efficiency
                  </option>
                  <option value="efficientnet_b6">
                    EfficientNet-B6 - Ultra efficiency
                  </option>
                  <option value="efficientnet_b7">
                    EfficientNet-B7 - Maximum efficiency
                  </option>
                </optgroup>
                <optgroup label="MobileNet">
                  <option value="mobilenet_v2">
                    MobileNet-V2 - Mobile-optimized network
                  </option>
                  <option value="mobilenet_v3_small">
                    MobileNet-V3 Small - Compact mobile model
                  </option>
                  <option value="mobilenet_v3_large">
                    MobileNet-V3 Large - Enhanced mobile model
                  </option>
                </optgroup>
                <optgroup label="Inception">
                  <option value="inception_v3">
                    Inception-V3 - Multi-scale feature extraction
                  </option>
                  <option value="googlenet">
                    GoogLeNet - Original inception architecture
                  </option>
                </optgroup>
                <optgroup label="Other">
                  <option value="alexnet">AlexNet - Classic deep CNN</option>
                  <option value="squeezenet">
                    SqueezeNet - Compact architecture
                  </option>
                </optgroup>
              </select>
            </div>
          </div>

          {/* Data Upload */}
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Initial Dataset
            </h2>
            {hasDataset ? (
              <div className="relative border-2 border-solid border-primary bg-primary/5 rounded-lg p-8 text-center transition-all">
                <button
                  onClick={() => setHasDataset(false)}
                  className="absolute top-4 right-4 p-1 text-primary hover:bg-red-100 rounded-full transition-colors"
                  title="Remove dataset"
                >
                  <X size={20} />
                </button>
                <div className="w-16 h-16 bg-card rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm border border-red-100">
                  <FileJson className="text-primary" size={32} />
                </div>
                <h3 className="font-semibold text-foreground mb-1">
                  Dataset Configured
                </h3>
                <p className="text-sm text-muted-foreground">
                  client_allidb (Default Local Dataset)
                </p>
                <div className="flex items-center justify-center gap-2 mt-4 text-sm text-primary font-medium">
                  <Check size={16} />
                  Ready for training
                </div>
              </div>
            ) : (
              <div
                onClick={() => setHasDataset(true)}
                className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary hover:bg-primary/5 transition-all cursor-pointer group"
              >
                <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-card group-hover:shadow-sm transition-all">
                  <Upload
                    className="text-muted-foreground group-hover:text-primary transition-colors"
                    size={32}
                  />
                </div>
                <p className="font-medium text-foreground group-hover:text-primary transition-colors">
                  Click to configure local dataset
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Uses default path: /content/drive/MyDrive/...
                </p>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <Button variant="ghost" onClick={onBack} disabled={loading}>
              Cancel
            </Button>
            <Button onClick={handleSubmit} disabled={loading} className="px-8">
              {loading ? (
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
