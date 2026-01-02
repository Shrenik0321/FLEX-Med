"use client";

import { ArrowLeft, Upload, Check, X, FileJson } from "lucide-react";
import { useState } from "react";

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
      const response = await fetch("http://localhost:8000/api/clients", {
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
      {/* Header */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-[#B80028] hover:text-red-700 mb-6 font-medium"
      >
        <ArrowLeft size={18} />
        Back to Clients
      </button>

      <h1 className="text-3xl font-semibold text-gray-900 mb-2">
        Add New Client
      </h1>
      <p className="text-[#718096] mb-8">
        Register and configure a new healthcare client for federated learning
      </p>

      {/* Form */}
      <div className="max-w-2xl">
        <div className="bg-white rounded-lg p-8 flex-card-shadow space-y-6">
          {/* Organization Info */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Organization Information
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Organization Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., City Medical Center"
                  value={clientName}
                  onChange={(e) => setClientName(e.target.value)}
                  className="w-full px-4 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                />
              </div>
            </div>
          </div>

          {/* Model Selection */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Select Model
            </h2>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                CNN Model Architecture
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full px-4 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
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
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Initial Dataset
            </h2>
            {hasDataset ? (
              <div className="relative border-2 border-solid border-[#B80028] bg-[rgba(184,0,40,0.02)] rounded-lg p-8 text-center transition-all">
                <button
                  onClick={() => setHasDataset(false)}
                  className="absolute top-4 right-4 p-1 text-[#B80028] hover:bg-red-100 rounded-full transition-colors"
                  title="Remove dataset"
                >
                  <X size={20} />
                </button>
                <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm border border-red-100">
                  <FileJson className="text-[#B80028]" size={32} />
                </div>
                <h3 className="font-semibold text-gray-900 mb-1">
                  Dataset Configured
                </h3>
                <p className="text-sm text-[#718096]">
                  client_allidb (Default Local Dataset)
                </p>
                <div className="flex items-center justify-center gap-2 mt-4 text-sm text-[#B80028] font-medium">
                  <Check size={16} />
                  Ready for training
                </div>
              </div>
            ) : (
              <div
                onClick={() => setHasDataset(true)}
                className="border-2 border-dashed border-[#E2E8F0] rounded-lg p-8 text-center hover:border-[#B80028] hover:bg-[rgba(184,0,40,0.02)] transition-all cursor-pointer group"
              >
                <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-white group-hover:shadow-sm transition-all">
                  <Upload
                    className="text-[#718096] group-hover:text-[#B80028] transition-colors"
                    size={32}
                  />
                </div>
                <p className="font-medium text-gray-900 group-hover:text-[#B80028] transition-colors">
                  Click to configure local dataset
                </p>
                <p className="text-xs text-[#718096] mt-2">
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
            <button
              onClick={onBack}
              className="px-6 py-2 border border-[#E2E8F0] text-gray-900 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="px-6 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Registering..." : "Register Client"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
