"use client";

import { useState } from "react";
import { ArrowLeft, Download, Upload } from "lucide-react";

interface Client {
  id: number;
  client_name: string;
  client_email: string;
  status: string;
  model_type: number | string;
  created_at: string;
}

interface ClientDetailsPageProps {
  client: Client;
  onBack: () => void;
}

export default function ClientDetailsPage({
  client = {
    id: 1,
    client_name: "Sample Client",
    client_email: "client@example.com",
    status: "Active",
    model_type: "ResNet18",
    created_at: new Date().toISOString(),
  },
  onBack = () => console.log("Back clicked"),
}: ClientDetailsPageProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [formState, setFormState] = useState({
    client_name: client?.client_name || "",
    client_email: client?.client_email || "",
    status: client?.status || "Unknown",
    model_type: String(client?.model_type ?? ""),
  });
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isRunningInference, setIsRunningInference] = useState(false);
  const [inferenceResult, setInferenceResult] = useState<any>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "inference", label: "Inference & XAI" },
    { id: "training", label: "Training Logs" },
  ];

  const handleChange = (field: keyof typeof formState, value: string) => {
    setFormState((prev) => ({ ...prev, [field]: value }));
    setSaveMessage(null);
  };

  const handleSave = async () => {
    try {
      setIsSaving(true);
      setSaveMessage(null);

      const response = await fetch(`http://localhost:8000/api/clients/${client.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_name: formState.client_name,
          client_email: formState.client_email,
          status: formState.status,
          model_type: formState.model_type,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to update client");
      }

      setSaveMessage("Client details updated successfully");
    } catch (error) {
      console.error("Failed to update client:", error);
      setSaveMessage("Failed to update client. Please try again.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setInferenceResult(null);
      setInferenceError(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRunInference = async () => {
    if (!selectedImage) {
      setInferenceError("Please select an image first");
      return;
    }

    try {
      setIsRunningInference(true);
      setInferenceError(null);
      setInferenceResult(null);

      // Create FormData and append the image file
      const formData = new FormData();
      formData.append("file", selectedImage);

      // Send POST request with FormData
      const response = await fetch("http://localhost:8000/predict/upload", {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Inference failed: ${response.statusText} - ${errorText}`
        );
      }

      const result = await response.json();
      setInferenceResult(result);
    } catch (error) {
      console.error("Inference error:", error);
      setInferenceError(
        error instanceof Error ? error.message : "Failed to run inference"
      );
    } finally {
      setIsRunningInference(false);
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

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-gray-900">
            {client.client_name}
          </h1>
          <p className="text-[#718096] mt-2">{client.client_email}</p>
        </div>
        <span className="inline-block px-3 py-1 rounded-full text-sm font-medium border border-[#B80028] text-[#B80028] bg-[rgba(184,0,40,0.05)]">
          {formState.status || "Unknown"}
        </span>
      </div>

      {/* Tabs */}
      <div className="flex gap-8 border-b border-[#E2E8F0] mb-8">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`pb-4 text-sm font-medium transition-colors relative ${
              activeTab === tab.id
                ? "text-gray-900"
                : "text-[#718096] hover:text-gray-900"
            }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-[#B80028]" />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* Editable Client Profile */}
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">
                  Client Profile
                </h2>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="px-4 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg text-sm font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSaving ? "Saving..." : "Save Changes"}
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Organization Name
                  </label>
                  <input
                    type="text"
                    value={formState.client_name}
                    onChange={(e) =>
                      handleChange("client_name", e.target.value)
                    }
                    className="w-full px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Email
                  </label>
                  <input
                    type="email"
                    value={formState.client_email}
                    onChange={(e) =>
                      handleChange("client_email", e.target.value)
                    }
                    className="w-full px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Status
                  </label>
                  <select
                    value={formState.status}
                    onChange={(e) => handleChange("status", e.target.value)}
                    className="w-full px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                  >
                    <option value="Training">Training</option>
                    <option value="Completed">Completed</option>
                    <option value="Idle">Idle</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Model Type
                  </label>
                  <input
                    type="text"
                    value={formState.model_type}
                    onChange={(e) => handleChange("model_type", e.target.value)}
                    className="w-full px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                  />
                </div>
              </div>

              {saveMessage && (
                <p className="mt-4 text-xs text-[#718096]">{saveMessage}</p>
              )}
            </div>

            {/* Client Summary */}
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Client Summary
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Client ID", value: client.id, icon: "🆔" },
                  {
                    label: "Model Type",
                    value: formState.model_type || "-",
                    icon: "🧠",
                  },
                  {
                    label: "Status",
                    value: formState.status || "-",
                    icon: "📡",
                  },
                  {
                    label: "Created",
                    value: new Date(client.created_at).toLocaleDateString(),
                    icon: "🕐",
                  },
                ].map((item, idx) => (
                  <div key={idx}>
                    <p className="text-[#718096] text-xs uppercase tracking-wider flex items-center gap-1">
                      <span>{item.icon}</span>
                      {item.label}
                    </p>
                    <p className="text-xl font-semibold text-gray-900 mt-1">
                      {item.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === "inference" && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Run Inference
              </h2>
              <p className="text-sm text-[#718096] mb-4">
                Upload an image from{" "}
                <span className="font-medium">{client.client_name}</span> to the
                configured model endpoint and inspect the prediction and
                explanation.
              </p>

              <div className="space-y-3 mb-4">
                <label className="block text-xs font-medium text-gray-700">
                  Upload Image
                </label>

                {!imagePreview ? (
                  <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-[#E2E8F0] rounded-lg cursor-pointer hover:border-[#B80028] transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-10 h-10 mb-3 text-[#718096]" />
                      <p className="mb-2 text-sm text-gray-500">
                        <span className="font-semibold">Click to upload</span>{" "}
                        or drag and drop
                      </p>
                      <p className="text-xs text-gray-500">
                        PNG, JPG, JPEG (MAX. 10MB)
                      </p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleImageSelect}
                    />
                  </label>
                ) : (
                  <div className="relative">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="w-full h-auto max-h-96 object-contain rounded-lg border border-[#E2E8F0]"
                    />
                    <button
                      onClick={() => {
                        setSelectedImage(null);
                        setImagePreview(null);
                        setInferenceResult(null);
                        setInferenceError(null);
                      }}
                      className="absolute top-2 right-2 px-3 py-1 bg-white border border-gray-300 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
                      type="button"
                    >
                      Remove
                    </button>
                  </div>
                )}
              </div>

              {inferenceError && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-700">{inferenceError}</p>
                </div>
              )}

              <button
                onClick={handleRunInference}
                disabled={!selectedImage || isRunningInference}
                className="px-4 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg text-sm font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                type="button"
              >
                {isRunningInference ? "Running Inference..." : "Run Inference"}
              </button>
            </div>

            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                {inferenceResult ? "Prediction Results" : "Awaiting Results"}
              </h2>
              <div className="space-y-4">
                {inferenceResult ? (
                  <div className="space-y-3">
                    <div
                      className={`px-4 py-3 rounded-lg border-2 ${
                        inferenceResult.prediction
                          ?.toLowerCase()
                          .includes("healthy") ||
                        inferenceResult.prediction
                          ?.toLowerCase()
                          .includes("hem")
                          ? "border-green-500 bg-green-50"
                          : "border-[#B80028] bg-[rgba(184,0,40,0.05)]"
                      }`}
                    >
                      <p className="text-xs text-gray-600 mb-1">Prediction</p>
                      <p
                        className={`text-lg font-semibold ${
                          inferenceResult.prediction
                            ?.toLowerCase()
                            .includes("healthy") ||
                          inferenceResult.prediction
                            ?.toLowerCase()
                            .includes("hem")
                            ? "text-green-700"
                            : "text-[#B80028]"
                        }`}
                      >
                        {inferenceResult.prediction}
                      </p>
                    </div>

                    {inferenceResult.confidence !== undefined && (
                      <div className="px-4 py-3 rounded-lg border-2 border-blue-500 bg-blue-50">
                        <p className="text-xs text-gray-600 mb-1">Confidence</p>
                        <p className="text-lg font-semibold text-blue-700">
                          {(inferenceResult.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    )}

                    {inferenceResult.model && (
                      <div className="px-4 py-3 rounded-lg border-2 border-purple-500 bg-purple-50">
                        <p className="text-xs text-gray-600 mb-1">Model</p>
                        <p className="text-sm font-medium text-purple-700">
                          {inferenceResult.model}
                        </p>
                      </div>
                    )}

                    {inferenceResult.all_probabilities && (
                      <div className="px-4 py-3 rounded-lg border-2 border-gray-300 bg-gray-50">
                        <p className="text-xs text-gray-600 mb-2">
                          All Classes
                        </p>
                        <div className="space-y-1">
                          {Object.entries(
                            inferenceResult.all_probabilities
                          ).map(([className, prob]: [string, any]) => (
                            <div
                              key={className}
                              className="flex justify-between items-center"
                            >
                              <span className="text-sm text-gray-700">
                                {className}
                              </span>
                              <span className="text-sm font-medium text-gray-900">
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    <p className="text-sm">
                      Upload an image and run inference to see results
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                XAI Heatmap
              </h2>
              <div className="bg-gradient-to-r from-blue-300 via-red-400 to-yellow-300 rounded-lg h-48 flex items-center justify-center">
                <p className="text-white font-medium text-sm bg-black bg-opacity-50 px-4 py-2 rounded">
                  Coming Soon
                </p>
              </div>
              <p className="text-xs text-[#718096] mt-3">
                Feature importance visualization showing model decision drivers.
              </p>
            </div>
          </div>
        )}

        {activeTab === "training" && (
          <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900">
                Training Logs
              </h2>
              <button className="flex items-center gap-2 px-4 py-2 border border-[#B80028] text-[#B80028] rounded-lg hover:bg-[rgba(184,0,40,0.08)] transition-colors text-sm font-medium">
                <Download size={16} />
                Export Report
              </button>
            </div>
            <div className="space-y-2">
              {[
                {
                  time: "2024-12-01 14:32",
                  level: "INFO",
                  message: "Training epoch 50/100 completed",
                },
                {
                  time: "2024-12-01 14:20",
                  level: "INFO",
                  message: "Validation accuracy improved to 94.2%",
                },
                {
                  time: "2024-12-01 14:05",
                  level: "WARNING",
                  message: "Learning rate adjusted",
                },
                {
                  time: "2024-12-01 13:50",
                  level: "ERROR",
                  message: "Detected data drift in batch 45",
                },
              ].map((log, idx) => (
                <div
                  key={idx}
                  className="flex gap-3 px-4 py-3 text-sm border border-[#E2E8F0] rounded-lg"
                >
                  <span className="text-[#718096] min-w-fit">{log.time}</span>
                  <span
                    className={`font-medium ${
                      log.level === "ERROR"
                        ? "text-[#B80028]"
                        : log.level === "WARNING"
                        ? "text-orange-600"
                        : "text-gray-600"
                    }`}
                  >
                    {log.level}
                  </span>
                  <span className="text-gray-900">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
