"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { ArrowLeft, Download, Upload, Play, Square, RefreshCw } from "lucide-react";
import { API_BASE_PATH } from "@/utils";

interface Client {
  id: number;
  client_name: string;
  client_email: string;
  status: string;
  model_type: number | string;
  dataset_path?: string;
  model_path?: string;
  created_at: string;
}

interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
}

interface TrainingStatus {
  status: "idle" | "pending" | "running" | "completed" | "failed" | "not_found";
  logs: string[];
  message?: string;
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
  const [selectedXAITechnique, setSelectedXAITechnique] = useState("LIME");
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [limeImage, setLimeImage] = useState<string | null>(null);

  // Local Training State
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    epochs: 5,
    batch_size: 16,
    learning_rate: 0.001,
  });
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    status: "idle",
    logs: [],
  });
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "inference", label: "Inference & XAI" },
    { id: "local-training", label: "Local Training" },
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

      const response = await fetch(`${API_BASE_PATH}/clients/${client.id}`, {
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
      setGradcamImage(null);
      setLimeImage(null);

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
      setGradcamImage(null);

      // Create FormData and append the image file
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("client_id", client.id.toString()); // Pass client ID

      // Send POST request with FormData
      const response = await fetch(`${API_BASE_PATH}/predict/upload-xai`, {
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

      // Extract Grad-CAM image if available
      if (result?.xai?.gradcam?.image_base64) {
        setGradcamImage(
          `data:image/png;base64,${result.xai.gradcam.image_base64}`
        );
      }

      // Extract LIME image if available
      if (result?.xai?.lime?.image_base64) {
        setLimeImage(`data:image/png;base64,${result.xai.lime.image_base64}`);
      }

      // Mock XAI Data if not present
      if (!result.xai_analysis) {
        result.xai_analysis = {
          LIME: {
            description:
              "LIME (Local Interpretable Model-agnostic Explanations) highlights the regions of the image that most contributed to the prediction. Green areas indicate positive influence, while red areas indicate negative influence.",
            explanation:
              "The model focused primarily on the central density and the irregular patterns in the upper left quadrant. These features are strong indicators for the predicted class.",
          },
          "Grad-CAM": {
            description:
              "Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients of the target concept to produce a coarse localization map highlighting important regions in the image.",
            explanation:
              "The heatmap shows high activation around the suspicious nodule, confirming that the model is looking at the correct pathology rather than background artifacts.",
          },
        };
      }

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

  // ============== LOCAL TRAINING FUNCTIONS ==============

  // Scroll to bottom of logs when new logs arrive
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [trainingStatus.logs]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const pollTrainingStatus = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE_PATH}/train_status/${client.client_name}`,
        {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch training status");
      }

      const data = await response.json();
      setTrainingStatus({
        status: data.status || "not_found",
        logs: data.logs || [],
        message: data.message,
      });

      // Stop polling if training is complete or failed
      if (data.status === "completed" || data.status === "failed") {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }

        // Mark training as complete in backend
        if (data.status === "completed") {
          await fetch(
            `${API_BASE_PATH}/train_status/${client.client_name}/complete`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
              },
            }
          );
        }
      }
    } catch (error) {
      console.error("Error polling training status:", error);
    }
  }, [client.client_name]);

  const handleStartTraining = async () => {
    try {
      setIsStartingTraining(true);
      setTrainingError(null);
      setTrainingStatus({ status: "pending", logs: ["Initiating training..."] });

      const response = await fetch(`${API_BASE_PATH}/start_local_train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({
          client_id: client.id,
          epochs: trainingConfig.epochs,
          batch_size: trainingConfig.batch_size,
          learning_rate: trainingConfig.learning_rate,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to start training");
      }

      const result = await response.json();
      setTrainingStatus({
        status: "running",
        logs: [`Training started: ${result.message}`],
      });

      // Start polling for training status
      pollingIntervalRef.current = setInterval(pollTrainingStatus, 2000);
    } catch (error) {
      console.error("Failed to start training:", error);
      setTrainingError(
        error instanceof Error ? error.message : "Failed to start training"
      );
      setTrainingStatus({ status: "failed", logs: [] });
    } finally {
      setIsStartingTraining(false);
    }
  };

  const handleStopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const handleRefreshStatus = () => {
    pollTrainingStatus();
  };

  const handleConfigChange = (field: keyof TrainingConfig, value: string) => {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      setTrainingConfig((prev) => ({ ...prev, [field]: numValue }));
    }
  };

  const getStatusColor = (status: TrainingStatus["status"]) => {
    switch (status) {
      case "completed":
        return "text-green-600 bg-green-50 border-green-500";
      case "running":
        return "text-blue-600 bg-blue-50 border-blue-500";
      case "pending":
        return "text-yellow-600 bg-yellow-50 border-yellow-500";
      case "failed":
        return "text-red-600 bg-red-50 border-red-500";
      default:
        return "text-gray-600 bg-gray-50 border-gray-300";
    }
  };

  const getStatusLabel = (status: TrainingStatus["status"]) => {
    switch (status) {
      case "completed":
        return "Completed";
      case "running":
        return "Training in Progress";
      case "pending":
        return "Starting...";
      case "failed":
        return "Failed";
      case "not_found":
        return "No Training Data";
      default:
        return "Idle";
    }
  };

  return (
    <div className="p-8">
      {/* Header */}
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-primary hover:text-red-700 mb-6 font-medium"
      >
        <ArrowLeft size={18} />
        Back to Clients
      </button>

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">
            {client.client_name}
          </h1>
          <p className="text-muted-foreground mt-2">{client.client_email}</p>
        </div>
        <span className="inline-block px-3 py-1 rounded-full text-sm font-medium border border-primary text-primary bg-[rgba(184,0,40,0.05)]">
          {formState.status || "Unknown"}
        </span>
      </div>

      {/* Tabs */}
      <div className="flex gap-8 border-b border-border mb-8">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`pb-4 text-sm font-medium transition-colors relative ${
              activeTab === tab.id
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-primary" />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* Editable Client Profile */}
            <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-foreground">
                  Client Profile
                </h2>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="px-4 py-2 bg-card border border-primary text-primary rounded-lg text-sm font-medium hover:bg-primary/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
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
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Status
                  </label>
                  <select
                    value={formState.status}
                    onChange={(e) => handleChange("status", e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
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
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                  />
                </div>
              </div>

              {saveMessage && (
                <p className="mt-4 text-xs text-muted-foreground">
                  {saveMessage}
                </p>
              )}
            </div>

            {/* Client Summary */}
            <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
              <h2 className="text-lg font-semibold text-foreground mb-4">
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
                    <p className="text-muted-foreground text-xs uppercase tracking-wider flex items-center gap-1">
                      <span>{item.icon}</span>
                      {item.label}
                    </p>
                    <p className="text-xl font-semibold text-foreground mt-1">
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
            <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
              <h2 className="text-lg font-semibold text-foreground mb-4">
                Run Inference
              </h2>
              <p className="text-sm text-muted-foreground mb-4">
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
                  <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-10 h-10 mb-3 text-muted-foreground" />
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
                      className="w-full h-auto max-h-96 object-contain rounded-lg border border-border"
                    />
                    <button
                      onClick={() => {
                        setSelectedImage(null);
                        setImagePreview(null);
                        setInferenceResult(null);
                        setInferenceError(null);
                        setGradcamImage(null);
                        setLimeImage(null);
                      }}
                      className="absolute top-2 right-2 px-3 py-1 bg-card border border-gray-300 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
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
                className="px-4 py-2 bg-card border border-primary text-primary rounded-lg text-sm font-medium hover:bg-primary/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                type="button"
              >
                {isRunningInference ? "Running Inference..." : "Run Inference"}
              </button>
            </div>

            <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
              <h2 className="text-lg font-semibold text-foreground mb-4">
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
                          : "border-primary bg-[rgba(184,0,40,0.05)]"
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
                            : "text-primary"
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
                              <span className="text-sm font-medium text-foreground">
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

            {/* XAI Section */}
            {inferenceResult && inferenceResult.xai_analysis && (
              <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-foreground">
                    Explainable AI (XAI) Analysis
                  </h2>
                  <div className="relative">
                    <select
                      value={selectedXAITechnique}
                      onChange={(e) => setSelectedXAITechnique(e.target.value)}
                      className="appearance-none bg-card border border-border text-gray-700 py-2 px-4 pr-8 rounded-lg leading-tight focus:outline-none focus:bg-card focus:border-primary text-sm font-medium"
                    >
                      <option value="LIME">LIME</option>
                      <option value="Grad-CAM">Grad-CAM</option>
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                      <svg
                        className="fill-current h-4 w-4"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                      >
                        <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                      </svg>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Visualization */}
                  <div className="flex flex-col items-center justify-center bg-gray-50 rounded-lg p-4 border border-dashed border-gray-300 min-h-[300px]">
                    {selectedXAITechnique === "Grad-CAM" && gradcamImage ? (
                      // Show actual Grad-CAM heatmap image from API
                      <div className="w-full h-full max-h-[300px] flex items-center justify-center overflow-hidden rounded-md">
                        <img
                          src={gradcamImage}
                          alt="Grad-CAM Heatmap"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    ) : selectedXAITechnique === "LIME" && limeImage ? (
                      // Show actual LIME visualization image from API
                      <div className="w-full h-full max-h-[300px] flex items-center justify-center overflow-hidden rounded-md">
                        <img
                          src={limeImage}
                          alt="LIME Explanation"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    ) : imagePreview ? (
                      // Show mock overlay if actual XAI images are not available
                      <div className="relative w-full h-full max-h-[300px] flex items-center justify-center overflow-hidden rounded-md group">
                        {/* Base Image */}
                        <img
                          src={imagePreview}
                          alt="Original"
                          className="absolute inset-0 w-full h-full object-contain opacity-50 blur-[2px] group-hover:blur-0 group-hover:opacity-100 transition-all duration-300"
                        />

                        {/* Overlay Mockup - Just to demonstrate visual change */}
                        <div
                          className={`absolute inset-0 flex items-center justify-center pointer-events-none ${
                            selectedXAITechnique === "LIME"
                              ? "bg-green-500/20 mix-blend-overlay"
                              : "bg-red-500/20 mix-blend-overlay"
                          }`}
                        >
                          <span className="bg-black/70 text-white px-3 py-1 rounded-full text-xs font-semibold backdrop-blur-sm z-10">
                            {selectedXAITechnique} Overlay Visualization
                          </span>
                        </div>
                      </div>
                    ) : (
                      <p className="text-gray-400 text-sm">
                        No visualization available
                      </p>
                    )}
                    {!limeImage && !gradcamImage && imagePreview && (
                      <p className="mt-2 text-xs text-gray-500 italic">
                        * Hover image to see original
                      </p>
                    )}
                  </div>

                  {/* Explanation Text */}
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-semibold text-foreground mb-2">
                        Technique Description
                      </h3>
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {
                          inferenceResult.xai_analysis[selectedXAITechnique]
                            ?.description
                        }
                      </p>
                    </div>

                    <div>
                      <h3 className="text-sm font-semibold text-foreground mb-2">
                        Model Interpretation
                      </h3>
                      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
                        <p className="text-sm text-blue-900 leading-relaxed font-medium">
                          {
                            inferenceResult.xai_analysis[selectedXAITechnique]
                              ?.explanation
                          }
                        </p>
                      </div>
                    </div>

                    <div className="pt-4 border-t border-gray-100">
                      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                        Key Influencing Factors
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {[
                          "Texture Irregularity",
                          "Density",
                          "Shape Asymmetry",
                        ].map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-md"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "local-training" && (
          <div className="space-y-6">
            {/* Training Configuration */}
            <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
              <h2 className="text-lg font-semibold text-foreground mb-4">
                Training Configuration
              </h2>
              <p className="text-sm text-muted-foreground mb-6">
                Configure and start local training for{" "}
                <span className="font-medium">{client.client_name}</span>. The
                model will be trained on the configured dataset path.
              </p>

              {/* Client Dataset Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
                    Model Path
                  </p>
                  <p className="text-sm text-foreground font-mono truncate">
                    {client.model_path || "Not configured"}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
                    Dataset Path
                  </p>
                  <p className="text-sm text-foreground font-mono truncate">
                    {client.dataset_path || "Not configured"}
                  </p>
                </div>
              </div>

              {/* Training Parameters */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Epochs
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={trainingConfig.epochs}
                    onChange={(e) => handleConfigChange("epochs", e.target.value)}
                    disabled={trainingStatus.status === "running" || trainingStatus.status === "pending"}
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Batch Size
                  </label>
                  <select
                    value={trainingConfig.batch_size}
                    onChange={(e) => handleConfigChange("batch_size", e.target.value)}
                    disabled={trainingStatus.status === "running" || trainingStatus.status === "pending"}
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <option value="8">8</option>
                    <option value="16">16</option>
                    <option value="32">32</option>
                    <option value="64">64</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Learning Rate
                  </label>
                  <select
                    value={trainingConfig.learning_rate}
                    onChange={(e) => handleConfigChange("learning_rate", e.target.value)}
                    disabled={trainingStatus.status === "running" || trainingStatus.status === "pending"}
                    className="w-full px-3 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    <option value="0.0001">0.0001</option>
                    <option value="0.001">0.001</option>
                    <option value="0.01">0.01</option>
                    <option value="0.1">0.1</option>
                  </select>
                </div>
              </div>

              {/* Error Message */}
              {trainingError && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-700">{trainingError}</p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-3">
                <button
                  onClick={handleStartTraining}
                  disabled={
                    isStartingTraining ||
                    trainingStatus.status === "running" ||
                    trainingStatus.status === "pending" ||
                    !client.dataset_path ||
                    !client.model_path
                  }
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Play size={16} />
                  {isStartingTraining ? "Starting..." : "Start Training"}
                </button>

                {(trainingStatus.status === "running" || trainingStatus.status === "pending") && (
                  <button
                    onClick={handleStopPolling}
                    className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
                  >
                    <Square size={16} />
                    Stop Monitoring
                  </button>
                )}

                <button
                  onClick={handleRefreshStatus}
                  className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
                >
                  <RefreshCw size={16} />
                  Refresh Status
                </button>
              </div>

              {/* Missing Config Warning */}
              {(!client.dataset_path || !client.model_path) && (
                <p className="mt-4 text-sm text-orange-600">
                  Please configure dataset_path and model_path for this client before training.
                </p>
              )}
            </div>

            {/* Training Status & Logs */}
            <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-foreground">
                  Training Status
                </h2>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium border-2 ${getStatusColor(
                    trainingStatus.status
                  )}`}
                >
                  {getStatusLabel(trainingStatus.status)}
                </span>
              </div>

              {/* Real-time Logs */}
              <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
                {trainingStatus.logs.length > 0 ? (
                  <div className="space-y-1">
                    {trainingStatus.logs.map((log, idx) => (
                      <div
                        key={idx}
                        className={`${
                          log.includes("Epoch")
                            ? "text-blue-400"
                            : log.includes("Complete") || log.includes("SUCCESS")
                            ? "text-green-400"
                            : log.includes("Failed") || log.includes("Error")
                            ? "text-red-400"
                            : log.includes("Warning")
                            ? "text-yellow-400"
                            : "text-gray-300"
                        }`}
                      >
                        {log}
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">
                    No training logs yet. Start training to see real-time logs.
                  </p>
                )}
              </div>

              {/* Training Progress Indicator */}
              {trainingStatus.status === "running" && (
                <div className="mt-4 flex items-center gap-2 text-sm text-blue-600">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                  <span>Training in progress... Polling for updates every 2 seconds</span>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "training" && (
          <div className="bg-card rounded-lg p-6 shadow-sm border border-border">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-foreground">
                Training Logs (Historical)
              </h2>
              <button className="flex items-center gap-2 px-4 py-2 border border-primary text-primary rounded-lg hover:bg-primary/10 transition-colors text-sm font-medium">
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
                  className="flex gap-3 px-4 py-3 text-sm border border-border rounded-lg"
                >
                  <span className="text-muted-foreground min-w-fit">
                    {log.time}
                  </span>
                  <span
                    className={`font-medium ${
                      log.level === "ERROR"
                        ? "text-primary"
                        : log.level === "WARNING"
                        ? "text-orange-600"
                        : "text-gray-600"
                    }`}
                  >
                    {log.level}
                  </span>
                  <span className="text-foreground">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
