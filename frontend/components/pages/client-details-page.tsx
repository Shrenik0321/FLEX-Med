import { EmptyState } from "@/components/ui/empty-state";

import { useState } from "react";
import { Upload, Loader2, Activity, FileText } from "lucide-react";
import { API_BASE_PATH } from "@/utils";
import { toast } from "sonner";

interface Client {
  id: number;
  client_name: string;
  model_type: number | string;
  status: string;
}

interface ClientDetailsPageProps {
  client: Client;
}

export default function ClientDetailsPage({ client }: ClientDetailsPageProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isRunningInference, setIsRunningInference] = useState(false);
  const [inferenceResult, setInferenceResult] = useState<any>(null);
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [limeImage, setLimeImage] = useState<string | null>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setInferenceResult(null);
      setGradcamImage(null);
      setLimeImage(null);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleRunInference = async () => {
    if (!selectedImage) {
      toast.error("Please select an image first");
      return;
    }
    try {
      setIsRunningInference(true);
      const formData = new FormData();
      formData.append("file", selectedImage);
      formData.append("client_id", client.id.toString());

      const response = await fetch(`${API_BASE_PATH}/inference`, {
        method: "POST",
        headers: { "ngrok-skip-browser-warning": "true" },
        body: formData,
      });

      if (!response.ok) throw new Error("Inference failed");

      const result = await response.json();

      if (result?.xai?.gradcam?.image_base64) {
        setGradcamImage(`data:image/png;base64,${result.xai.gradcam.image_base64}`);
      }
      if (result?.xai?.lime?.image_base64) {
        setLimeImage(`data:image/png;base64,${result.xai.lime.image_base64}`);
      }

      setInferenceResult(result);
      toast.success("Inference completed successfully");
    } catch (error) {
      console.error("Inference error:", error);
      toast.error("Failed to run inference");
    } finally {
      setIsRunningInference(false);
    }
  };

  return (
    <div className="px-6 py-8 space-y-6">
      {/* Client header */}
      <div className="rounded-xl border border-gray-100 bg-white p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-base font-semibold text-gray-900 tracking-tight">
              {client.client_name}
            </p>
            <div className="flex items-center gap-3 mt-1.5">
              <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
                {client.model_type}
              </span>
              <span className="text-gray-300 select-none">·</span>
              <span
                className={`text-xs font-semibold uppercase tracking-widest ${
                  client.status === "Active"
                    ? "text-green-600"
                    : client.status === "Inactive"
                      ? "text-red-600"
                      : "text-gray-500"
                }`}
              >
                {client.status}
              </span>
            </div>
          </div>
          <span className="text-xs text-gray-400 tabular-nums font-mono">
            ID #{client.id}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: Upload */}
        <div className="lg:col-span-1">
          <div className="rounded-xl border border-gray-100 bg-white p-5">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">
              Upload Image
            </p>
            <div className="space-y-4">
              <div className="relative">
                <input
                  type="file"
                  id="image-upload"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                <label
                  htmlFor="image-upload"
                  className="w-full h-28 border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center gap-2 cursor-pointer hover:border-gray-400 hover:bg-gray-50/60 transition-colors"
                >
                  <Upload className="h-6 w-6 text-gray-500" />
                  <span className="text-sm text-gray-600 font-medium">
                    {selectedImage ? "Change Image" : "Click to upload"}
                  </span>
                </label>
              </div>

              {selectedImage && (
                <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg border border-gray-100">
                  <FileText className="h-3.5 w-3.5 shrink-0 text-gray-500" />
                  <span className="text-sm text-gray-600 truncate">
                    {selectedImage.name}
                  </span>
                </div>
              )}

              {imagePreview && (
                <div className="rounded-lg overflow-hidden border border-gray-100 bg-gray-50">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="w-full h-auto object-contain max-h-44"
                  />
                </div>
              )}

              <button
                onClick={handleRunInference}
                disabled={!selectedImage || isRunningInference}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-[#b80028] text-white text-sm font-medium hover:bg-[#9b0022] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {isRunningInference ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Analyzing…
                  </>
                ) : (
                  "Run Diagnosis"
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-2 space-y-4">
          {!inferenceResult ? (
            <EmptyState
              title="No diagnosis run yet"
              description="Upload a medical image and click 'Run Diagnosis' to see clinical results and AI explainability modules."
              icon={Activity}
              className="h-full min-h-[400px]"
            />
          ) : (
            <div className="space-y-4">
              {/* Prediction result */}
              <div className="rounded-xl border border-gray-100 bg-white p-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-4">
                  Diagnosis Outcome
                </p>
                <div className="space-y-4">
                  <p className="text-2xl font-semibold text-gray-900 tracking-tight">
                    {inferenceResult.prediction}
                  </p>
                  <div className="pt-3 border-t border-gray-50">
                    <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-2">
                      Confidence
                    </p>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${
                            inferenceResult.confidence > 0.8
                              ? "bg-emerald-500"
                              : "bg-amber-500"
                          }`}
                          style={{
                            width: `${inferenceResult.confidence * 100}%`,
                          }}
                        />
                      </div>
                      <span className="text-sm font-semibold text-gray-900 tabular-nums">
                        {(inferenceResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {inferenceResult.llm_explanation && (
                <div className="rounded-xl border border-gray-100 bg-white p-5">
                  <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
                    Clinical Insight
                  </p>
                  <p className="text-sm text-gray-600 leading-relaxed italic">
                    "{inferenceResult.llm_explanation}"
                  </p>
                </div>
              )}

              {/* XAI Visualizations */}
              {(gradcamImage || limeImage) && (
                <div className="rounded-xl border border-gray-100 bg-white p-5">
                  <div className="flex items-center justify-between mb-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-gray-500">
                      Explainability Modules
                    </p>
                    <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold border text-gray-600 bg-gray-50 border-gray-200">
                      Model Reasoning
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    {gradcamImage && (
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-gray-700">
                          Grad-CAM Heatmap
                        </p>
                        <div className="rounded-lg overflow-hidden border border-gray-100">
                          <img
                            src={gradcamImage}
                            alt="Grad-CAM"
                            className="w-full h-auto"
                          />
                        </div>
                        <p className="text-xs text-gray-500 leading-relaxed">
                          Visualizes regions with strongest gradient influence on
                          the model's prediction.
                        </p>
                      </div>
                    )}
                    {limeImage && (
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-gray-700">
                          LIME Local Features
                        </p>
                        <div className="rounded-lg overflow-hidden border border-gray-100">
                          <img
                            src={limeImage}
                            alt="LIME"
                            className="w-full h-auto"
                          />
                        </div>
                        <p className="text-xs text-gray-500 leading-relaxed">
                          Highlights superpixels contributing to the local
                          classification via perturbation analysis.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
