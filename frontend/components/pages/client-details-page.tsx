"use client";

import { useState } from "react";
import { ArrowLeft, Upload } from "lucide-react";
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
  onBack: () => void;
}

export default function ClientDetailsPage({
  client,
  onBack,
}: ClientDetailsPageProps) {
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
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Inference failed");
      }

      const result = await response.json();

      // Extract Grad-CAM image
      if (result?.xai?.gradcam?.image_base64) {
        setGradcamImage(
          `data:image/png;base64,${result.xai.gradcam.image_base64}`,
        );
      }

      // Extract LIME image
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
    <div className="p-8 max-w-7xl">
      {/* Header */}
      <div className="mb-12">
        <button
          onClick={onBack}
          className="group flex items-center gap-2 text-slate-500 hover:text-slate-900 mb-6 transition-colors"
        >
          <div className="p-2 rounded-full bg-slate-100 group-hover:bg-rose-50 group-hover:text-rose-600 transition-all duration-300">
            <ArrowLeft size={16} />
          </div>
          <span className="font-medium text-sm group-hover:text-rose-700 transition-colors">
            Back to Clients
          </span>
        </button>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-slate-900 tracking-tight mb-3">
              {client.client_name}
            </h1>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-slate-100 rounded-full text-xs font-semibold text-slate-600 uppercase tracking-wide border border-slate-200">
                {client.model_type}
              </span>
              <span className="text-slate-300">•</span>
              <span className="text-sm font-medium text-slate-500">
                Status: <span className="text-slate-700">{client.status}</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Image Upload Section */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-8 mb-12">
        <h2 className="text-xl font-bold text-slate-900 mb-6 tracking-tight">
          Upload Diagnostic Image
        </h2>
        <div className="space-y-8">
          <div className="flex items-center gap-6">
            <label className="cursor-pointer group">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
              <div className="px-5 py-2.5 bg-rose-50 text-rose-600 border border-rose-100 rounded-xl group-hover:bg-rose-100 transition-all font-medium flex items-center gap-2">
                <Upload size={18} />
                Select Image
              </div>
            </label>
            {selectedImage && (
              <span className="text-sm font-medium text-slate-600 bg-slate-50 px-3 py-1.5 rounded-md border border-slate-200">
                {selectedImage.name}
              </span>
            )}
          </div>

          {imagePreview && (
            <div className="border border-slate-200 rounded-xl p-2 bg-slate-50 inline-block">
              <img
                src={imagePreview}
                alt="Preview"
                className="max-h-64 rounded-lg object-contain shadow-sm"
              />
            </div>
          )}

          <div className="pt-2">
            <button
              onClick={handleRunInference}
              disabled={!selectedImage || isRunningInference}
              className="px-8 py-3 bg-[#B80028] text-white rounded-xl font-medium hover:bg-[#960020] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-rose-900/10 hover:shadow-xl active:scale-95 duration-200 flex items-center gap-3"
            >
              {isRunningInference ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>Run Diagnosis</>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {inferenceResult && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Prediction Result */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-8 relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <div className="w-32 h-32 bg-rose-500 rounded-full blur-3xl"></div>
              </div>
              <h2 className="text-lg font-bold text-slate-900 mb-6 tracking-tight">
                Diagnosis Result
              </h2>
              <div className="space-y-8 relative z-10">
                <div>
                  <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
                    Predicted Class
                  </p>
                  <p className="text-4xl font-bold text-slate-900">
                    {inferenceResult.prediction}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
                    Confidence Score
                  </p>
                  <div className="flex items-baseline gap-2">
                    <p
                      className={`text-5xl font-bold ${inferenceResult.confidence > 0.8 ? "text-green-600" : "text-slate-900"}`}
                    >
                      {(inferenceResult.confidence * 100).toFixed(1)}
                      <span className="text-2xl ml-1 text-slate-400">%</span>
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* LLM Explanation */}
            {inferenceResult.llm_explanation && (
              <div className="bg-slate-50/50 rounded-2xl border border-slate-200 p-8">
                <h2 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-3">
                  <span className="block w-2 h-2 rounded-full bg-purple-500 ring-4 ring-purple-100"></span>
                  Clinical Analysis
                </h2>
                <div className="prose prose-slate prose-sm max-w-none">
                  <p className="text-slate-600 leading-relaxed whitespace-pre-wrap">
                    {inferenceResult.llm_explanation}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Explainability Visualizations */}
          {(gradcamImage || limeImage) && (
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-8">
              <div className="flex items-center gap-3 mb-8">
                <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                  Explainability Analysis
                </h2>
                <span className="px-3 py-1 bg-slate-100 text-slate-600 text-xs font-medium rounded-full">
                  AI Reasoning
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Grad-CAM */}
                {gradcamImage && (
                  <div className="space-y-4 group">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>
                        Grad-CAM
                      </h3>
                      <span className="text-xs text-slate-400 font-medium">
                        Attention Heatmap
                      </span>
                    </div>
                    <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 shadow-sm transition-shadow hover:shadow-md">
                      <img
                        src={gradcamImage}
                        alt="Grad-CAM"
                        className="w-full h-auto"
                      />
                    </div>
                    <p className="text-sm text-slate-500 leading-relaxed">
                      Heatmap indicating which regions of the cell structure
                      strongly influenced the model's classification decision.
                    </p>
                  </div>
                )}

                {/* LIME */}
                {limeImage && (
                  <div className="space-y-4 group">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-orange-500"></span>
                        LIME Analysis
                      </h3>
                      <span className="text-xs text-slate-400 font-medium">
                        Feature Importance
                      </span>
                    </div>
                    <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 shadow-sm transition-shadow hover:shadow-md">
                      <img
                        src={limeImage}
                        alt="LIME"
                        className="w-full h-auto"
                      />
                    </div>
                    <p className="text-sm text-slate-500 leading-relaxed">
                      Superpixel segmentation highlighting the exact boundaries
                      and features that contributed positively to the diagnosis.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
