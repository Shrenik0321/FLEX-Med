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

      const response = await fetch(`${API_BASE_PATH}/predict/upload-xai`, {
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
          `data:image/png;base64,${result.xai.gradcam.image_base64}`
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
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          Back to Clients
        </button>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold text-foreground">
              {client.client_name}
            </h1>
            <p className="text-muted-foreground mt-1">
              Model: {client.model_type} • Status: {client.status}
            </p>
          </div>
        </div>
      </div>

      {/* Image Upload Section */}
      <div className="bg-card rounded-lg shadow-sm border border-border p-6 mb-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">
          Upload Blood Cell Image
        </h2>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="cursor-pointer">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
              <div className="px-4 py-2 bg-card border border-primary text-primary rounded-lg hover:bg-primary/10 transition-colors flex items-center gap-2">
                <Upload size={18} />
                Select Image
              </div>
            </label>
            {selectedImage && (
              <span className="text-sm text-muted-foreground">
                {selectedImage.name}
              </span>
            )}
          </div>

          {imagePreview && (
            <div className="border border-border rounded-lg p-4 bg-gray-50">
              <p className="text-sm font-medium text-foreground mb-2">
                Preview:
              </p>
              <img
                src={imagePreview}
                alt="Preview"
                className="max-w-xs rounded border border-border"
              />
            </div>
          )}

          <button
            onClick={handleRunInference}
            disabled={!selectedImage || isRunningInference}
            className="px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunningInference ? "Analyzing..." : "Run Diagnosis"}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {inferenceResult && (
        <div className="space-y-6">
          {/* Prediction Result */}
          <div className="bg-card rounded-lg shadow-sm border border-border p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4">
              Diagnosis Result
            </h2>
            <div className="space-y-3">
              <div>
                <p className="text-sm text-muted-foreground">Prediction</p>
                <p className="text-2xl font-semibold text-foreground">
                  {inferenceResult.prediction}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Confidence</p>
                <p className="text-xl font-semibold text-foreground">
                  {(inferenceResult.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* LLM Explanation */}
          {inferenceResult.llm_explanation && (
            <div className="bg-card rounded-lg shadow-sm border border-border p-6">
              <h2 className="text-lg font-semibold text-foreground mb-4">
                Clinical Explanation
              </h2>
              <div className="prose prose-sm max-w-none">
                <p className="text-foreground whitespace-pre-wrap leading-relaxed">
                  {inferenceResult.llm_explanation}
                </p>
              </div>
            </div>
          )}

          {/* Explainability Visualizations */}
          {(gradcamImage || limeImage) && (
            <div className="bg-card rounded-lg shadow-sm border border-border p-6">
              <h2 className="text-lg font-semibold text-foreground mb-4">
                Explainability Analysis
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Grad-CAM */}
                {gradcamImage && (
                  <div>
                    <h3 className="text-sm font-semibold text-foreground mb-2">
                      Grad-CAM (Attention Heatmap)
                    </h3>
                    <p className="text-xs text-muted-foreground mb-3">
                      Shows which regions of the image the model focused on for
                      the diagnosis
                    </p>
                    <img
                      src={gradcamImage}
                      alt="Grad-CAM"
                      className="w-full rounded border border-border"
                    />
                  </div>
                )}

                {/* LIME */}
                {limeImage && (
                  <div>
                    <h3 className="text-sm font-semibold text-foreground mb-2">
                      LIME (Feature Importance)
                    </h3>
                    <p className="text-xs text-muted-foreground mb-3">
                      Highlights image regions that contributed most to the
                      prediction
                    </p>
                    <img
                      src={limeImage}
                      alt="LIME"
                      className="w-full rounded border border-border"
                    />
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
