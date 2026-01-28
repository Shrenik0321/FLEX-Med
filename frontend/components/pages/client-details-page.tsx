"use client";

import { useState } from "react";
import { Upload, Loader2, Microscope, Activity, FileText } from "lucide-react";
import { API_BASE_PATH } from "@/utils";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

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
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      {/* Client Overview Card */}
      <div className="bg-card rounded-2xl shadow-sm border border-border p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <h1 className="text-3xl font-bold text-foreground tracking-tight">
              {client.client_name}
            </h1>
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <span className="flex items-center gap-1.5 px-2 py-0.5 bg-muted rounded-md border border-border font-medium text-foreground capitalize">
                <Microscope className="h-3.5 w-3.5" />
                {client.model_type}
              </span>
              <span>•</span>
              <span className="flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                Status:{" "}
                <span className="text-foreground font-medium">
                  {client.status}
                </span>
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <p className="text-xs text-muted-foreground font-mono">
              ID: {client.id}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 text-foreground">
        {/* Left Column: Analysis Controls */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-card rounded-2xl shadow-sm border border-border p-6 overflow-hidden">
            <h2 className="text-lg font-bold mb-6 flex items-center gap-2">
              <Upload className="h-5 w-5 text-primary" />
              Upload Image
            </h2>

            <div className="space-y-6">
              <div className="relative group">
                <input
                  type="file"
                  id="image-upload"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                <Button
                  asChild
                  variant="outline"
                  className="w-full h-32 border-dashed border-2 hover:border-primary/50 hover:bg-primary/5 cursor-pointer flex flex-col gap-2 transition-all bg-transparent"
                >
                  <label htmlFor="image-upload">
                    <Upload className="h-8 w-8 text-muted-foreground group-hover:text-primary transition-colors" />
                    <span className="text-sm font-medium">
                      {selectedImage ? "Change Image" : "Click to upload"}
                    </span>
                  </label>
                </Button>
              </div>

              {selectedImage && (
                <div className="p-3 bg-muted/50 rounded-lg border border-border flex items-center justify-between overflow-hidden">
                  <div className="flex items-center gap-2 overflow-hidden mx-auto">
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <span className="text-xs font-medium truncate">
                      {selectedImage.name}
                    </span>
                  </div>
                </div>
              )}

              {imagePreview && (
                <div className="rounded-xl overflow-hidden border border-border bg-slate-100/50 p-2">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="w-full h-auto rounded-lg object-contain shadow-sm max-h-48 mx-auto"
                  />
                </div>
              )}

              <Button
                onClick={handleRunInference}
                disabled={!selectedImage || isRunningInference}
                className="w-full h-11 text-base shadow-sm"
              >
                {isRunningInference ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Run Diagnosis"
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Right Column: Results & XAI */}
        <div className="lg:col-span-2 space-y-6">
          {!inferenceResult ? (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center bg-card/30 border-2 border-dashed border-border rounded-2xl text-muted-foreground p-8 text-center">
              <Activity className="h-12 w-12 mb-4 opacity-20" />
              <p className="text-lg font-medium">No diagnosis run yet</p>
              <p className="text-sm max-w-xs mt-1">
                Upload a medical image and click "Run Diagnosis" to see clinical
                results and AI explainability modules.
              </p>
            </div>
          ) : (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* Primary Results Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-card rounded-2xl shadow-sm border border-border p-6 overflow-hidden relative">
                  <div className="absolute -right-4 -top-4 w-24 h-24 bg-primary/5 rounded-full blur-2xl"></div>
                  <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">
                    Diagnosis Outcome
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <p className="text-4xl font-bold tracking-tight text-foreground">
                        {inferenceResult.prediction}
                      </p>
                    </div>
                    <div className="pt-2 border-t border-border/50">
                      <p className="text-xs text-muted-foreground mb-1">
                        Confidence Score
                      </p>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-1000 ${
                              inferenceResult.confidence > 0.8
                                ? "bg-green-500"
                                : "bg-primary"
                            }`}
                            style={{
                              width: `${inferenceResult.confidence * 100}%`,
                            }}
                          ></div>
                        </div>
                        <span className="text-xl font-bold">
                          {(inferenceResult.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {inferenceResult.llm_explanation && (
                  <div className="bg-card rounded-2xl shadow-sm border border-border p-6 flex flex-col">
                    <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-4">
                      Clinical Insight
                    </h3>
                    <div className="prose prose-sm dark:prose-invert max-w-none flex-1">
                      <p className="text-sm leading-relaxed text-foreground/80 italic">
                        "{inferenceResult.llm_explanation}"
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* XAI Visualizations */}
              {(gradcamImage || limeImage) && (
                <div className="bg-card rounded-2xl shadow-sm border border-border p-6">
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                      <Activity className="h-5 w-5 text-primary" />
                      Explainability Modules
                    </h3>
                    <span className="text-[10px] px-2 py-0.5 bg-primary/10 text-primary font-bold rounded uppercase tracking-tighter">
                      Model Reasoning
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {gradcamImage && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-semibold">
                            Grad-CAM Heatmap
                          </span>
                          <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
                        </div>
                        <div className="rounded-xl overflow-hidden border border-border shadow-sm group cursor-zoom-in">
                          <img
                            src={gradcamImage}
                            alt="Grad-CAM"
                            className="w-full h-auto transition-transform group-hover:scale-105 duration-500"
                          />
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Visualizes areas within the image that had the
                          strongest gradient influence on the model's final
                          prediction.
                        </p>
                      </div>
                    )}

                    {limeImage && (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-semibold">
                            LIME Local Features
                          </span>
                          <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                        </div>
                        <div className="rounded-xl overflow-hidden border border-border shadow-sm group cursor-zoom-in">
                          <img
                            src={limeImage}
                            alt="LIME"
                            className="w-full h-auto transition-transform group-hover:scale-105 duration-500"
                          />
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          Highlights specific superpixels/features contributing
                          to the local classification via perturbation analysis.
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
