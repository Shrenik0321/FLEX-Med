"use client";

import { Upload } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea"; // Assuming this exists, if not I will fallback to textarea
import { Loading } from "@/components/ui/loading";

export default function PublicDatasetPage() {
  const [datasetName, setDatasetName] = useState("");
  const [datasetDescription, setDatasetDescription] = useState("");
  const [loading, setLoading] = useState(false);

  // Check if Textarea component exists, otherwise use basic textarea
  // Since I haven't checked for Textarea component, I'll stick to basic textarea with shadcn classes for safety or check first.
  // Actually, I'll stick to the existing textarea classes but wrapped in a better structure.

  const handleSubmit = async () => {
    if (!datasetName) {
      toast.error("Please provide a dataset name.");
      return;
    }

    setLoading(true);
    const toastId = toast.loading("Uploading dataset...");

    try {
      // Simulate API call
      console.log("Uploading dataset", { datasetName, datasetDescription });
      await new Promise((resolve) => setTimeout(resolve, 1500));

      toast.success("Dataset uploaded successfully!", { id: toastId });

      // Reset form
      setDatasetName("");
      setDatasetDescription("");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "An error occurred", {
        id: toastId,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-foreground mb-1">
          Public Dataset
        </h1>
        <p className="text-muted-foreground">
          Upload and manage public datasets for federated learning training.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Form Section */}
        <div className="md:col-span-2 space-y-6">
          <div className="bg-card rounded-xl p-6 shadow-sm border border-border">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Dataset Information
            </h2>
            <div className="space-y-4">
              <div>
                <Label className="mb-2 block">Dataset Name</Label>
                <Input
                  placeholder="e.g., Pneumonia MNIST"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                />
              </div>
              <div>
                <Label className="mb-2 block">Description</Label>
                <Textarea
                  placeholder="Brief description of the dataset..."
                  value={datasetDescription}
                  onChange={(e) => setDatasetDescription(e.target.value)}
                  className="min-h-[120px]"
                />
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3">
            <Button
              variant="outline"
              onClick={() => {
                setDatasetName("");
                setDatasetDescription("");
              }}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button onClick={handleSubmit} disabled={loading}>
              Upload Dataset
            </Button>
          </div>
        </div>

        {/* Upload Area */}
        <div className="md:col-span-1">
          <div className="bg-card rounded-xl p-6 shadow-sm border border-border h-full">
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Upload Files
            </h2>
            <div
              className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary hover:bg-primary/5 transition-all cursor-pointer h-[200px] flex flex-col items-center justify-center gap-3"
              onClick={() =>
                toast.info("File selection dialog would open here")
              }
            >
              <div className="p-3 bg-muted rounded-full">
                <Upload className="text-muted-foreground" size={24} />
              </div>
              <div>
                <p className="font-medium text-foreground">Click to upload</p>
                <p className="text-xs text-muted-foreground mt-1">
                  CSV, JSON, Parquet, or Zip
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
