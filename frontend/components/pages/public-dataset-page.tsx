"use client";

import { Upload } from "lucide-react";
import { useState } from "react";

export default function PublicDatasetPage() {
  const [datasetName, setDatasetName] = useState("");
  const [datasetDescription, setDatasetDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!datasetName) {
      setError("Please provide a dataset name.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Simulate API call for now or implement if backend endpoint exists
      console.log("Uploading dataset", { datasetName, datasetDescription });
      await new Promise((resolve) => setTimeout(resolve, 1000));
      alert("Dataset uploaded request simulated.");
      // Reset form
      setDatasetName("");
      setDatasetDescription("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-semibold text-foreground mb-2">
        Public Dataset
      </h1>
      <p className="text-muted-foreground mb-8">
        Upload and manage public datasets for federated learning training.
      </p>

      {/* Form */}
      <div className="max-w-2xl">
        <div className="bg-card rounded-lg p-8 shadow-sm border border-border space-y-6">
          {/* Dataset Info */}
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Dataset Information
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Dataset Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., Pneumonia MNIST"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent bg-card"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Description
                </label>
                <textarea
                  placeholder="Brief description of the dataset..."
                  value={datasetDescription}
                  onChange={(e) => setDatasetDescription(e.target.value)}
                  className="w-full px-4 py-2 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent min-h-[100px] bg-card"
                />
              </div>
            </div>
          </div>

          {/* Data Upload */}
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Upload Files
            </h2>
            <div
              className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary hover:bg-primary/5 transition-colors cursor-pointer"
              onClick={() => alert("File selection dialog would open here")}
            >
              <Upload className="mx-auto text-muted-foreground mb-3" size={32} />
              <p className="font-medium text-foreground">
                Drop files here or click to upload
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Supported formats: CSV, JSON, Parquet, or Zip archives
              </p>
            </div>
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
              onClick={() => {
                setDatasetName("");
                setDatasetDescription("");
              }}
              className="px-6 py-2 border border-border text-foreground rounded-lg font-medium hover:bg-muted transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="px-6 py-2 bg-card border border-primary text-primary rounded-lg font-medium hover:bg-primary/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Uploading..." : "Upload Dataset"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
