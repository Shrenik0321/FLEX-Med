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
      <h1 className="text-3xl font-semibold text-gray-900 mb-2">
        Public Dataset
      </h1>
      <p className="text-[#718096] mb-8">
        Upload and manage public datasets for federated learning training.
      </p>

      {/* Form */}
      <div className="max-w-2xl">
        <div className="bg-white rounded-lg p-8 flex-card-shadow space-y-6">
          {/* Dataset Info */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Dataset Information
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Dataset Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., Pneumonia MNIST"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className="w-full px-4 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Description
                </label>
                <textarea
                  placeholder="Brief description of the dataset..."
                  value={datasetDescription}
                  onChange={(e) => setDatasetDescription(e.target.value)}
                  className="w-full px-4 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-[#B80028] focus:border-transparent min-h-[100px]"
                />
              </div>
            </div>
          </div>

          {/* Data Upload */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <span className="flex-red-dot" />
              Upload Files
            </h2>
            <div
              className="border-2 border-dashed border-[#E2E8F0] rounded-lg p-8 text-center hover:border-[#B80028] hover:bg-[rgba(184,0,40,0.02)] transition-colors cursor-pointer"
              onClick={() => alert("File selection dialog would open here")}
            >
              <Upload className="mx-auto text-[#718096] mb-3" size={32} />
              <p className="font-medium text-gray-900">
                Drop files here or click to upload
              </p>
              <p className="text-xs text-[#718096] mt-1">
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
              className="px-6 py-2 border border-[#E2E8F0] text-gray-900 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="px-6 py-2 bg-white border border-[#B80028] text-[#B80028] rounded-lg font-medium hover:bg-[rgba(184,0,40,0.08)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Uploading..." : "Upload Dataset"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
