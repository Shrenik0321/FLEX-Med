"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { mockClients } from "@/lib/mock-data";

interface FLConfigModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit?: (config: FLConfig) => void;
}

export interface FLConfig {
  num_rounds: number;
  lr: number;
  local_epochs: number;
  batch_size: number;
  distill_lr: number;
  distill_epochs: number;
  temperature: number;
  selected_client_ids: number[];
}

const presets = {
  fast_test: {
    num_rounds: 5,
    lr: 0.001,
    local_epochs: 3,
    batch_size: 32,
    distill_lr: 0.001,
    distill_epochs: 3,
    temperature: 3.0,
    description: "Quick validation run (~10 min)"
  },
  balanced: {
    num_rounds: 15,
    lr: 0.0001,
    local_epochs: 8,
    batch_size: 32,
    distill_lr: 0.001,
    distill_epochs: 3,
    temperature: 3.0,
    description: "Standard training (~45 min)"
  },
  high_accuracy: {
    num_rounds: 25,
    lr: 0.00005,
    local_epochs: 15,
    batch_size: 16,
    distill_lr: 0.0005,
    distill_epochs: 5,
    temperature: 3.0,
    description: "Maximum performance (~2 hours)"
  }
};

export function FLConfigModal({ open, onOpenChange, onSubmit }: FLConfigModalProps) {
  const [config, setConfig] = useState<FLConfig>({
    num_rounds: 15,
    lr: 0.0001,
    local_epochs: 8,
    batch_size: 32,
    distill_lr: 0.001,
    distill_epochs: 3,
    temperature: 3.0,
    selected_client_ids: [0, 1, 2]
  });

  const handlePresetSelect = (preset: keyof typeof presets) => {
    const presetConfig = presets[preset];
    setConfig({
      ...config,
      num_rounds: presetConfig.num_rounds,
      lr: presetConfig.lr,
      local_epochs: presetConfig.local_epochs,
      batch_size: presetConfig.batch_size,
      distill_lr: presetConfig.distill_lr,
      distill_epochs: presetConfig.distill_epochs,
      temperature: presetConfig.temperature
    });
  };

  const handleClientToggle = (clientId: number) => {
    setConfig(prev => ({
      ...prev,
      selected_client_ids: prev.selected_client_ids.includes(clientId)
        ? prev.selected_client_ids.filter(id => id !== clientId)
        : [...prev.selected_client_ids, clientId]
    }));
  };

  const estimatedDuration = () => {
    const baseTime = config.num_rounds * config.local_epochs * config.selected_client_ids.length * 0.5;
    return Math.round(baseTime);
  };

  const handleSubmit = () => {
    onSubmit?.(config);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Configure Federated Learning Run</DialogTitle>
          <DialogDescription>
            Set training parameters and select participating clients
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Quick Presets */}
          <div>
            <Label className="text-sm font-medium mb-2 block">Quick Presets</Label>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePresetSelect("fast_test")}
              >
                Fast Test
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePresetSelect("balanced")}
              >
                Balanced
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePresetSelect("high_accuracy")}
              >
                High Accuracy
              </Button>
            </div>
          </div>

          {/* Training Parameters */}
          <div className="space-y-4 border rounded-lg p-4">
            <h3 className="font-medium">Training Parameters</h3>

            <div className="space-y-2">
              <Label htmlFor="num_rounds">
                Number of Rounds: {config.num_rounds}
              </Label>
              <Slider
                id="num_rounds"
                min={5}
                max={30}
                step={1}
                value={[config.num_rounds]}
                onValueChange={([value]) => setConfig({ ...config, num_rounds: value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="lr">Learning Rate</Label>
              <Select
                value={config.lr.toString()}
                onValueChange={(value) => setConfig({ ...config, lr: parseFloat(value) })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0.001">0.001</SelectItem>
                  <SelectItem value="0.0001">0.0001</SelectItem>
                  <SelectItem value="0.00005">0.00005</SelectItem>
                  <SelectItem value="0.00001">0.00001</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="local_epochs">
                Local Epochs: {config.local_epochs}
              </Label>
              <Slider
                id="local_epochs"
                min={3}
                max={20}
                step={1}
                value={[config.local_epochs]}
                onValueChange={([value]) => setConfig({ ...config, local_epochs: value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="batch_size">Batch Size</Label>
              <Select
                value={config.batch_size.toString()}
                onValueChange={(value) => setConfig({ ...config, batch_size: parseInt(value) })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="16">16</SelectItem>
                  <SelectItem value="32">32</SelectItem>
                  <SelectItem value="64">64</SelectItem>
                  <SelectItem value="128">128</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Distillation Parameters */}
          <div className="space-y-4 border rounded-lg p-4">
            <h3 className="font-medium">Distillation Parameters</h3>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="distill_lr">Distillation LR</Label>
                <Input
                  id="distill_lr"
                  type="number"
                  step="0.0001"
                  value={config.distill_lr}
                  onChange={(e) => setConfig({ ...config, distill_lr: parseFloat(e.target.value) })}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="distill_epochs">Distillation Epochs</Label>
                <Input
                  id="distill_epochs"
                  type="number"
                  value={config.distill_epochs}
                  onChange={(e) => setConfig({ ...config, distill_epochs: parseInt(e.target.value) })}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="temperature">Temperature</Label>
              <Input
                id="temperature"
                type="number"
                step="0.1"
                value={config.temperature}
                onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
              />
            </div>
          </div>

          {/* Client Selection */}
          <div className="space-y-4 border rounded-lg p-4">
            <h3 className="font-medium">Client Selection</h3>
            <div className="space-y-2">
              {mockClients.map((client) => (
                <div key={client.id} className="flex items-center space-x-3 p-2 rounded hover:bg-muted">
                  <Checkbox
                    id={`client-${client.id}`}
                    checked={config.selected_client_ids.includes(client.id)}
                    onCheckedChange={() => handleClientToggle(client.id)}
                  />
                  <Label
                    htmlFor={`client-${client.id}`}
                    className="flex-1 cursor-pointer flex items-center justify-between"
                  >
                    <span className="flex items-center gap-2">
                      {client.client_name}
                      <Badge variant="outline" className="text-xs">
                        {client.model_type}
                      </Badge>
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {client.has_local_data
                        ? `${client.dataset_size} samples`
                        : "FREE RIDER"}
                    </span>
                  </Label>
                </div>
              ))}
            </div>
          </div>

          {/* Estimated Duration */}
          <div className="bg-muted p-4 rounded-lg">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Estimated Duration:</span>
              <span className="text-lg font-semibold">~{estimatedDuration()} minutes</span>
            </div>
            {config.selected_client_ids.some(id => !mockClients.find(c => c.id === id)?.has_local_data) && (
              <p className="text-xs text-muted-foreground mt-2">
                Warning: Selected clients include free riders without local data
              </p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit}>
            Start Training
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
