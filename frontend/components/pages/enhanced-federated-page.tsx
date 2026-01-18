"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { FLConfigModal } from "@/components/fl-config-modal";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";
import {
  mockClients,
  mockFLRuns,
  mockRoundMetrics,
  mockClassBalance,
  mockArchitecturePerformance,
  mockContributionHeatmap
} from "@/lib/mock-data";
import { Activity, TrendingUp, Users, Play, Settings, Award, Target } from "lucide-react";

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

export function EnhancedFederatedPage() {
  const [configModalOpen, setConfigModalOpen] = useState(false);

  const totalClients = mockClients.length;
  const completedRuns = mockFLRuns.filter(r => r.status === "completed").length;
  const avgImprovement = mockClients
    .filter(c => c.has_local_data)
    .reduce((sum, c) => sum + c.improvement, 0) / mockClients.filter(c => c.has_local_data).length;

  const contributionData = mockClients
    .filter(c => c.has_local_data)
    .map(c => ({
      name: c.client_name,
      value: c.contribution_weight
    }));

  // Client performance comparison data
  const clientComparisonData = mockClients.map(c => ({
    name: c.client_name,
    "Pre-FL": c.pre_fl_accuracy,
    "Post-FL": c.post_fl_accuracy
  }));

  // Multi-metric radar chart data
  const radarData = mockClients.slice(0, 3).map(c => ({
    metric: c.client_name,
    Accuracy: c.post_fl_accuracy,
    F1: c.post_fl_accuracy - 2, // Mock F1 data
    Precision: c.post_fl_accuracy - 1,
    Recall: c.post_fl_accuracy - 3,
    "ROC-AUC": c.post_fl_accuracy + 1
  }));

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Federated Learning Dashboard</h1>
          <p className="text-muted-foreground">
            Collaborative medical AI training with privacy preservation
          </p>
        </div>
        <Button onClick={() => setConfigModalOpen(true)} size="lg">
          <Play className="mr-2 h-4 w-4" />
          Start New FL Run
        </Button>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="clients">Client Performance</TabsTrigger>
          <TabsTrigger value="training">Training Dynamics</TabsTrigger>
          <TabsTrigger value="fairness">Fairness & Heterogeneity</TabsTrigger>
        </TabsList>

        {/* OVERVIEW TAB */}
        <TabsContent value="overview" className="space-y-6">
          {/* Hero Metrics */}
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Clients Enrolled</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{totalClients} Hospitals</div>
                <p className="text-xs text-muted-foreground">
                  {mockClients.filter(c => c.has_local_data).length} with local data
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Completed FL Runs</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{completedRuns} Successful Runs</div>
                <p className="text-xs text-muted-foreground">
                  Latest: {new Date(mockFLRuns[mockFLRuns.length - 1].started_at).toLocaleDateString()}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Average Improvement</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">+{avgImprovement.toFixed(1)}%</div>
                <p className="text-xs text-muted-foreground">Accuracy gain across clients</p>
              </CardContent>
            </Card>
          </div>

          {/* Recent Runs Table */}
          <Card>
            <CardHeader>
              <CardTitle>Recent FL Runs</CardTitle>
              <CardDescription>Historical training sessions with performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {mockFLRuns.slice().reverse().map((run) => (
                  <div
                    key={run.run_id}
                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted cursor-pointer transition-colors"
                  >
                    <div className="flex items-center gap-4 flex-1">
                      <div className="flex flex-col">
                        <span className="font-medium">{run.run_name}</span>
                        <span className="text-sm text-muted-foreground">
                          {new Date(run.started_at).toLocaleDateString()} - {run.num_clients} clients, {run.num_rounds} rounds
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <div className="text-sm font-medium">+{run.avg_improvement_accuracy.toFixed(1)}%</div>
                        <div className="text-xs text-muted-foreground">Avg Improvement</div>
                      </div>

                      <Badge variant={run.status === "completed" ? "default" : "secondary"}>
                        {run.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Best Performing Run</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Run Name:</span>
                    <span className="font-medium">High Accuracy Run</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Improvement:</span>
                    <span className="font-medium text-green-600">+21.7%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Rounds:</span>
                    <span className="font-medium">25</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">F1 Gain:</span>
                    <span className="font-medium">+20.5%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Overview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Active Models:</span>
                  <span className="font-medium">{new Set(mockClients.map(c => c.model_type)).size} Architectures</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Total Dataset:</span>
                  <span className="font-medium">{mockClients.reduce((sum, c) => sum + c.dataset_size, 0).toLocaleString()} Images</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Free Riders:</span>
                  <span className="font-medium">{mockClients.filter(c => !c.has_local_data).length} Clients</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Privacy Level:</span>
                  <Badge variant="default">100% Data Privacy</Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* CLIENT PERFORMANCE TAB */}
        <TabsContent value="clients" className="space-y-6">
          {/* Per-Client Performance Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {mockClients.map((client) => (
              <Card key={client.id} className="relative overflow-hidden">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{client.client_name}</CardTitle>
                    <Badge variant="outline">{client.model_type}</Badge>
                  </div>
                  <CardDescription>
                    {client.has_local_data
                      ? `${client.dataset_path} (${client.dataset_size.toLocaleString()} images)`
                      : "Free Rider - No Local Data"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Pre-FL:</span>
                      <span className="font-medium">{client.pre_fl_accuracy.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Post-FL:</span>
                      <span className="font-medium text-green-600">{client.post_fl_accuracy.toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Improvement:</span>
                      <span className="font-bold text-green-600">+{client.improvement.toFixed(1)}%</span>
                    </div>
                    <Progress value={(client.improvement / 25) * 100} className="h-2" />
                  </div>

                  {client.has_local_data && (
                    <>
                      <div className="pt-2 border-t space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Contribution Weight:</span>
                          <span className="font-medium">{client.contribution_weight}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Quality Multiplier:</span>
                          <span className="font-medium">{client.quality_multiplier.toFixed(2)}x</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Convergence:</span>
                          <Badge variant="default" className="text-xs">
                            {client.convergence_status}
                          </Badge>
                        </div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Comparison Charts */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Pre-FL vs Post-FL Accuracy</CardTitle>
                <CardDescription>Performance improvement across all clients</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={clientComparisonData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Pre-FL" fill="#94a3b8" />
                    <Bar dataKey="Post-FL" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Client Contribution Distribution</CardTitle>
                <CardDescription>Normalized consensus weights</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={contributionData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {contributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Multi-Metric Radar Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Multi-Metric Performance Comparison</CardTitle>
              <CardDescription>Comprehensive evaluation across multiple metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar name="Hospital_A" dataKey="Accuracy" stroke="#0088FE" fill="#0088FE" fillOpacity={0.6} />
                  <Tooltip />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* TRAINING DYNAMICS TAB */}
        <TabsContent value="training" className="space-y-6">
          {/* Accuracy Over Rounds */}
          <Card>
            <CardHeader>
              <CardTitle>Accuracy Evolution Over Training Rounds</CardTitle>
              <CardDescription>Tracking model performance improvement through federated rounds</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={mockRoundMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                  <YAxis domain={[60, 95]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="overall_accuracy" stroke="#8884d8" strokeWidth={2} name="Overall Avg" />
                  <Line type="monotone" dataKey="best_client_accuracy" stroke="#10b981" strokeWidth={2} name="Best Client" />
                  <Line type="monotone" dataKey="worst_client_accuracy" stroke="#ef4444" strokeWidth={2} name="Worst Client" />
                  <Line type="monotone" dataKey="consensus_quality" stroke="#f59e0b" strokeWidth={2} name="Consensus Quality" yAxisId={0} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Loss Charts */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Training & Distillation Loss</CardTitle>
                <CardDescription>Convergence patterns over rounds</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={mockRoundMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="training_loss" stroke="#f97316" strokeWidth={2} name="Training Loss" />
                    <Line type="monotone" dataKey="distillation_loss" stroke="#3b82f6" strokeWidth={2} name="Distillation Loss" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Class Balance Evolution</CardTitle>
                <CardDescription>Leukemia vs Healthy accuracy</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={mockClassBalance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="round" />
                    <YAxis domain={[60, 95]} />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="leukemia_accuracy" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} name="Leukemia" />
                    <Area type="monotone" dataKey="healthy_accuracy" stackId="2" stroke="#10b981" fill="#10b981" fillOpacity={0.6} name="Healthy" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Round-by-Round Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Training Summary</CardTitle>
              <CardDescription>Key metrics from the latest FL run</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">15</div>
                  <div className="text-sm text-muted-foreground">Total Rounds</div>
                </div>
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-green-600">84.2%</div>
                  <div className="text-sm text-muted-foreground">Final Accuracy</div>
                </div>
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">0.18</div>
                  <div className="text-sm text-muted-foreground">Final Loss</div>
                </div>
                <div className="text-center p-4 border rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">0.89</div>
                  <div className="text-sm text-muted-foreground">Consensus Quality</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* FAIRNESS & HETEROGENEITY TAB */}
        <TabsContent value="fairness" className="space-y-6">
          {/* Architecture Performance */}
          <Card>
            <CardHeader>
              <CardTitle>Architecture Performance Comparison</CardTitle>
              <CardDescription>Model heterogeneity analysis across different architectures</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {mockArchitecturePerformance.map((arch, idx) => (
                  <div key={idx} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4 flex-1">
                      <Badge variant="outline">{arch.model}</Badge>
                      <span className="text-sm text-muted-foreground">{arch.clients} client(s)</span>
                    </div>
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <div className="text-sm font-medium">{arch.avg_accuracy.toFixed(1)}%</div>
                        <div className="text-xs text-muted-foreground">Avg Accuracy</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-green-600">+{arch.improvement.toFixed(1)}%</div>
                        <div className="text-xs text-muted-foreground">Improvement</div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium">{arch.suitability_bonus.toFixed(2)}x</div>
                        <div className="text-xs text-muted-foreground">Suitability</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Free Rider Analysis */}
          <Card>
            <CardHeader>
              <CardTitle>Free Rider Analysis</CardTitle>
              <CardDescription>Learning without local data contribution</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockClients.filter(c => !c.has_local_data).map((client) => (
                  <div key={client.id} className="p-4 border rounded-lg bg-muted/50">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="font-medium">{client.client_name}</h4>
                        <p className="text-sm text-muted-foreground">No local training data</p>
                      </div>
                      <Badge variant="secondary">FREE RIDER</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <div className="text-2xl font-bold">{client.post_fl_accuracy.toFixed(1)}%</div>
                        <div className="text-xs text-muted-foreground">Achieved Accuracy</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-green-600">+{client.improvement.toFixed(1)}%</div>
                        <div className="text-xs text-muted-foreground">Improvement</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold">100%</div>
                        <div className="text-xs text-muted-foreground">From Distillation</div>
                      </div>
                    </div>
                    <p className="text-sm mt-4 text-muted-foreground">
                      This client learned purely through knowledge distillation from other participants,
                      achieving {client.post_fl_accuracy.toFixed(1)}% accuracy without contributing any training data.
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Fairness Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Fairness Metrics</CardTitle>
              <CardDescription>Distribution and equality analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="p-4 border rounded-lg text-center">
                  <div className="text-sm text-muted-foreground mb-2">Gini Coefficient</div>
                  <div className="text-3xl font-bold">0.12</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Low inequality in weight distribution
                  </p>
                </div>
                <div className="p-4 border rounded-lg text-center">
                  <div className="text-sm text-muted-foreground mb-2">Weight Std Dev</div>
                  <div className="text-3xl font-bold">0.05</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Stable contribution weights
                  </p>
                </div>
                <div className="p-4 border rounded-lg text-center">
                  <div className="text-sm text-muted-foreground mb-2">Max Dominance</div>
                  <div className="text-3xl font-bold">45%</div>
                  <p className="text-xs text-muted-foreground mt-2">
                    No single client dominates
                  </p>
                </div>
              </div>

              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center gap-2 text-green-800">
                  <Award className="h-5 w-5" />
                  <span className="font-medium">Fairness Status: Excellent</span>
                </div>
                <p className="text-sm text-green-700 mt-2">
                  All fairness metrics indicate balanced participation and equitable contribution across clients.
                  No single participant has excessive influence on the global model.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Comparison: FL vs Centralized */}
          <Card>
            <CardHeader>
              <CardTitle>Privacy-Utility Tradeoff</CardTitle>
              <CardDescription>FL performance vs centralized baseline</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="p-6 border rounded-lg">
                  <div className="text-sm text-muted-foreground mb-2">Centralized Training</div>
                  <div className="text-3xl font-bold mb-4">88.5%</div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Privacy:</span>
                      <Badge variant="destructive">0%</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Data Sharing:</span>
                      <span className="text-red-600">Required</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Compliance:</span>
                      <Badge variant="destructive">Violation Risk</Badge>
                    </div>
                  </div>
                </div>

                <div className="p-6 border rounded-lg bg-green-50">
                  <div className="text-sm text-muted-foreground mb-2">Federated Learning</div>
                  <div className="text-3xl font-bold mb-4 text-green-600">84.2%</div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Privacy:</span>
                      <Badge variant="default">100%</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Data Sharing:</span>
                      <span className="text-green-600">Never</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Compliance:</span>
                      <Badge variant="default">Full HIPAA</Badge>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center gap-2 text-blue-800">
                  <Target className="h-5 w-5" />
                  <span className="font-medium">Result: 95.1% of Centralized Accuracy</span>
                </div>
                <p className="text-sm text-blue-700 mt-2">
                  FLEX-Med achieved 95.1% of centralized accuracy while maintaining 100% data privacy.
                  This demonstrates excellent privacy-utility tradeoff for medical AI applications.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <FLConfigModal
        open={configModalOpen}
        onOpenChange={setConfigModalOpen}
        onSubmit={(config) => {
          console.log("FL Config submitted:", config);
          // In real implementation, this would trigger the FL run
        }}
      />
    </div>
  );
}
