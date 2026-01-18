"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { mockXAIAnalysis } from "@/lib/mock-data";
import { Brain, Target, Sparkles, AlertCircle, CheckCircle2, Info } from "lucide-react";

export function EnhancedXAIDisplay() {
  const { prediction, confidence, gradcam, lime, llm_explanation } = mockXAIAnalysis;

  return (
    <div className="space-y-6">
      {/* Prediction Header */}
      <Card className="border-2">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <CardTitle className="text-2xl flex items-center gap-2">
                <Brain className="h-6 w-6" />
                Diagnosis Result
              </CardTitle>
              <CardDescription>AI-powered medical image analysis with explainability</CardDescription>
            </div>
            <Badge
              variant={confidence > 0.8 ? "default" : "secondary"}
              className="text-lg px-4 py-2"
            >
              {(confidence * 100).toFixed(1)}% Confidence
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="text-3xl font-bold text-red-600 mb-2">{prediction}</div>
              <Progress value={confidence * 100} className="h-3" />
            </div>

            <Alert className="bg-amber-50 border-amber-200">
              <AlertCircle className="h-4 w-4 text-amber-600" />
              <AlertDescription className="text-amber-800">
                This is an AI-assisted diagnosis. All predictions should be reviewed by a qualified
                medical professional before clinical decision-making.
              </AlertDescription>
            </Alert>
          </div>
        </CardContent>
      </Card>

      {/* Explainability Tabs */}
      <Tabs defaultValue="visual" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="visual">Visual Explanations</TabsTrigger>
          <TabsTrigger value="insights">Key Insights</TabsTrigger>
          <TabsTrigger value="medical">Medical Explanation</TabsTrigger>
        </TabsList>

        {/* Visual Explanations Tab */}
        <TabsContent value="visual" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {/* Grad-CAM */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Grad-CAM Heatmap
                </CardTitle>
                <CardDescription>
                  Regions of highest model attention during classification
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <Target className="h-16 w-16 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">Grad-CAM Heatmap</p>
                    <p className="text-xs">Upload image to generate</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Attention Analysis</h4>
                  {gradcam.analysis.top_regions.map((region, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-muted rounded">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs capitalize">
                          {region.location.replace('_', ' ')}
                        </Badge>
                        <span className="text-sm text-muted-foreground">{region.region_size} of image</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Progress value={region.intensity * 100} className="w-20 h-2" />
                        <span className="text-sm font-medium">{(region.intensity * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-2 text-blue-800 text-sm">
                    <Info className="h-4 w-4" />
                    <span className="font-medium">
                      Focus: {gradcam.analysis.focus_concentration} concentration in {gradcam.analysis.primary_quadrant.replace('_', ' ')}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* LIME */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  LIME Segmentation
                </CardTitle>
                <CardDescription>
                  Super-pixel regions contributing to the prediction
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <Sparkles className="h-16 w-16 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">LIME Explanation</p>
                    <p className="text-xs">Upload image to generate</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium">Interpretation Confidence</h4>
                    <span className="text-sm font-medium">{(lime.analysis.interpretation_confidence * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={lime.analysis.interpretation_confidence * 100} className="h-2" />
                </div>

                <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="text-xs text-green-700 mb-1">
                    <CheckCircle2 className="h-3 w-3 inline mr-1" />
                    Positive Contributors
                  </div>
                  <div className="text-sm font-medium text-green-800">
                    {lime.analysis.positive_features.length} features identified
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Key Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Key Influencing Factors</CardTitle>
              <CardDescription>
                Features identified by LIME as most important for this prediction
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="text-sm font-medium mb-3 text-green-700">Positive Contributors (Support Diagnosis)</h4>
                <div className="space-y-2">
                  {lime.analysis.positive_features.map((feature, idx) => (
                    <div key={idx} className="p-3 border rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="default" className="capitalize">
                          {feature.description}
                        </Badge>
                        <span className="text-sm font-bold text-green-600">
                          {(feature.weight * 100).toFixed(0)}% importance
                        </span>
                      </div>
                      <Progress value={feature.weight * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-3 text-gray-600">Negative Contributors (Against Diagnosis)</h4>
                <div className="space-y-2">
                  {lime.analysis.negative_features.map((feature, idx) => (
                    <div key={idx} className="p-3 border rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex items-center justify-between mb-2">
                        <Badge variant="outline" className="capitalize">
                          {feature.description}
                        </Badge>
                        <span className="text-sm font-medium text-gray-600">
                          {Math.abs(feature.weight * 100).toFixed(0)}% against
                        </span>
                      </div>
                      <Progress value={Math.abs(feature.weight) * 100} className="h-2 [&>div]:bg-gray-400" />
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Attention Heatmap Summary</CardTitle>
              <CardDescription>Model focus areas from Grad-CAM analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3">
                <div className="p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Primary Focus Region</div>
                  <div className="text-lg font-bold capitalize">
                    {gradcam.analysis.primary_quadrant.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    {gradcam.analysis.focus_concentration} concentration pattern
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Peak Attention Intensity</div>
                  <div className="text-lg font-bold">
                    {(gradcam.analysis.top_regions[0].intensity * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Model's strongest activation signal
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Coverage Area</div>
                  <div className="text-lg font-bold">
                    {gradcam.analysis.top_regions.reduce((sum, r) => sum + parseInt(r.region_size), 0)}%
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Total image area analyzed
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Medical Explanation Tab */}
        <TabsContent value="medical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI-Generated Medical Explanation
              </CardTitle>
              <CardDescription>
                Natural language interpretation powered by large language models
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none">
                <div className="whitespace-pre-line text-sm leading-relaxed">
                  {llm_explanation}
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-start gap-3">
                  <Info className="h-5 w-5 text-blue-600 mt-0.5" />
                  <div className="text-sm text-blue-800">
                    <p className="font-medium mb-1">How this explanation was generated:</p>
                    <ul className="space-y-1 text-xs">
                      <li>• Model prediction analyzed with Grad-CAM and LIME</li>
                      <li>• Structural features extracted from attention heatmaps</li>
                      <li>• Medical domain knowledge integrated via LLM</li>
                      <li>• Explanation grounded in quantitative xAI outputs</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Clinical Recommendations</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Alert className="bg-amber-50 border-amber-200">
                <AlertCircle className="h-4 w-4 text-amber-600" />
                <AlertDescription className="text-amber-800 text-sm">
                  <strong>Next Steps:</strong> This AI-assisted diagnosis should be confirmed through:
                  <ul className="mt-2 ml-4 space-y-1 text-xs">
                    <li>• Manual microscopy review by a pathologist</li>
                    <li>• Complete blood count (CBC) analysis</li>
                    <li>• Bone marrow biopsy if clinically indicated</li>
                    <li>• Flow cytometry for definitive diagnosis</li>
                  </ul>
                </AlertDescription>
              </Alert>

              <div className="grid gap-2 md:grid-cols-2">
                <div className="p-3 border rounded-lg">
                  <div className="text-xs text-muted-foreground mb-1">Confidence Level</div>
                  <div className="text-sm font-medium">
                    {confidence > 0.9 ? "Very High" : confidence > 0.8 ? "High" : "Moderate"}
                  </div>
                </div>
                <div className="p-3 border rounded-lg">
                  <div className="text-xs text-muted-foreground mb-1">Review Priority</div>
                  <div className="text-sm font-medium text-red-600">
                    Urgent - Requires Specialist Review
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Technical Details */}
      <Card className="border-dashed">
        <CardHeader>
          <CardTitle className="text-sm">Technical Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Grad-CAM Target Layer:</span>
              <span className="font-mono">model.layer4[-1]</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">LIME Samples:</span>
              <span className="font-mono">1000 perturbations</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">LIME Features:</span>
              <span className="font-mono">{lime.analysis.feature_count} super-pixels</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">LLM Model:</span>
              <span className="font-mono">Claude 3.5 Sonnet</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
