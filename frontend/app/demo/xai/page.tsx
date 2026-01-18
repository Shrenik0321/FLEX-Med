import { EnhancedXAIDisplay } from "@/components/enhanced-xai-display";

export default function XAIDemoPage() {
  return (
    <div className="container mx-auto py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Explainable AI Demo</h1>
        <p className="text-muted-foreground">
          Enhanced xAI with Grad-CAM, LIME, and LLM-generated medical explanations
        </p>
      </div>
      <EnhancedXAIDisplay />
    </div>
  );
}
