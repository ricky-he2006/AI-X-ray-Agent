import { useState } from 'react';
import XrayUploader from '@/components/XrayUploader';
import AnalysisResults from '@/components/AnalysisResults';
import { Activity } from 'lucide-react';

export interface Finding {
  name: string;
  confidence: number;
  severity: 'critical' | 'moderate' | 'mild' | 'none';
  region: { x: number; y: number; w: number; h: number };
  description: string;
}

export interface AnalysisResult {
  body_region: string;
  region_confidence: number;
  model_version: string;
  timestamp: string;
  findings: Finding[];
  differentials: string[];
  urgency: 'critical' | 'moderate' | 'low';
  auroc: number;
}

const Index = () => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary">
      {/* Header */}
      <header className="border-b border-border/40 bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Activity className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-foreground">
                AI X-Ray Analyzer
              </h1>
              <p className="text-muted-foreground">
                Advanced medical imaging analysis powered by AI
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          <XrayUploader
            onAnalysisComplete={(result, image) => {
              setAnalysisResult(result);
              setSelectedImage(image);
            }}
          />

          {analysisResult && selectedImage && (
            <div className="mt-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
              <AnalysisResults
                result={analysisResult}
                imageUrl={selectedImage}
              />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-20 border-t border-border/40 bg-card/30 backdrop-blur-sm py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>
            Powered by Lovable AI • For research and educational purposes only
          </p>
          <p className="mt-1 text-xs">
            This is not a substitute for professional medical advice
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
