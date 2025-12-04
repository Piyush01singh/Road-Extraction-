
export enum StepStatus {
  PENDING = 'PENDING',
  ACTIVE = 'ACTIVE',
  COMPLETED = 'COMPLETED',
}

export type ModelType = 'Logistic Regression' | 'Decision Tree' | 'XGBoost' | 'SVM' | 'AutoML';

export interface StepConfig {
  id: number;
  title: string;
  description: string;
  pythonSnippet: string;
  pythonFile?: string; // Path to external Python file
}

export interface PixelData {
  id: number;
  r: number;
  g: number;
  b: number;
  texture: number; // Variance or entropy
  ndvi: number; // Normalized Difference Vegetation Index simulation
  isRoad: number; // 0 or 1
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc: number;
  modelName?: string;
}

export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface TrainingLog {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export interface ChatMessage {
  role: 'user' | 'ai';
  text: string;
}

// Data Science Pipeline Context
export interface PipelineContext {
  rawData: PixelData[];
  processedData: PixelData[];
  trainData: PixelData[];
  testData: PixelData[];
  predictions: { id: number; actual: number; predicted: number; prob: number }[];
  metrics: ModelMetrics | null;
  comparisonResults: ModelMetrics[]; // For AutoML
  selectedModel: ModelType;
  confusionMatrix: ConfusionMatrix | null;
  rocCurve: RocPoint[];
  trainingHistory: TrainingLog[];
}
