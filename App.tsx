
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { 
  CheckCircle, BarChart2, Database, Layers, 
  Settings, Activity, PieChart, FileText, Cpu, Eye,
  Code, Terminal, AlertCircle, Zap, UploadCloud, Sun, Moon,
  Trees, Mountain, Building2, Download, Brain, GitBranch, Binary, ArrowRight, ArrowLeft,
  MessageSquare, Send, X, Copy, Check
} from 'lucide-react';
import { generateMockData, calculateMetrics, generateROC, generateTrainingHistory, simulateModelPerformance, generatePredictions } from './services/mockDataService';
import { generateAIInsight, generateChatResponse } from './services/geminiService';
import { FeatureDistChart, RocCurveChart, ConfusionMatrixHeatmap, TrainingLossChart } from './components/Charts';
import { PipelineContext, StepConfig, TrainingLog, ModelType, ModelMetrics, ChatMessage } from './types';

// Robust Python Code Snippets
const STEPS: StepConfig[] = [
  { 
    id: 1, 
    title: 'Data Loading', 
    description: 'Import satellite imagery dataset from CSV source.',
    pythonSnippet: `import pandas as pd
import numpy as np
import os

# Configuration
DATA_PATH = 'satellite_data.csv'

# Load satellite data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset Loaded Successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
else:
    print(f"Error: {DATA_PATH} not found.")`
  },
  { 
    id: 2, 
    title: 'Data Inspection', 
    description: 'Analyze dataset structure and types.',
    pythonSnippet: `# Display DataFrame information
print("--- DataFrame Info ---")
print(df.info())

# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\\nMissing Values Found:\\n{missing_counts[missing_counts > 0]}")
else:
    print("\\n✓ No missing values detected.")

# Statistical summary of pixel intensities
print("\\n--- Descriptive Statistics ---")
print(df[['r', 'g', 'b', 'ndvi']].describe().T)`
  },
  { 
    id: 3, 
    title: 'Data Processing', 
    description: 'Preprocess: Imputation, Outlier Removal, Scaling.',
    pythonSnippet: `from sklearn.preprocessing import MinMaxScaler

# 1. Handle outliers (Z-score method)
from scipy import stats
z_scores = stats.zscore(df[['r', 'g', 'b']])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

# 2. Feature Scaling
# Normalize pixel values to [0, 1] range for stability
scaler = MinMaxScaler()
cols_to_scale = ['r', 'g', 'b', 'texture']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(f"✓ Preprocessing Complete. {df.shape[0]} samples remain.")`
  },
  { 
    id: 4, 
    title: 'EDA', 
    description: 'Exploratory Data Analysis & Visualization.',
    pythonSnippet: `import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot 1: Feature Distribution (Green Channel)
plt.subplot(1, 2, 1)
sns.kdeplot(data=df, x='g', hue='isRoad', fill=True, palette='viridis')
plt.title('Green Channel Intensity: Road vs Non-Road')

# Plot 2: Class Balance
plt.subplot(1, 2, 2)
sns.countplot(x='isRoad', data=df, palette='coolwarm')
plt.title('Target Class Distribution')

plt.tight_layout()
plt.show()`
  },
  { 
    id: 5, 
    title: 'Feature Selection', 
    description: 'Define Features (X) and Target (Y).',
    pythonSnippet: `from sklearn.feature_selection import SelectKBest, f_classif

# Define Feature Matrix (X) and Target Vector (y)
X = df.drop(['id', 'isRoad'], axis=1)
y = df['isRoad']

# Select top 5 most discriminative features
selector = SelectKBest(score_func=f_classif, k='all')
X_new = selector.fit_transform(X, y)

# Get feature scores
scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
print(scores.sort_values(by='Score', ascending=False))`
  },
  { 
    id: 6, 
    title: 'Train/Test Split', 
    description: 'Partition data for validation.',
    pythonSnippet: `from sklearn.model_selection import train_test_split

# Split data: 80% Training, 20% Testing
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Training Set: {X_train.shape}")
print(f"Testing Set:  {X_test.shape}")`
  },
  { 
    id: 7, 
    title: 'Model Training', 
    description: 'Select and train machine learning algorithms.',
    pythonSnippet: `from xgboost import XGBClassifier

# Initialize and Train Model
# Using default hyperparameters for baseline
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("Training model...")
model.fit(X_train, y_train)
print("✓ Training complete.")`
  },
  { 
    id: 8, 
    title: 'Model Testing', 
    description: 'Evaluate accuracy on unseen data.',
    pythonSnippet: `from sklearn.metrics import accuracy_score

# Generate predictions on the test set
preds = model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {acc:.4f}")

# Check for overfitting
train_acc = model.score(X_train, y_train)
print(f"Training Accuracy:   {train_acc:.4f}")`
  },
  { 
    id: 9, 
    title: 'Sample Predictions', 
    description: 'Inspect individual inference results.',
    pythonSnippet: `# Get probability scores for confidence analysis
probs = model.predict_proba(X_test)[:, 1]

# Create a results DataFrame
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': preds,
    'Confidence': probs
})

# Show samples where the model was unsure (prob ~ 0.5)
uncertain = results[(results['Confidence'] > 0.4) & (results['Confidence'] < 0.6)]
print("Uncertain Predictions:\\n", uncertain.head())`
  },
  { 
    id: 10, 
    title: 'Statistical Metrics', 
    description: 'Confusion Matrix & Classification Report.',
    pythonSnippet: `from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Detailed Classification Report
print("--- Classification Report ---")
print(classification_report(y_test, preds, target_names=['Non-Road', 'Road']))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()`
  },
  { 
    id: 11, 
    title: 'ROC AUC', 
    description: 'Receiver Operating Characteristic Curve.',
    pythonSnippet: `from sklearn.metrics import roc_curve, auc

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()`
  },
];

// Syntax Highlighter Component
const CodeBlock = ({ code }: { code: string }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const keywords = new Set([
    'import', 'from', 'as', 'in', 'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'print',
    'True', 'False', 'None', 'with', 'global', 'lambda'
  ]);
  const functions = new Set([
    'read_csv', 'head', 'info', 'describe', 'fillna', 'mean', 'fit_transform', 'kdeplot', 'figure', 'show', 'title',
    'train_test_split', 'value_counts', 'XGBClassifier', 'fit', 'predict', 'predict_proba', 'accuracy_score',
    'classification_report', 'roc_curve', 'auc', 'plot', 'legend', 'DataFrame', 'SVC', 'DecisionTreeClassifier', 
    'LogisticRegression', 'sum', 'isnull', 'drop', 'sort_values', 'subplot', 'tight_layout', 'zscore', 'all'
  ]);
  const builtins = new Set(['list', 'dict', 'len', 'range', 'enumerate', 'int', 'float', 'str']);

  const lines = code.split('\n');

  const highlightLine = (line: string) => {
    // Regex to split by delimiters but keep them
    const parts = line.split(/([a-zA-Z_]\w*|'.*?'|".*?"|#.*|[(){}\[\],:.<>=+\-*/%])/g);
    
    return parts.map((part, index) => {
      if (!part) return null;
      if (keywords.has(part)) return <span key={index} className="text-[#c678dd] font-semibold">{part}</span>; // Purple
      if (functions.has(part)) return <span key={index} className="text-[#61afef]">{part}</span>; // Blue
      if (builtins.has(part)) return <span key={index} className="text-[#e5c07b]">{part}</span>; // Yellow
      if (part.startsWith("'") || part.startsWith('"')) return <span key={index} className="text-[#98c379]">{part}</span>; // Green
      if (part.trim().startsWith("#")) return <span key={index} className="text-[#7f848e] italic">{part}</span>; // Grey comment
      if (!isNaN(parseFloat(part)) && isFinite(Number(part))) return <span key={index} className="text-[#d19a66]">{part}</span>; // Orange number
      return <span key={index} className="text-[#abb2bf]">{part}</span>; // Default FG
    });
  };

  return (
    <div className="relative font-mono text-[13px] leading-6 group">
        <button 
            onClick={handleCopy}
            className="absolute top-2 right-2 p-1.5 rounded-lg bg-white/10 text-gray-400 hover:text-white hover:bg-white/20 transition-all opacity-0 group-hover:opacity-100 z-10"
            title="Copy Code"
        >
            {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
        </button>
      <div className="flex">
        {/* Line Numbers */}
        <div className="flex-none w-10 text-right pr-3 select-none text-[#4b5263] bg-transparent border-r border-[#282c34]/50">
           {lines.map((_, i) => (
             <div key={i}>{i + 1}</div>
           ))}
        </div>
        {/* Code Content */}
        <div className="flex-1 pl-4 overflow-x-auto custom-scrollbar pb-4">
           {lines.map((line, i) => (
             <div key={i} className="whitespace-pre">
               {highlightLine(line)}
             </div>
           ))}
        </div>
      </div>
    </div>
  );
};

// Helper to get step icon
const getStepIcon = (id: number) => {
  switch (id) {
    case 1: return <Database size={18} />;
    case 2: return <FileText size={18} />;
    case 3: return <Settings size={18} />;
    case 4: return <PieChart size={18} />;
    case 5: return <Layers size={18} />;
    case 6: return <GitBranch size={18} />;
    case 7: return <Cpu size={18} />;
    case 8: return <Activity size={18} />;
    case 9: return <Eye size={18} />;
    case 10: return <BarChart2 size={18} />;
    case 11: return <Activity size={18} />;
    default: return <CheckCircle size={18} />;
  }
};

export default function App() {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [loading, setLoading] = useState(false);
  const [showCode, setShowCode] = useState(true);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  
  // AI Chat & Insight State
  const [aiInsight, setAiInsight] = useState<string>('');
  const [aiLoading, setAiLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  
  // Manual API Key for user convenience
  const [manualApiKey, setManualApiKey] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const skipChatScrollRef = useRef(false);

  // AutoML State
  const [autoMLRunning, setAutoMLRunning] = useState(false);
  const [autoMLCurrentModel, setAutoMLCurrentModel] = useState<string>('');

  // Pipeline State
  const [pipeline, setPipeline] = useState<PipelineContext>({
    rawData: [],
    processedData: [],
    trainData: [],
    testData: [],
    predictions: [],
    metrics: null,
    comparisonResults: [],
    selectedModel: 'XGBoost',
    confusionMatrix: null,
    rocCurve: [],
    trainingHistory: []
  });

  const [trainingEpoch, setTrainingEpoch] = useState(0);
  const [displayedHistory, setDisplayedHistory] = useState<TrainingLog[]>([]);
  const [trainSplit, setTrainSplit] = useState(80); // Train/Test split percentage

  // Colors based on theme
  const colors = darkMode ? {
    bg: 'bg-[#000000]',
    text: 'text-[#f5f5f7]',
    textSec: 'text-gray-400',
    card: 'bg-[#1c1c1e]',
    border: 'border-white/10',
    input: 'bg-[#2c2c2e]',
    hover: 'hover:bg-[#2c2c2e]',
    activeStep: 'bg-[#1c1c1e] shadow-lg border-white/10',
    tableHeader: 'bg-white/5 text-gray-400',
    tableRowHover: 'hover:bg-white/5',
    accent: '#0a84ff',
    accentClass: 'text-[#0a84ff]',
  } : {
    bg: 'bg-[#F2F2F7]',
    text: 'text-[#1d1d1f]',
    textSec: 'text-[#86868b]',
    card: 'bg-white',
    border: 'border-black/[0.04]',
    input: 'bg-[#F5F5F7]',
    hover: 'hover:bg-[#F5F5F7]',
    activeStep: 'bg-white shadow-md border-black/5',
    tableHeader: 'bg-[#F5F5F7] text-[#86868b]',
    tableRowHover: 'hover:bg-[#F5F5F7]',
    accent: '#007aff',
    accentClass: 'text-[#007aff]',
  };

  const toggleTheme = (e: React.MouseEvent) => {
    const x = e.clientX;
    const y = e.clientY;
    const endRadius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    );

    // @ts-ignore
    if (!document.startViewTransition) {
      setDarkMode(!darkMode);
      return;
    }

    // @ts-ignore
    const transition = document.startViewTransition(() => {
      flushSync(() => {
        setDarkMode(!darkMode);
      });
    });

    transition.ready.then(() => {
      document.documentElement.animate(
        {
          clipPath: [
            `circle(0px at ${x}px ${y}px)`,
            `circle(${endRadius}px at ${x}px ${y}px)`
          ],
        },
        {
          duration: 750, // Slowed down for smoother effect
          easing: "cubic-bezier(0.65, 0, 0.35, 1)", // Luxurious easing
          pseudoElement: "::view-transition-new(root)",
        }
      );
    });
  };

  useEffect(() => {
    document.body.className = darkMode ? 'dark' : 'light';
    document.body.style.backgroundColor = darkMode ? '#000000' : '#F2F2F7';
  }, [darkMode]);

  // Scroll to top when step changes
  useEffect(() => {
    if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
    // Reset Chat
    // prevent chat auto-scroll when switching steps
    skipChatScrollRef.current = true;
    setChatMessages([]);
    setIsChatting(false);
  }, [currentStep]);

  // Recompute train/test slices and regenerate predictions/metrics when split changes
  useEffect(() => {
    // Ensure we have processed data to slice
    setPipeline(prev => {
      const processed = prev.processedData.length > 0 ? prev.processedData : generateMockData(1000).map(d => ({
        ...d,
        r: parseFloat((d.r / 255).toFixed(4)),
        g: parseFloat((d.g / 255).toFixed(4)),
        b: parseFloat((d.b / 255).toFixed(4))
      }));

      const total = Math.max(processed.length, 100);
      const splitIdx = Math.floor((trainSplit / 100) * total);
      const trainData = processed.slice(0, splitIdx);
      const testData = processed.slice(splitIdx);

      // Determine target accuracy similar to executeStep
      let targetAccuracy = 0.94;
      if (prev.metrics?.accuracy) targetAccuracy = prev.metrics.accuracy;
      else if (prev.selectedModel === 'Logistic Regression') targetAccuracy = 0.76;
      else if (prev.selectedModel === 'Decision Tree') targetAccuracy = 0.82;
      else if (prev.selectedModel === 'SVM') targetAccuracy = 0.89;

      const newPreds = (testData.length > 0) ? generatePredictions(testData, targetAccuracy) : [];

      // Recalculate metrics so the UI shows updated accuracy
      const res = newPreds.length > 0 ? calculateMetrics(newPreds, prev.selectedModel) : { metrics: prev.metrics, cm: prev.confusionMatrix };

      return {
        ...prev,
        processedData: processed,
        trainData,
        testData,
        predictions: newPreds,
        metrics: res.metrics,
        confusionMatrix: res.cm
      };
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainSplit]);

  // Scroll chat to bottom
  useEffect(() => {
      if (skipChatScrollRef.current) {
        // skip one automatic scroll triggered by step change
        skipChatScrollRef.current = false;
        return;
      }
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, aiInsight]);


  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFile(file);
    setLoading(true);

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.split('\n').filter(l => l.trim() !== '');
      const count = Math.min(lines.length - 1, 2000);
      const newData = generateMockData(count > 0 ? count : 100);
      
      setPipeline(prev => ({
        ...prev,
        rawData: newData,
        processedData: [],
        trainData: [],
        testData: [],
        predictions: [],
        metrics: null,
        rocCurve: []
      }));
      setLoading(false);
    };
    reader.readAsText(file);
  };

  const generateSampleCSV = (type: 'urban' | 'forest' | 'desert' | 'night') => {
    let csvContent = "id,r,g,b,ndvi,texture,isRoad\n";
    const count = 500;
    
    for (let i = 0; i < count; i++) {
        const isRoad = Math.random() > 0.7 ? 1 : 0;
        let r=0, g=0, b=0, ndvi=0, texture=0;

        if (isRoad) {
             const base = type === 'night' ? 40 : (type === 'desert' ? 140 : 100);
             r = base + Math.random() * 20;
             g = base + Math.random() * 20;
             b = base + Math.random() * 20;
             ndvi = -0.1 + Math.random() * 0.15;
             texture = 0.1 + Math.random() * 0.2;
        } else {
             if (type === 'urban') { 
                r = 150 + Math.random() * 50;
                g = 150 + Math.random() * 50;
                b = 160 + Math.random() * 50;
                ndvi = 0.1 + Math.random() * 0.2;
                texture = 0.6 + Math.random() * 0.4;
             } else if (type === 'forest') { 
                r = 20 + Math.random() * 30;
                g = 100 + Math.random() * 80;
                b = 20 + Math.random() * 30;
                ndvi = 0.6 + Math.random() * 0.3;
                texture = 0.8 + Math.random() * 0.2;
             } else if (type === 'desert') {
                r = 200 + Math.random() * 55;
                g = 180 + Math.random() * 55;
                b = 120 + Math.random() * 55;
                ndvi = 0.05 + Math.random() * 0.1;
                texture = 0.3 + Math.random() * 0.2;
             } else if (type === 'night') {
                r = 5 + Math.random() * 15;
                g = 5 + Math.random() * 15;
                b = 10 + Math.random() * 20;
                ndvi = 0;
                texture = 0.1;
             }
        }
        
        csvContent += `${i},${Math.floor(r)},${Math.floor(g)},${Math.floor(b)},${ndvi.toFixed(4)},${texture.toFixed(4)},${isRoad}\n`;
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `${type}_dataset_full.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleNext = () => {
    if (currentStep < STEPS.length) {
      setCurrentStep(prev => prev + 1);
      setAiInsight(''); 
    }
  };

  const handlePrev = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
      setAiInsight('');
    }
  };

  const handleSendMessage = async () => {
      if (!chatInput.trim()) return;
      
      setIsChatting(true);
      const userMsg: ChatMessage = { role: 'user', text: chatInput };
      setChatInput('');

      // Update state with user message
      setChatMessages(prev => [...prev, userMsg]);

      // Get Context
      const stepName = STEPS[currentStep - 1].title;
      const currentContext = getContextForStep(currentStep, pipeline, uploadedFile);

      // Generate Reply
      const replyText = await generateChatResponse([...chatMessages, userMsg], currentContext, stepName, manualApiKey);
      
      // Add AI response
      setChatMessages(prev => [...prev, { role: 'ai', text: replyText }]);
      setIsChatting(false);
  };

  useEffect(() => {
    if (currentStep === 7 && pipeline.trainingHistory.length > 0 && !autoMLRunning) {
      setTrainingEpoch(0);
      setDisplayedHistory([]);
      const interval = setInterval(() => {
        setTrainingEpoch(prev => {
          if (prev >= pipeline.trainingHistory.length) {
            clearInterval(interval);
            return prev;
          }
          const nextEpoch = prev + 1;
          setDisplayedHistory(pipeline.trainingHistory.slice(0, nextEpoch));
          return nextEpoch;
        });
      }, 50); 
      return () => clearInterval(interval);
    }
  }, [currentStep, pipeline.trainingHistory, autoMLRunning]);

  const updatePythonSnippet = (modelType: ModelType) => {
    let snippet = '';
    switch(modelType) {
        case 'XGBoost':
            snippet = `from xgboost import XGBClassifier\n\n# Initialize XGBoost\nmodel = XGBClassifier(\n    n_estimators=100,\n    learning_rate=0.1,\n    max_depth=5\n)\n\nprint("Training XGBoost...")\nmodel.fit(X_train, y_train)`;
            break;
        case 'SVM':
            snippet = `from sklearn.svm import SVC\n\n# Initialize Support Vector Machine\nmodel = SVC(\n    kernel='rbf',\n    C=1.0,\n    probability=True\n)\n\nprint("Training SVM...")\nmodel.fit(X_train, y_train)`;
            break;
        case 'Decision Tree':
            snippet = `from sklearn.tree import DecisionTreeClassifier\n\n# Initialize Decision Tree\nmodel = DecisionTreeClassifier(\n    criterion='gini',\n    max_depth=10,\n    random_state=42\n)\n\nprint("Training Decision Tree...")\nmodel.fit(X_train, y_train)`;
            break;
        case 'Logistic Regression':
            snippet = `from sklearn.linear_model import LogisticRegression\n\n# Initialize Logistic Regression\nmodel = LogisticRegression(\n    solver='liblinear',\n    C=1.0,\n    max_iter=1000\n)\n\nprint("Training Logistic Regression...")\nmodel.fit(X_train, y_train)`;
            break;
        case 'AutoML':
            snippet = `models = {\n    'Logistic Regression': LogisticRegression(),\n    'Decision Tree': DecisionTreeClassifier(),\n    'SVM': SVC(probability=True),\n    'XGBoost': XGBClassifier()\n}\n\nresults = {}\nprint("Starting AutoML Pipeline...")\n\nfor name, model in models.items():\n    print(f"Training {name}...")\n    model.fit(X_train, y_train)\n    acc = model.score(X_test, y_test)\n    results[name] = acc\n\n# Select Best Model\nbest_model_name = max(results, key=results.get)\nprint(f"\\n🏆 Best Model: {best_model_name} (Acc: {results[best_model_name]:.4f})")`;
            break;
    }
    STEPS[6].pythonSnippet = snippet; // Update step 7 snippet
  };

  const trainModel = async (model: ModelType) => {
      // Normal training
      setPipeline(prev => ({ ...prev, selectedModel: model, comparisonResults: [] }));
      updatePythonSnippet(model);
      
      // AutoML Simulation
      if (model === 'AutoML') {
          setAutoMLRunning(true);
          const models: ModelType[] = ['Logistic Regression', 'Decision Tree', 'SVM', 'XGBoost'];
          const newResults: ModelMetrics[] = [];
          
          for (const m of models) {
              setAutoMLCurrentModel(m);
              // Wait for "training"
              await new Promise(resolve => setTimeout(resolve, 800));
              
              // Use simulateModelPerformance to get robust metrics
              const res = simulateModelPerformance(m);
              
              newResults.push(res);
              // Sort instantly to show current leaderboard
              setPipeline(prev => ({
                ...prev,
                comparisonResults: [...newResults].sort((a,b) => b.accuracy - a.accuracy)
              }));
          }
          
          setAutoMLRunning(false);
          setAutoMLCurrentModel('');
          
          // Set best model as selected
          const best = newResults.sort((a,b) => b.accuracy - a.accuracy)[0];
          setPipeline(prev => ({
              ...prev,
              selectedModel: 'AutoML',
              metrics: best
          }));
      }
  };

  // Helper to extract context string for AI
  const getContextForStep = (step: number, pl: PipelineContext, file: File | null) => {
    switch (step) {
        case 1: return pl.rawData.length === 0 ? "Default data 1000 rows" : `File: ${file?.name}, Rows: ${pl.rawData.length}`;
        case 2: return `Head: ${JSON.stringify(pl.rawData.slice(0,2))}`;
        case 3: return "Data normalized [0-1].";
        case 4: return "Green channel separation visible.";
        case 5: return "Features: R,G,B,NDVI,Texture.";
        case 6: return `Split 80/20. Train: ${pl.trainData.length}.`;
        case 7: return `Training ${pl.selectedModel}.`;
        case 8: return "Inference complete.";
        case 9: return "Reviewing samples.";
        case 10: return `Accuracy: ${(pl.metrics?.accuracy ? (pl.metrics.accuracy * 100).toFixed(1) : 0)}`;
        case 11: return `AUC: ${pl.metrics?.auc.toFixed(3) || 0.8}`;
        default: return "";
    }
  };

  const executeStep = useCallback(async () => {
    setLoading(true);
    setAiLoading(true);
    
    await new Promise(resolve => setTimeout(resolve, 500));

    let newPipeline = { ...pipeline };

    // --- CASCADING DATA CHECKS (Fixes empty table issues) ---
    // If user jumps to a later step, ensure previous data exists

    // 1. Raw Data (Required for Step 1+)
    if (newPipeline.rawData.length === 0) {
      newPipeline.rawData = generateMockData(1000);
    }

    // 2. Processed Data (Required for Step 3+)
    if (currentStep >= 3 && newPipeline.processedData.length === 0) {
      newPipeline.processedData = newPipeline.rawData.map(d => ({
        ...d,
        r: parseFloat((d.r / 255).toFixed(4)), 
        g: parseFloat((d.g / 255).toFixed(4)),
        b: parseFloat((d.b / 255).toFixed(4))
      }));
    }

    // 3. Train/Test Split (Required for Step 6+)
    if (currentStep >= 6 && newPipeline.trainData.length === 0) {
      const splitIdx = Math.floor(newPipeline.processedData.length * 0.8);
      newPipeline.trainData = newPipeline.processedData.slice(0, splitIdx);
      newPipeline.testData = newPipeline.processedData.slice(splitIdx);
    }

    // 4. Training History (Required for Step 7+)
    if (currentStep >= 7 && newPipeline.trainingHistory.length === 0) {
      newPipeline.trainingHistory = generateTrainingHistory(50);
    }

    // 5. Predictions (Required for Step 8+)
    if (currentStep >= 8 && newPipeline.predictions.length === 0) {
       // Check if we have a target accuracy from AutoML/Metrics
       let targetAccuracy = 0.94; // Default baseline (XGBoost)
       if (newPipeline.metrics?.accuracy) {
         targetAccuracy = newPipeline.metrics.accuracy;
       } else if (newPipeline.selectedModel === 'Logistic Regression') targetAccuracy = 0.76;
       else if (newPipeline.selectedModel === 'Decision Tree') targetAccuracy = 0.82;
       else if (newPipeline.selectedModel === 'SVM') targetAccuracy = 0.89;

       newPipeline.predictions = generatePredictions(newPipeline.testData, targetAccuracy);
    }

    // --- STEP SPECIFIC LOGIC ---
    // (Used to recalculate metrics if predictions changed)
    if (currentStep === 10) {
        // Recalculate metrics based on current predictions
        const modelToUse = pipeline.selectedModel === 'AutoML' ? 'XGBoost' : pipeline.selectedModel;
        const res = calculateMetrics(newPipeline.predictions, modelToUse);
        // Only update if we don't have better metrics from AutoML already, or if we want to sync perfectly
        // We prefer the calculated ones to match the confusion matrix
        newPipeline.metrics = res.metrics;
        newPipeline.confusionMatrix = res.cm;
    }

    if (currentStep === 11) {
        const rocModel = pipeline.selectedModel === 'AutoML' ? 'XGBoost' : pipeline.selectedModel;
        newPipeline.rocCurve = generateROC(rocModel);
    }

    setPipeline(newPipeline);
    setLoading(false);

    // AI Insight
    const context = getContextForStep(currentStep, newPipeline, uploadedFile);
    generateAIInsight(context, STEPS[currentStep - 1].title, manualApiKey).then(insight => {
        setAiInsight(insight);
        setAiLoading(false);
        // Clear chat messages so only the static insight shows initially
        setChatMessages([]);
    });

  }, [currentStep, pipeline, uploadedFile, manualApiKey]);

  useEffect(() => {
    executeStep();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  // --- RENDER HELPERS ---

  const renderContent = () => {
    if (loading) return (
      <div className="flex flex-col items-center justify-center h-full min-h-[400px] space-y-4">
        <div 
            className="w-10 h-10 border-4 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: colors.accent, borderTopColor: 'transparent' }}
        ></div>
        <p className={`${colors.textSec} font-medium text-sm animate-pulse tracking-wide`}>PROCESSING</p>
      </div>
    );

    switch(currentStep) {
      case 1: // Load
        return (
          <div className="flex flex-col justify-start animate-fade-in space-y-6">
             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className={`${colors.card} p-6 rounded-3xl border ${colors.border} flex flex-col justify-between ${colors.hover} transition-colors shadow-sm`}>
                    <div>
                        <div className={`flex items-center space-x-2 ${colors.textSec} mb-2`}>
                             <Database size={16} />
                             <span className="text-xs uppercase font-bold tracking-wider">Active Dataset</span>
                        </div>
                        <div className={`text-2xl font-bold ${colors.text} mb-1`}>
                            {uploadedFile ? uploadedFile.name : 'satellite_data.csv'}
                        </div>
                        <div className="text-sm text-gray-500 font-mono">
                            {pipeline.rawData.length} rows • {uploadedFile ? (uploadedFile.size/1024).toFixed(1) + ' KB' : '142 KB'}
                        </div>
                    </div>
                </div>

                <div className={`${colors.card} p-6 rounded-3xl border border-dashed ${colors.border} hover:border-[${colors.accent}] hover:bg-[${colors.accent}]/5 transition-all flex flex-col items-center justify-center text-center cursor-pointer group relative overflow-hidden shadow-sm`}
                     style={{ borderColor: 'transparent' }} 
                >
                    <input 
                        type="file" 
                        accept=".csv" 
                        onChange={handleFileUpload} 
                        className="absolute inset-0 opacity-0 cursor-pointer z-20"
                    />
                     <div className={`absolute inset-0 border border-dashed ${darkMode ? 'border-gray-700 group-hover:border-blue-500' : 'border-gray-300 group-hover:border-blue-500'} rounded-3xl pointer-events-none transition-colors`}></div>

                    <div className={`p-3 ${darkMode ? 'bg-white/10' : 'bg-gray-100'} rounded-full mb-3 group-hover:scale-110 transition-transform`}>
                        <UploadCloud style={{ color: colors.accent }} />
                    </div>
                    <div className={`text-sm font-semibold ${colors.text}`}>Upload Custom CSV</div>
                    <div className={`text-xs ${colors.textSec} mt-1`}>Drag & drop or click to browse</div>
                </div>
             </div>
             
             {/* Sample Datasets Downloader */}
             <div className={`${colors.card} rounded-3xl p-6 border ${colors.border} shadow-sm`}>
                <h4 className={`text-sm ${colors.textSec} mb-4 font-medium uppercase tracking-wider`}>Download Sample Datasets</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <button onClick={() => generateSampleCSV('urban')} className={`p-4 rounded-2xl border ${colors.border} ${colors.hover} transition-all flex flex-col items-center justify-center space-y-2 group`}>
                        <div className="p-2 bg-blue-500/10 rounded-full group-hover:bg-blue-500/20 text-blue-500"><Building2 size={20} /></div>
                        <span className={`text-xs font-medium ${colors.text}`}>Urban</span>
                        <Download size={14} className={colors.textSec} />
                    </button>
                    <button onClick={() => generateSampleCSV('forest')} className={`p-4 rounded-2xl border ${colors.border} ${colors.hover} transition-all flex flex-col items-center justify-center space-y-2 group`}>
                        <div className="p-2 bg-emerald-500/10 rounded-full group-hover:bg-emerald-500/20 text-emerald-500"><Trees size={20} /></div>
                        <span className={`text-xs font-medium ${colors.text}`}>Forest</span>
                        <Download size={14} className={colors.textSec} />
                    </button>
                    <button onClick={() => generateSampleCSV('desert')} className={`p-4 rounded-2xl border ${colors.border} ${colors.hover} transition-all flex flex-col items-center justify-center space-y-2 group`}>
                        <div className="p-2 bg-orange-500/10 rounded-full group-hover:bg-orange-500/20 text-orange-500"><Sun size={20} /></div>
                        <span className={`text-xs font-medium ${colors.text}`}>Desert</span>
                        <Download size={14} className={colors.textSec} />
                    </button>
                    <button onClick={() => generateSampleCSV('night')} className={`p-4 rounded-2xl border ${colors.border} ${colors.hover} transition-all flex flex-col items-center justify-center space-y-2 group`}>
                        <div className="p-2 bg-purple-500/10 rounded-full group-hover:bg-purple-500/20 text-purple-500"><Moon size={20} /></div>
                        <span className={`text-xs font-medium ${colors.text}`}>Night</span>
                        <Download size={14} className={colors.textSec} />
                    </button>
                </div>
             </div>

             <div className={`${colors.card} rounded-3xl p-6 border ${colors.border} shadow-sm`}>
                 <h4 className={`text-sm ${colors.textSec} mb-4 font-medium uppercase tracking-wider`}>Preview</h4>
                 <div className="overflow-hidden rounded-xl border border-gray-200 dark:border-white/10">
                   <table className={`w-full text-sm text-left ${colors.text}`}>
                      <thead className={`text-xs uppercase ${colors.tableHeader}`}>
                        <tr>
                          {['ID', 'R', 'G', 'B', 'NDVI', 'Texture', 'Label'].map(h => (
                            <th key={h} className="px-4 py-3 font-semibold">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                        {pipeline.rawData.slice(0, 4).map(r => (
                          <tr key={r.id} className={`${colors.tableRowHover} transition-colors`}>
                             <td className="px-4 py-2 font-mono text-xs opacity-70">{r.id}</td>
                             <td className="px-4 py-2">{r.r}</td>
                             <td className="px-4 py-2">{r.g}</td>
                             <td className="px-4 py-2">{r.b}</td>
                             <td className="px-4 py-2 text-emerald-500">{r.ndvi}</td>
                             <td className="px-4 py-2">{r.texture}</td>
                             <td className="px-4 py-2">{r.isRoad}</td>
                          </tr>
                        ))}
                      </tbody>
                   </table>
                   <div className={`px-4 py-2 text-xs text-center ${colors.textSec} bg-gray-50 dark:bg-white/5`}>
                      ... {pipeline.rawData.length - 4} more rows
                   </div>
                 </div>
             </div>
          </div>
        );
      case 2: // Inspect
        return (
          <div className="flex flex-col animate-slide-up h-full">
            <div className={`${colors.card} rounded-3xl overflow-hidden border ${colors.border} shadow-md`}>
                <div className={`px-6 py-4 border-b ${colors.border} flex justify-between items-center ${darkMode ? 'bg-white/5' : 'bg-gray-50/80'} backdrop-blur-md`}>
                    <div className="flex space-x-2">
                        <div className="w-3 h-3 rounded-full bg-[#ff453a]"></div>
                        <div className="w-3 h-3 rounded-full bg-[#ff9f0a]"></div>
                        <div className="w-3 h-3 rounded-full bg-[#30d158]"></div>
                    </div>
                    <span className={`text-xs font-mono ${colors.textSec}`}>DataFrame.head()</span>
                </div>
              <div className="overflow-x-auto">
                <table className={`w-full text-sm text-left ${colors.text}`}>
                  <thead className={`text-xs uppercase ${colors.tableHeader}`}>
                    <tr>
                      <th className="px-6 py-4 font-semibold">ID</th>
                      <th className="px-6 py-4 font-semibold">Red</th>
                      <th className="px-6 py-4 font-semibold">Green</th>
                      <th className="px-6 py-4 font-semibold">Blue</th>
                      <th className="px-6 py-4 font-semibold">NDVI</th>
                      <th className="px-6 py-4 font-semibold text-right">Label</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y ${darkMode ? 'divide-white/5' : 'divide-gray-100'}`}>
                    {pipeline.rawData.slice(0, 7).map((row) => (
                      <tr key={row.id} className={`${colors.tableRowHover} transition-colors`}>
                        <td className={`px-6 py-3 font-mono ${colors.textSec}`}>{row.id}</td>
                        <td className="px-6 py-3">{row.r}</td>
                        <td className="px-6 py-3">{row.g}</td>
                        <td className="px-6 py-3">{row.b}</td>
                        <td className="px-6 py-3 text-[#30d158]">{row.ndvi}</td>
                        <td className="px-6 py-3 text-right">
                            <span className={`px-2.5 py-1 rounded-full text-[10px] uppercase font-bold tracking-wide ${row.isRoad ? 'bg-[#ff9f0a]/20 text-[#ff9f0a]' : 'bg-[#0a84ff]/20 text-[#0a84ff]'}`}>
                                {row.isRoad ? 'ROAD' : 'OTHER'}
                            </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      case 3: // Process
        return (
           <div className="flex flex-col space-y-4 animate-fade-in justify-center min-h-[500px]">
              <div className="grid grid-cols-3 gap-4">
                  {[
                      { title: "Missing Values", val: "0 Found", status: "Clean" },
                      { title: "Outliers", val: "12 Removed", status: "Fixed" },
                      { title: "Scaling", val: "MinMax", status: "Done" }
                  ].map((stat, i) => (
                      <div key={i} className={`${colors.card} p-6 rounded-3xl border ${colors.border} text-center shadow-sm`}>
                          <div className={`${colors.textSec} text-xs uppercase tracking-wider mb-2`}>{stat.title}</div>
                          <div className={`text-2xl font-bold ${colors.text} mb-1`}>{stat.val}</div>
                          <div className="text-[#30d158] text-xs font-semibold bg-[#30d158]/10 inline-block px-2 py-0.5 rounded-full">{stat.status}</div>
                      </div>
                  ))}
              </div>
              
              <div className={`${colors.card} p-6 rounded-3xl border ${colors.border} flex-1 flex flex-col justify-center items-center relative overflow-hidden shadow-sm min-h-[200px]`}>
                   <div className="absolute inset-0 bg-gradient-to-r from-[#0a84ff]/5 to-[#bf5af2]/5"></div>
                   <div className="relative z-10 w-full max-w-lg">
                       <div className={`flex justify-between text-xs ${colors.textSec} mb-2 font-medium`}>
                           <span>Normalization Progress</span>
                           <span>100%</span>
                       </div>
                       <div className={`w-full ${colors.input} rounded-full h-2 overflow-hidden`}>
                           <div className="bg-gradient-to-r from-[#0a84ff] to-[#5e5ce6] h-full rounded-full" style={{ width: '100%' }}></div>
                       </div>
                       <p className={`text-center text-xs ${colors.textSec} mt-4 font-mono`}>
                           Values transformed from [0-255] to [0.0-1.0]
                       </p>
                   </div>
              </div>
           </div>
        );
      case 4: // EDA
        return (
            <div className={`animate-slide-up ${colors.card} p-6 rounded-3xl border ${colors.border} min-h-[500px] flex flex-col shadow-md`}>
                <div className="flex justify-between items-center mb-6">
                   <div>
                       <h4 className={`${colors.text} font-bold`}>Feature Distribution</h4>
                       <p className={`text-xs ${colors.textSec}`}>Green Channel Intensity</p>
                   </div>
                   <div className="flex space-x-2">
                       <span className={`flex items-center text-[10px] ${colors.textSec}`}><div className="w-2 h-2 rounded-full bg-[#ff9f0a] mr-1"></div> Road</span>
                       <span className={`flex items-center text-[10px] ${colors.textSec}`}><div className="w-2 h-2 rounded-full bg-[#0a84ff] mr-1"></div> Other</span>
                   </div>
                </div>
                <div className="flex-1 min-h-0">
                    <FeatureDistChart data={pipeline.rawData} darkMode={darkMode} />
                </div>
            </div>
        );
      case 5: // Features
        return (
            <div className="flex flex-col justify-center min-h-[500px] space-y-6 animate-fade-in">
                <div className={`${colors.card} p-8 rounded-3xl border ${colors.border} relative overflow-hidden group shadow-sm`}>
                    <div className="absolute top-0 right-0 w-32 h-32 bg-[#0a84ff]/10 blur-3xl rounded-full group-hover:bg-[#0a84ff]/20 transition-all"></div>
                    <h4 className={`${colors.textSec} text-xs font-bold uppercase tracking-widest mb-4`}>Input Matrix (X)</h4>
                    <div className="flex flex-wrap gap-3">
                        {['Red', 'Green', 'Blue', 'NDVI', 'Texture'].map(f => (
                            <span key={f} className={`px-5 py-2.5 ${colors.input} ${colors.text} rounded-xl text-sm font-medium border ${colors.border} shadow-sm`}>
                                {f}
                            </span>
                        ))}
                    </div>
                </div>
                <div className={`${colors.card} p-8 rounded-3xl border ${colors.border} relative overflow-hidden group shadow-sm`}>
                     <div className="absolute top-0 right-0 w-32 h-32 bg-[#ff9f0a]/10 blur-3xl rounded-full group-hover:bg-[#ff9f0a]/20 transition-all"></div>
                    <h4 className={`${colors.textSec} text-xs font-bold uppercase tracking-widest mb-4`}>Target Vector (y)</h4>
                     <span className="px-5 py-2.5 bg-[#ff9f0a]/20 text-[#ff9f0a] rounded-xl text-sm font-bold border border-[#ff9f0a]/20 shadow-sm inline-block">
                        isRoad
                    </span>
                    <p className={`text-xs ${colors.textSec} mt-4 max-w-md`}>
                        Binary classification target. 0 represents background (vegetation/water), 1 represents road surfaces.
                    </p>
                </div>
            </div>
        );
      case 6: // Split
        const trainPercentage = trainSplit;
        const testPercentage = 100 - trainSplit;
        const totalSamples = Math.max(pipeline.processedData.length, 1000);
        const trainCount = Math.floor((trainPercentage / 100) * totalSamples);
        const testCount = totalSamples - trainCount;
        
        // Impact on accuracy: more train data generally improves accuracy up to a point
        // Less training data = lower potential accuracy
        const baseAccuracy = 0.93; // Base accuracy with 80/20 split
        const accuracyImpact = Math.pow((trainPercentage / 80), 0.3); // Diminishing returns
        const predictedAccuracy = Math.min(0.99, baseAccuracy * accuracyImpact);

        return (
             <div className={`${colors.card} p-8 rounded-3xl border ${colors.border} animate-slide-up min-h-[500px] flex flex-col justify-start shadow-sm space-y-8`}>
                 <div>
                    <h3 className={`${colors.text} font-bold text-lg mb-2`}>Data Split Configuration</h3>
                    <p className={`${colors.textSec} text-sm`}>Adjust the training and testing ratio to see how it affects model accuracy</p>
                 </div>

                 <div className="flex items-center justify-center space-x-12 mb-6">
                     <div className="text-center">
                         <div className={`text-5xl font-bold ${colors.text} mb-1 tracking-tighter`}>{trainCount.toLocaleString()}</div>
                         <div className={`text-[10px] ${colors.textSec} uppercase tracking-widest font-bold`}>Training Samples</div>
                         <div className={`text-xs font-mono ${colors.textSec} mt-1`}>{trainPercentage}%</div>
                     </div>
                     <div className={`h-16 w-px ${darkMode ? 'bg-white/10' : 'bg-black/10'}`}></div>
                     <div className="text-center">
                         <div className={`text-5xl font-bold ${colors.text} mb-1 tracking-tighter`}>{testCount.toLocaleString()}</div>
                         <div className={`text-[10px] ${colors.textSec} uppercase tracking-widest font-bold`}>Testing Samples</div>
                         <div className={`text-xs font-mono ${colors.textSec} mt-1`}>{testPercentage}%</div>
                     </div>
                 </div>
                 
                 <div className="w-full max-w-2xl mx-auto space-y-6">
                    {/* Split Ratio Slider */}
                    <div className={`${colors.input} p-6 rounded-2xl border ${colors.border}`}>
                       <div className={`flex justify-between text-xs ${colors.textSec} mb-4 uppercase tracking-widest font-bold`}>
                           <span>Train / Test Split</span>
                           <span>{trainPercentage}% / {testPercentage}%</span>
                       </div>
                         <input
                           type="range"
                           min="10"
                           max="90"
                           value={trainSplit}
                           onChange={(e) => setTrainSplit(Number(e.target.value))}
                           className="w-full h-2 rounded-full appearance-none cursor-pointer"
                           style={{ WebkitAppearance: 'none', ['--p' as any]: `${trainPercentage}%` }}
                         />
                         <style>{`
                           /* Track styling uses CSS variable --p (percentage) set on the input element */
                           input[type="range"] { position: relative; z-index: 1; }

                           input[type="range"]::-webkit-slider-runnable-track {
                             height: 8px;
                             border-radius: 9999px;
                             background: linear-gradient(to right, #0a84ff 0%, #0a84ff var(--p), #bf5af2 var(--p), #bf5af2 100%);
                           }

                           input[type="range"]::-webkit-slider-thumb {
                             -webkit-appearance: none;
                             appearance: none;
                             width: 20px;
                             height: 20px;
                             border-radius: 50%;
                             background: white;
                             cursor: pointer;
                             box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                             border: 3px solid #ffffff;
                             position: relative;
                             z-index: 2;
                             margin-top: -6px; /* center the thumb vertically over the track */
                           }

                           /* Firefox */
                           input[type="range"]::-moz-range-track {
                             height: 8px;
                             border-radius: 9999px;
                             background: linear-gradient(to right, #0a84ff 0%, #0a84ff var(--p), #bf5af2 var(--p), #bf5af2 100%);
                           }
                           input[type="range"]::-moz-range-thumb {
                             width: 20px;
                             height: 20px;
                             border-radius: 50%;
                             background: white;
                             cursor: pointer;
                             box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                             border: 3px solid #ffffff;
                             position: relative;
                             z-index: 2;
                           }
                         `}</style>
                       <div className="flex justify-between text-[10px] text-gray-500 mt-2">
                           <span>10%</span>
                           <span>50%</span>
                           <span>90%</span>
                       </div>
                    </div>

                 </div>
             </div>
        );
      case 7: // Train (UPDATED: Model Selection)
        const models: {id: ModelType, icon: any, desc: string}[] = [
            { id: 'Logistic Regression', icon: Binary, desc: 'Linear baseline.' },
            { id: 'Decision Tree', icon: GitBranch, desc: 'Tree-based model.' },
            { id: 'XGBoost', icon: Zap, desc: 'Gradient boosting.' },
            { id: 'SVM', icon: Activity, desc: 'High dimensional.' },
        ];

        return (
            <div className={`animate-fade-in flex flex-col space-y-6 min-h-[500px]`}>
                {/* PRIMARY ACTION: AUTOML */}
                 <button 
                    onClick={() => trainModel('AutoML')}
                    disabled={autoMLRunning}
                    className={`w-full ${colors.card} border-2 ${pipeline.selectedModel === 'AutoML' ? `border-[#bf5af2]` : colors.border} p-6 rounded-3xl flex items-center justify-between group transition-all hover:bg-gradient-to-r hover:from-[#bf5af2]/10 hover:to-[#5e5ce6]/10 shadow-lg`}
                    style={{ borderColor: pipeline.selectedModel === 'AutoML' ? '#bf5af2' : undefined }}
                >
                    <div className="flex items-center space-x-5">
                         <div className="p-3 rounded-2xl bg-gradient-to-br from-[#bf5af2] to-[#5e5ce6] text-white shadow-xl shadow-purple-500/30">
                             {autoMLRunning ? <div className="animate-spin"><Zap size={24} /></div> : <Brain size={24} />}
                         </div>
                         <div className="text-left">
                             <div className={`${colors.text} font-bold text-lg`}>{autoMLRunning ? "Training all models..." : "Run AutoML Comparison"}</div>
                             <div className={`${colors.textSec} text-sm`}>Train Logistic Regression, Decision Tree, SVM, & XGBoost</div>
                         </div>
                    </div>
                    <div className="bg-[#bf5af2]/10 px-4 py-2 rounded-full text-[#bf5af2] text-xs font-bold tracking-wide">
                        RECOMMENDED
                    </div>
                </button>

                <div className="flex items-center my-2">
                    <div className={`h-px flex-1 ${colors.border} bg-current opacity-10`}></div>
                    <span className={`px-4 text-xs ${colors.textSec} uppercase tracking-widest`}>Or Select Manually</span>
                    <div className={`h-px flex-1 ${colors.border} bg-current opacity-10`}></div>
                </div>
                
                <div className="grid grid-cols-4 gap-3">
                    {models.map(m => (
                        <button 
                            key={m.id}
                            disabled={autoMLRunning}
                            onClick={() => trainModel(m.id)}
                            className={`${colors.card} border ${pipeline.selectedModel === m.id && pipeline.selectedModel !== 'AutoML' ? `border-[${colors.accent}] ring-1 ring-[${colors.accent}]` : colors.border} p-3 rounded-xl text-center transition-all hover:scale-[1.02] shadow-sm group ${autoMLRunning ? 'opacity-50 cursor-not-allowed' : ''} flex flex-col items-center justify-center space-y-2 h-24`}
                            style={{ borderColor: pipeline.selectedModel === m.id && pipeline.selectedModel !== 'AutoML' ? colors.accent : undefined }}
                        >
                            <div className={`p-1.5 rounded-lg ${pipeline.selectedModel === m.id ? 'bg-[#0a84ff] text-white' : 'bg-[#0a84ff]/10 text-[#0a84ff]'}`}>
                                <m.icon size={16} />
                            </div>
                            <span className={`${colors.text} font-medium text-[10px] leading-tight`}>{m.id}</span>
                        </button>
                    ))}
                </div>

                {/* Training Viz */}
                {pipeline.selectedModel && !autoMLRunning && pipeline.selectedModel !== 'AutoML' && (
                    <div className={`${colors.card} p-6 rounded-3xl border ${colors.border} mt-4 shadow-md`}>
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <div className="flex items-center space-x-2">
                                     <h3 className={`text-lg font-bold ${colors.text}`}>Training Process</h3>
                                     {trainingEpoch < 50 && <span className="relative flex h-3 w-3">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#30d158] opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-3 w-3 bg-[#30d158]"></span>
                                     </span>}
                                </div>
                                <p className={`text-xs ${colors.textSec} font-medium mt-1`}>
                                    {pipeline.selectedModel} • Epoch {trainingEpoch}/50
                                </p>
                            </div>
                        </div>

                        <div className={`flex-1 w-full ${darkMode ? 'bg-black' : 'bg-gray-50'} rounded-2xl border ${colors.border} mb-4 relative p-4 overflow-hidden h-48`}>
                            {displayedHistory.length > 0 && <TrainingLossChart data={displayedHistory} darkMode={darkMode} />}
                        </div>
                    </div>
                )}
                
                {/* AutoML Live Leaderboard */}
                {pipeline.comparisonResults.length > 0 && (
                     <div className={`${colors.card} p-4 rounded-3xl border ${colors.border} mt-4`}>
                        <h4 className={`text-xs ${colors.textSec} uppercase tracking-widest font-bold mb-3`}>Comparison Leaderboard</h4>
                        <div className="space-y-2">
                             {pipeline.comparisonResults.map((res, idx) => (
                                 <div key={res.modelName} className={`flex items-center justify-between p-3 rounded-xl ${idx === 0 ? 'bg-[#30d158]/10 border border-[#30d158]/20' : `${colors.input}`} animate-slide-up`}>
                                      <div className="flex items-center space-x-2">
                                         <span className={`text-xs font-bold w-4 ${idx===0 ? 'text-[#30d158]' : colors.textSec}`}>#{idx+1}</span>
                                         <span className={`${colors.text} text-sm font-medium`}>{res.modelName}</span>
                                      </div>
                                      <span className={`${idx === 0 ? 'text-[#30d158]' : colors.textSec} font-mono text-sm`}>{(res.accuracy * 100).toFixed(1)}</span>
                                 </div>
                             ))}
                        </div>
                     </div>
                )}
            </div>
        );
      case 8: // Testing
         // Generate unique, realistic accuracy scores per model based on model type and split
         const getModelAccuracy = (modelName: string, splitRatio: number) => {
           // Base accuracy for each model (0-1)
           const baseAccuracies: { [key: string]: number } = {
             'Logistic Regression': 0.76,
             'Decision Tree': 0.82,
             'Random Forest': 0.91,
             'XGBoost': 0.94,
             'SVM': 0.89,
             'SVM (RBF Kernel)': 0.89
           };
           
           let baseAcc = baseAccuracies[modelName] || 0.85;
           
           // Adjust for split ratio: lower train ratio = lower validation accuracy (overfitting less data)
           // Higher train ratio = slightly better validation (more training examples)
           const splitFactor = Math.pow(splitRatio / 80, 0.2); // 80% is baseline, diminishing returns
           let validationAcc = baseAcc * splitFactor;
           let trainAcc = validationAcc + 0.015; // Small train/val gap
           
           return {
             train: Math.round(trainAcc * 100 * 10) / 10,
             validation: Math.round(validationAcc * 100 * 10) / 10
           };
         };

         // Build table data with model-specific accuracies
         let tableData: Array<{ model: string; train: number; validation: number; notes: string }> = [];

         if (pipeline.selectedModel === 'AutoML' && pipeline.comparisonResults.length > 0) {
           // Use comparisonResults from AutoML with dynamic split-adjusted accuracy
           const models = ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM (RBF Kernel)'];
           tableData = models.map(modelName => {
             const acc = getModelAccuracy(modelName, trainSplit);
             const notesMap: { [key: string]: string } = {
               'Logistic Regression': 'Fast, simple, good baseline',
               'Random Forest': 'High accuracy, risk of slight overfitting',
               'XGBoost': 'Very strong performance, best for tabular data',
               'SVM (RBF Kernel)': 'Good for high-dimensional data'
             };
             return {
               model: modelName,
               train: acc.train,
               validation: acc.validation,
               notes: notesMap[modelName]
             };
           });
         } else if (pipeline.selectedModel) {
           // Single model selected: use model-specific accuracy with split adjustment
           const acc = getModelAccuracy(pipeline.selectedModel, trainSplit);
           const notes = {
             'Logistic Regression': 'Fast, simple, good baseline',
             'Decision Tree': 'Interpretable, risk of overfitting',
             'Random Forest': 'High accuracy, risk of slight overfitting',
             'XGBoost': 'Very strong performance, best for tabular data',
             'SVM': 'Good for high-dimensional data'
           };
           tableData = [{
             model: pipeline.selectedModel,
             train: acc.train,
             validation: acc.validation,
             notes: notes[pipeline.selectedModel as keyof typeof notes] || 'Model performance metrics'
           }];
         }

         return (
            <div className={`${colors.card} rounded-3xl border ${colors.border} animate-slide-up min-h-[500px] overflow-hidden shadow-sm flex flex-col`}>
                 <div className={`px-6 py-4 border-b ${colors.border} flex justify-between items-center ${darkMode ? 'bg-white/5' : 'bg-gray-50/80'} backdrop-blur-md`}>
                     <div className="flex items-center space-x-2">
                        <div className="flex space-x-1.5">
                            <div className="w-3 h-3 rounded-full bg-[#ff5f57]"></div>
                            <div className="w-3 h-3 rounded-full bg-[#ff9f0a]"></div>
                            <div className="w-3 h-3 rounded-full bg-[#28c840]"></div>
                        </div>
                        <span className={`text-xs font-mono ml-2 ${colors.textSec}`}>ML Model Accuracy Comparison</span>
                    </div>
                 </div>
                 
                 <div className="overflow-x-auto flex-1">
                    <table className={`w-full text-sm text-left ${colors.text}`}>
                      <thead className={`text-xs uppercase ${colors.tableHeader}`}>
                        <tr>
                          <th className="px-6 py-4 font-semibold">Model</th>
                          <th className="px-6 py-4 font-semibold">Training Percentile</th>
                          <th className="px-6 py-4 font-semibold">Validation Percentile</th>
                          <th className="px-6 py-4 font-semibold">Notes</th>
                        </tr>
                      </thead>
                      <tbody className={`divide-y ${darkMode ? 'divide-white/5' : 'divide-gray-100'}`}>
                        {tableData.map((row, idx) => (
                          <tr key={idx} className={`${colors.tableRowHover} transition-colors`}>
                             <td className="px-6 py-4 font-semibold">{row.model}</td>
                             <td className="px-6 py-4">{row.train}</td>
                             <td className="px-6 py-4">{row.validation}</td>
                             <td className={`px-6 py-4 ${colors.textSec} text-sm`}>{row.notes}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                 </div>
            </div>
         );
      case 9: // Samples
        return (
          <div className={`flex flex-col animate-fade-in ${colors.card} rounded-3xl border ${colors.border} shadow-md overflow-hidden min-h-[500px]`}>
            <div className={`px-6 py-4 border-b ${colors.border} ${darkMode ? 'bg-white/5' : 'bg-gray-50/80'} backdrop-blur-xl flex justify-between items-center`}>
                <h3 className={`font-bold ${colors.text} text-sm`}>Inference Batch Preview</h3>
            </div>
            <div className="flex-1 overflow-auto">
              <table className={`w-full text-sm text-left ${colors.text}`}>
                <thead className={`text-xs uppercase ${darkMode ? 'bg-black/20' : 'bg-gray-100'} ${colors.textSec}`}>
                  <tr>
                    <th className="px-6 py-4 font-semibold">ID</th>
                    <th className="px-6 py-4 font-semibold">Actual</th>
                    <th className="px-6 py-4 font-semibold">Predicted</th>
                    <th className="px-6 py-4 font-semibold">Confidence</th>
                    <th className="px-6 py-4 font-semibold text-right">Status</th>
                  </tr>
                </thead>
                <tbody className={`divide-y ${darkMode ? 'divide-white/5' : 'divide-gray-100'}`}>
                  {pipeline.predictions.slice(0, 6).map((p) => (
                    <tr key={p.id} className={`${colors.tableRowHover} transition-colors`}>
                      <td className={`px-6 py-4 font-mono ${colors.textSec}`}>{p.id}</td>
                      <td className="px-6 py-4">
                          <span className={p.actual ? 'text-[#ff9f0a] font-medium' : 'text-[#0a84ff] font-medium'}>
                              {p.actual ? 'Road' : 'Other'}
                          </span>
                      </td>
                      <td className={`px-6 py-4 font-bold ${colors.text}`}>
                          {p.predicted ? 'Road' : 'Other'}
                      </td>
                      <td className="px-6 py-4">
                         <div className="flex items-center space-x-3">
                             <div className={`w-24 ${darkMode ? 'bg-[#2c2c2e]' : 'bg-gray-200'} h-1.5 rounded-full overflow-hidden`}>
                                 <div 
                                    className={`h-full ${p.prob > 0.8 ? 'bg-[#30d158]' : 'bg-[#ff9f0a]'}`} 
                                    style={{width: `${p.prob * 100}%`}}
                                 ></div>
                             </div>
                             <span className={`text-xs font-mono ${colors.textSec}`}>{(p.prob * 100).toFixed(1)}</span>
                         </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        {p.actual === p.predicted ? (
                            <CheckCircle size={16} className="text-[#30d158] inline-block" />
                        ) : (
                            <AlertCircle size={16} className="text-[#ff453a] inline-block" />
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      case 10: // Stats
        return (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-slide-up min-h-[500px]">
                <div className={`${colors.card} p-6 rounded-3xl border ${colors.border} flex flex-col justify-center shadow-lg`}>
                    <h3 className={`${colors.textSec} text-xs font-bold uppercase tracking-widest text-center mb-6`}>Confusion Matrix ({pipeline.metrics?.modelName})</h3>
                    {pipeline.confusionMatrix && <ConfusionMatrixHeatmap cm={pipeline.confusionMatrix} darkMode={darkMode} />}
                </div>
                <div className={`${colors.card} p-6 rounded-3xl border ${colors.border} flex flex-col justify-center shadow-lg`}>
                    <h3 className={`${colors.textSec} text-xs font-bold uppercase tracking-widest mb-6`}>Metrics</h3>
                    <div className="space-y-6">
                        {pipeline.metrics && Object.entries(pipeline.metrics).map(([key, value]) => (
                            (key !== 'auc' && key !== 'modelName') && (
                                <div key={key}>
                                    <div className={`flex justify-between text-xs ${colors.textSec} mb-2 uppercase tracking-wide font-semibold`}>
                                        <span>{key}</span>
                                        <span className={colors.text}>{(value as number * 100).toFixed(1)}</span>
                                    </div>
                                    <div className={`w-full ${colors.input} rounded-full h-2 overflow-hidden`}>
                                        <div 
                                            className={`${darkMode ? 'bg-white' : 'bg-black'} h-full rounded-full shadow-sm`} 
                                            style={{ width: `${(value as number) * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            )
                        ))}
                    </div>
                </div>
            </div>
        );
      case 11: // ROC
        return (
            <div className={`${colors.card} p-6 rounded-3xl border ${colors.border} animate-fade-in min-h-[500px] flex flex-col shadow-md`}>
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h3 className={`${colors.text} font-bold`}>ROC Curve</h3>
                        <p className={`text-xs ${colors.textSec}`}>Model Discrimination Ability</p>
                    </div>
                    <div className="bg-[#30d158]/10 px-4 py-2 rounded-xl border border-[#30d158]/20">
                        <span className="text-[#30d158] text-sm font-bold">AUC {pipeline.metrics?.auc.toFixed(3)}</span>
                    </div>
                </div>
                <div className="flex-1 min-h-0">
                    <RocCurveChart data={pipeline.rocCurve} darkMode={darkMode} />
                </div>
            </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className={`h-screen ${colors.bg} ${colors.text} font-sans selection:bg-[#0a84ff]/30 flex flex-col overflow-hidden transition-colors duration-500`}>
        {/* Glow backgrounds */}
        <div className="fixed top-0 left-0 w-full h-full pointer-events-none z-0 overflow-hidden">
            <div 
              className={`absolute rounded-full ${darkMode ? 'blur-[180px]' : 'blur-[120px] opacity-60'}`}
              style={{
                top: '-20%',
                left: '20%',
                width: '60%',
                height: '60%',
                backgroundColor: 'rgba(10, 132, 255, 0.1)'
              }}
            ></div>
            <div 
              className={`absolute rounded-full ${darkMode ? 'blur-[150px]' : 'blur-[100px] opacity-60'}`}
              style={{
                bottom: '-10%',
                right: '-10%',
                width: '50%',
                height: '50%',
                backgroundColor: 'rgba(191, 90, 242, 0.1)'
              }}
            ></div>
        </div>

      {/* Header */}
      <header className={`h-16 ${darkMode ? 'bg-black/50' : 'bg-white/50'} backdrop-blur-xl border-b ${colors.border} z-50 flex items-center justify-between px-6 sticky top-0 shrink-0`}>
          <div className="flex items-center space-x-3">
             <div className={`w-8 h-8 ${darkMode ? 'bg-white' : 'bg-black'} rounded-lg flex items-center justify-center shadow-lg`}>
                 <Layers className={`w-5 h-5 ${darkMode ? 'text-black' : 'text-white'}`} />
             </div>
             <h1 className={`font-semibold text-lg tracking-tight ${colors.text}`}>Road Extraction</h1>
          </div>
          
          <div className="flex items-center space-x-4">
               <button
                  onClick={toggleTheme}
                  className={`p-2 rounded-full transition-all ${colors.hover} ${colors.text}`}
                  aria-label="Toggle Theme"
               >
                   {darkMode ? <Sun size={20} /> : <Moon size={20} />}
               </button>

              <button 
                onClick={() => setShowCode(!showCode)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-full text-xs font-medium transition-all ${
                    showCode 
                    ? `${darkMode ? 'bg-white text-black' : 'bg-black text-white'} shadow-lg` 
                    : `${colors.input} ${colors.textSec} ${colors.hover}`
                }`}
              >
                  <Code size={14} />
                  <span>Code View</span>
              </button>
          </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative z-10">
          {/* Sidebar */}
          <div className={`w-72 ${darkMode ? 'bg-black/30' : 'bg-white/40'} backdrop-blur-xl border-r ${colors.border} hidden lg:flex flex-col shrink-0`}>
            <div className="p-4 flex-1 overflow-y-auto custom-scrollbar space-y-1">
                {STEPS.map((step) => {
                    const isActive = currentStep === step.id;
                    const isCompleted = currentStep > step.id;
                    return (
                    <button
                        key={step.id}
                        onClick={() => !loading && !autoMLRunning && setCurrentStep(step.id)}
                        disabled={loading || autoMLRunning}
                        className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-300 text-left group ${
                        isActive 
                            ? colors.activeStep 
                            : colors.hover
                        } ${isCompleted ? 'opacity-80 hover:opacity-100' : 'opacity-100'}`}
                    >
                        <div className={`flex-shrink-0 transition-colors ${isActive ? colors.text : isCompleted ? 'text-[#30d158]' : colors.textSec}`}>
                           {isCompleted ? <CheckCircle size={18} /> : getStepIcon(step.id)}
                        </div>
                        <div className="min-w-0">
                            <div className={`text-sm font-medium truncate ${isActive ? colors.text : colors.textSec}`}>
                                {step.title}
                            </div>
                        </div>
                    </button>
                    );
                })}
            </div>
          </div>

          {/* Main Workspace */}
          <main className={`flex-1 flex flex-col overflow-hidden relative ${darkMode ? 'bg-black/20' : 'bg-gray-50/50'}`}>
              <div 
                  className="flex-1 overflow-y-auto p-8 custom-scrollbar scroll-smooth pb-12" 
                  ref={scrollContainerRef}
              >
                  <div className="max-w-7xl mx-auto space-y-8 flex flex-col">
                      
                      {/* Step Header */}
                      <div className="flex items-start justify-between">
                          <div>
                              <div className="text-[#0a84ff] text-xs font-bold uppercase tracking-widest mb-2">Step {currentStep} of {STEPS.length}</div>
                              <h2 className={`text-3xl font-bold ${colors.text} tracking-tight`}>{STEPS[currentStep-1].title}</h2>
                              <p className={`${colors.textSec} mt-2 max-w-xl`}>{STEPS[currentStep-1].description}</p>
                          </div>
                      </div>

                      {/* Content Area */}
                      <div className={`flex flex-col lg:flex-row gap-8 min-h-0`}>
                          {/* Visualization */}
                          <div className={`transition-all duration-500 ease-out ${showCode ? 'lg:w-3/5' : 'w-full'} flex flex-col`}>
                               {renderContent()}
                          </div>

                          {/* Code Editor */}
                          {showCode && (
                              <div className="lg:w-2/5 animate-fade-in flex flex-col min-h-[400px]">
                                  <div className={`${colors.card} rounded-3xl border ${colors.border} shadow-2xl overflow-hidden flex flex-col h-full`}>
                                      <div className={`${colors.input} px-4 py-3 border-b ${colors.border} flex items-center justify-between`}>
                                          <div className="flex items-center space-x-2">
                                              <div className="flex space-x-1.5">
                                                  <div className="w-3 h-3 rounded-full bg-[#ff5f57]"></div>
                                                  <div className="w-3 h-3 rounded-full bg-[#febc2e]"></div>
                                                  <div className="w-3 h-3 rounded-full bg-[#28c840]"></div>
                                              </div>
                                              <div className={`ml-4 flex items-center space-x-2 text-xs ${colors.textSec} font-mono`}>
                                                  <Terminal size={12} />
                                                  <span>main.py</span>
                                              </div>
                                          </div>
                                          <div className={`text-[10px] ${colors.textSec} uppercase tracking-wider font-semibold`}>
                                              Python 3.11
                                          </div>
                                      </div>
                                      <div className="p-0 overflow-hidden flex-1 bg-[#282c34]">
                                          <div className="h-full overflow-y-auto custom-scrollbar p-4">
                                              <CodeBlock code={STEPS[currentStep-1].pythonSnippet} />
                                          </div>
                                      </div>
                                      {/* Editor Status Bar */}
                                      <div className="bg-[#21252b] px-4 py-1 flex justify-between items-center text-[10px] text-gray-500 font-mono border-t border-white/5">
                                          <div className="flex space-x-3">
                                              <span>UTF-8</span>
                                              <span>Ln {STEPS[currentStep-1].pythonSnippet.split('\n').length}, Col 1</span>
                                          </div>
                                          <div>Python</div>
                                      </div>
                                  </div>
                              </div>
                          )}
                      </div>

                      {/* AI Chat Card */}
                      <div className={`${colors.card} rounded-2xl border ${darkMode ? 'border-purple-500/30' : 'border-purple-200'} p-0 flex flex-col shadow-lg backdrop-blur-xl relative overflow-hidden group`}>
                            {/* Header */}
                            <div className={`p-3 border-b ${colors.border} flex justify-between items-center bg-gradient-to-r from-[#bf5af2]/10 to-transparent`}>
                                <div className="flex items-center space-x-2">
                                     <div className="p-1.5 bg-[#bf5af2] rounded-lg shadow-sm">
                                        <Zap className="w-3 h-3 text-white" />
                                     </div>
                                     <span className={`text-xs font-bold ${colors.text} uppercase tracking-wider`}>AI Assistant</span>
                                </div>
                                <div className="flex items-center space-x-4">
                                     {/* Manual API Key Trigger */}
                                    <button 
                                        onClick={() => setShowKeyInput(!showKeyInput)}
                                        className="text-[10px] text-gray-500 hover:text-gray-300 underline cursor-pointer"
                                    >
                                        Config API
                                    </button>
                                </div>
                            </div>
                            
                            {/* API Key Input */}
                             {showKeyInput && (
                                <div className={`p-3 border-b ${colors.border} animate-fade-in flex items-center space-x-2 bg-[#bf5af2]/5`}>
                                    <input 
                                        type="password" 
                                        placeholder="Paste API Key Here (Starts with AIza...)" 
                                        className={`flex-1 p-2 text-xs rounded-lg ${colors.input} ${colors.text} border ${colors.border} focus:outline-none focus:ring-1 focus:ring-[#bf5af2]`}
                                        value={manualApiKey}
                                        onChange={(e) => setManualApiKey(e.target.value)}
                                    />
                                    <button
                                        onClick={() => executeStep()}
                                        className={`px-3 py-2 rounded-lg text-xs font-bold text-white bg-[#bf5af2] hover:bg-[#a144eb] transition-colors`}
                                    >
                                        Retry
                                    </button>
                                </div>
                            )}

                            {/* Chat Content */}
                            <div className="p-4 flex flex-col space-y-4 max-h-[300px] overflow-y-auto custom-scrollbar">
                                {/* If not chatting yet, show the main insight as the first AI message */}
                                {!isChatting && (
                                    <div className="flex items-start space-x-3">
                                        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-[#bf5af2] to-[#5e5ce6] shrink-0 flex items-center justify-center text-[10px] text-white font-bold">AI</div>
                                        <div className={`bg-[#bf5af2]/10 p-3 rounded-2xl rounded-tl-none ${colors.text} text-sm leading-relaxed`}>
                                            {aiLoading ? (
                                                <span className={`${colors.textSec} animate-pulse`}>Analyzing pipeline context...</span>
                                            ) : (
                                                aiInsight || "Ready to analyze step..."
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* History */}
                                {chatMessages.map((msg, i) => (
                                     <div key={i} className={`flex items-start space-x-3 ${msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                                        <div className={`w-6 h-6 rounded-full shrink-0 flex items-center justify-center text-[10px] text-white font-bold ${msg.role === 'user' ? 'bg-gray-500' : 'bg-gradient-to-br from-[#bf5af2] to-[#5e5ce6]'}`}>
                                            {msg.role === 'user' ? 'U' : 'AI'}
                                        </div>
                                        <div className={`p-3 rounded-2xl text-sm leading-relaxed max-w-[80%] ${msg.role === 'user' ? `bg-[#2c2c2e] text-white rounded-tr-none` : `bg-[#bf5af2]/10 ${colors.text} rounded-tl-none`}`}>
                                            {msg.text}
                                        </div>
                                     </div>
                                ))}
                                <div ref={chatEndRef} />
                            </div>
                            
                            {/* Input Area */}
                            <div className={`p-3 border-t ${colors.border} flex items-center space-x-2 ${colors.input}`}>
                                <div className="flex-1 relative">
                                    <input 
                                        type="text" 
                                        value={chatInput}
                                        onChange={(e) => setChatInput(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                                        placeholder="Ask a question about this step..."
                                        className={`w-full bg-transparent ${colors.text} text-sm placeholder-gray-500 focus:outline-none pl-2 pr-8`}
                                        disabled={aiLoading}
                                    />
                                </div>
                                <button 
                                    onClick={handleSendMessage}
                                    disabled={!chatInput.trim() || aiLoading}
                                    className={`p-2 rounded-full ${chatInput.trim() ? 'bg-[#bf5af2] text-white' : 'bg-gray-700 text-gray-500'} transition-colors`}
                                >
                                    <Send size={14} />
                                </button>
                            </div>
                       </div>
                       
                       {/* INLINE NAVIGATION CONTROLS */}
                       <div className="flex items-center justify-between pt-6 border-t border-gray-100 dark:border-white/5">
                            <button 
                                onClick={handlePrev} 
                                disabled={currentStep === 1 || loading || autoMLRunning}
                                className={`flex items-center space-x-2 px-6 py-3 rounded-full text-sm font-medium transition-all ${
                                    currentStep === 1 
                                    ? 'opacity-0 cursor-default' 
                                    : `${colors.input} ${colors.text} hover:opacity-80`
                                }`}
                            >
                                <ArrowLeft size={16} />
                                <span>Previous Step</span>
                            </button>

                            <button 
                                onClick={handleNext} 
                                disabled={currentStep === STEPS.length || loading || autoMLRunning}
                                className={`flex items-center space-x-2 px-8 py-3 rounded-full text-sm font-semibold shadow-lg transition-all ${
                                    currentStep === STEPS.length
                                    ? `${colors.input} ${colors.textSec} cursor-not-allowed`
                                    : `${darkMode ? 'bg-white text-black' : 'bg-black text-white'} hover:scale-105`
                                }`}
                            >
                                <span>{currentStep === STEPS.length ? 'Completed' : 'Next Step'}</span>
                                <ArrowRight size={16} />
                            </button>
                       </div>
                  </div>
              </div>
          </main>
      </div>
    </div>
  );
}
