
export type DiseaseClass =
  | "Cataract"
  | "Diabetic Retinopathy"
  | "Glaucoma"
  | "Normal";

export interface User {
  id: string;
  name: string;
  email: string;
  role: "user" | "admin";
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  role: "user" | "admin";
  user_id: string;
  name: string;
}

export interface PredictionResponse {
  prediction_id: string;
  prediction: DiseaseClass;
  confidence: number;
  probabilities: Record<string, number>;
  model: string;
  model_version: string;
  gradcam_available: boolean;
  gradcam_url: string | null;
  disclaimer: string;
}

export interface PredictionHistoryItem {
  id: string;
  predicted_class: DiseaseClass;
  confidence: number;
  model_name: string;
  image_path: string;
  created_at: string;
}

export interface PredictionDetail extends PredictionHistoryItem {
  probabilities: Record<string, number>;
  gradcam_path: string | null;
}

export interface ModelComparisonRow {
  model: string;
  accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  specificity_macro: number;
  roc_auc_macro: number | null;
  balanced_accuracy: number;
  inference_time_ms: number | null;
  training_time_seconds: number;
  n_parameters: number | null;
  model_size_kb: number | null;
  selection_score: number;
}
export interface EnsembleMetrics {
  ResNet_weight: number;
  EfficientNet_weight: number;
  accuracy: number;
  macro_f1: number;
  balanced_accuracy: number;
  macro_recall: number;
  macro_roc_auc: number;
  selection_score: number;
}
export interface ModelComparisonReport {
  selection_criterion: string;
  comparison_table: ModelComparisonRow[];
  best_model: string;
  best_model_score: number;
  production_model?: string;
  production_model_score?: number;
}
export interface ModelComparisonReport {
  selection_criterion: string;
  comparison_table: ModelComparisonRow[];
  best_model: string;
  best_model_score: number;
  ensemble?: EnsembleMetrics;
  production_model?: string;
  production_model_score?: number;
}
export interface DatasetStatistics {
  total_images: number;
  images_per_class: Record<string, number>;
  class_percentages: Record<string, number>;
  train_images: number;
  val_images: number;
  test_images: number;
  train_per_class: Record<string, number>;
  val_per_class: Record<string, number>;
  test_per_class: Record<string, number>;
}

export interface AdminOverview {
  total_users: number;
  total_predictions: number;
  disease_distribution: Record<string, number>;
  model_usage: Record<string, number>;
  recent_activity: {
    prediction_id: string;
    predicted_class: string;
    confidence: number;
    model_name: string;
    created_at: string;
  }[];
}
