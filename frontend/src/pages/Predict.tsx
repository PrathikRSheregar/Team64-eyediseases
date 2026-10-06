import { useState } from "react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import { Loader2, Sparkles } from "lucide-react";
import { api } from "../services/api";
import { ImageDropzone } from "../components/upload/ImageDropzone";
import { ProbabilityBarChart } from "../components/charts/ProbabilityBarChart";
import { Disclaimer } from "../components/ui/Disclaimer";
import { useAuthenticatedImage } from "../hooks/useAuthenticatedImage";
import type { PredictionResponse } from "../types";

export default function Predict() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const navigate = useNavigate();
  const gradcam = useAuthenticatedImage(result?.gradcam_url ?? null);

  async function handlePredict() {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await api.post<PredictionResponse>("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
      toast.success("Prediction complete.");
    } catch (err: any) {
      toast.error(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1 className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">New Prediction</h1>
      <p className="mt-1 text-sm text-slate-400">Upload an eye or retinal image to classify it.</p>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <ImageDropzone onFileSelected={setFile} />
          <button onClick={handlePredict} disabled={!file || loading} className="btn-primary mt-5 w-full">
            {loading ? <><Loader2 className="animate-spin" size={16} /> Analyzing image…</> : <><Sparkles size={16} /> Predict</>}
          </button>
          <div className="mt-4"><Disclaimer compact /></div>
        </div>

        <div className="card p-6">
          {!result && !loading && (
            <div className="flex h-full min-h-[280px] items-center justify-center text-center text-sm text-slate-400">
              Prediction results will appear here.
            </div>
          )}
          {loading && (
            <div className="flex h-full min-h-[280px] flex-col items-center justify-center gap-3 text-slate-400">
              <Loader2 className="animate-spin" size={28} />
              <p className="text-sm">Running preprocessing and inference…</p>
            </div>
          )}
          {result && (
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Prediction</p>
                  <p className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">{result.prediction}</p>
                </div>
                <span className="badge bg-brand-50 text-lg font-bold text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">
                  {Math.round(result.confidence * 100)}%
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-400">Model used: {result.model}</p>

              <div className="mt-5">
                <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Class probabilities</p>
                <ProbabilityBarChart probabilities={result.probabilities} />
              </div>

              {result.gradcam_available && result.gradcam_url && (
                <div className="mt-5">
                  <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Grad-CAM explanation</p>
                  {gradcam.loading && <div className="h-40 animate-pulse rounded-xl bg-slate-100 dark:bg-slate-800" />}
                  {gradcam.src && (
                    <img
                      src={gradcam.src}
                      alt="Grad-CAM heatmap overlay"
                      className="w-full rounded-xl border border-slate-200 dark:border-slate-700"
                    />
                  )}
                  <p className="mt-1 text-xs text-slate-400">
                    Highlighted regions influenced the model's decision most — this is not a definitive medical finding.
                  </p>
                </div>
              )}

              <div className="mt-5">
                <Disclaimer compact />
              </div>

              <button onClick={() => navigate(`/history/${result.prediction_id}`)} className="btn-secondary mt-4 w-full">
                View full details
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
