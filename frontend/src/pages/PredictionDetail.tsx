import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { api } from "../services/api";
import { ProbabilityBarChart } from "../components/charts/ProbabilityBarChart";
import { Disclaimer } from "../components/ui/Disclaimer";
import { Skeleton } from "../components/ui/Skeleton";
import { useAuthenticatedImage } from "../hooks/useAuthenticatedImage";
import type { PredictionDetail as PredictionDetailType } from "../types";

export default function PredictionDetail() {
  const { id } = useParams<{ id: string }>();
  const [detail, setDetail] = useState<PredictionDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    api.get<PredictionDetailType>(`/predictions/${id}`)
      .then((res) => setDetail(res.data))
      .catch((err) => setError(err.message));
  }, [id]);

  const originalImage = useAuthenticatedImage(id ? `/predictions/${id}/image` : null);
  const gradcamImage = useAuthenticatedImage(detail?.gradcam_path ? `/predictions/${id}/gradcam` : null);

  if (error) return <div className="card p-6 text-sm text-red-500">{error}</div>;

  return (
    <div>
      <Link to="/history" className="mb-4 inline-flex items-center gap-1 text-sm font-medium text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
        <ArrowLeft size={16} /> Back to history
      </Link>

      {!detail ? (
        <div className="card p-6"><Skeleton className="h-64 w-full" /></div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="card p-6">
            <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Original image</p>
            {originalImage.loading && <Skeleton className="h-72 w-full" />}
            {originalImage.src && <img src={originalImage.src} alt="Uploaded eye image" className="w-full rounded-xl border border-slate-200 dark:border-slate-700" />}

            {detail.gradcam_path && (
              <div className="mt-5">
                <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Grad-CAM explanation</p>
                {gradcamImage.loading && <Skeleton className="h-72 w-full" />}
                {gradcamImage.src && <img src={gradcamImage.src} alt="Grad-CAM heatmap" className="w-full rounded-xl border border-slate-200 dark:border-slate-700" />}
              </div>
            )}
          </div>

          <div className="card p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Prediction</p>
                <p className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">{detail.predicted_class}</p>
              </div>
              <span className="badge bg-brand-50 text-lg font-bold text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">
                {Math.round(detail.confidence * 100)}%
              </span>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div><dt className="text-slate-400">Model</dt><dd className="font-medium text-slate-700 dark:text-slate-200">{detail.model_name}</dd></div>
              <div><dt className="text-slate-400">Date</dt><dd className="font-medium text-slate-700 dark:text-slate-200">{new Date(detail.created_at).toLocaleString()}</dd></div>
            </dl>

            <div className="mt-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Class probabilities</p>
              <ProbabilityBarChart probabilities={detail.probabilities} />
            </div>

            <div className="mt-5"><Disclaimer compact /></div>
          </div>
        </div>
      )}
    </div>
  );
}
