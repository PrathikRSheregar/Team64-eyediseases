import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ScanEye, Images, TrendingUp, Cpu } from "lucide-react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { StatCard } from "../components/ui/StatCard";
import { Skeleton } from "../components/ui/Skeleton";
import { EmptyState } from "../components/ui/EmptyState";
import type { PredictionHistoryItem } from "../types";

export default function Dashboard() {
  const { user } = useAuth();
  const [predictions, setPredictions] = useState<PredictionHistoryItem[] | null>(null);
  const [productionModel, setProductionModel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get<PredictionHistoryItem[]>("/predictions?limit=5")
      .then((res) => setPredictions(res.data))
      .catch((err) => setError(err.message));

    api.get("/models")
      .then((res) => setProductionModel(res.data.production_model))
      .catch(() => setProductionModel(null));
  }, []);

  const mostFrequent = predictions && predictions.length > 0
    ? Object.entries(
        predictions.reduce<Record<string, number>>((acc, p) => {
          acc[p.predicted_class] = (acc[p.predicted_class] || 0) + 1;
          return acc;
        }, {})
      ).sort((a, b) => b[1] - a[1])[0][0]
    : "—";

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">Welcome back, {user?.name?.split(" ")[0]}</h1>
          <p className="text-sm text-slate-400">Here's a snapshot of your recent activity.</p>
        </div>
        <Link to="/predict" className="btn-primary"><ScanEye size={16} /> New prediction</Link>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Total predictions" value={predictions ? predictions.length : <Skeleton className="h-7 w-10" />} icon={<Images size={20} />} />
        <StatCard label="Most frequent class" value={predictions ? mostFrequent : <Skeleton className="h-7 w-24" />} icon={<TrendingUp size={20} />} />
        <StatCard label="Active model" value={productionModel ?? "Not trained yet"} icon={<Cpu size={20} />} hint={productionModel ? "Selected by macro F1 + balanced accuracy" : "Run the training pipeline"} />
        <StatCard label="Account role" value={<span className="capitalize">{user?.role}</span>} icon={<ScanEye size={20} />} />
      </div>

      <div className="mt-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-display text-lg font-semibold text-slate-800 dark:text-slate-100">Recent predictions</h2>
          <Link to="/history" className="text-sm font-medium text-brand-600">View all →</Link>
        </div>

        {predictions === null && !error && (
          <div className="card divide-y divide-slate-100 dark:divide-slate-800">
            {[...Array(3)].map((_, i) => <div key={i} className="p-4"><Skeleton className="h-6 w-full" /></div>)}
          </div>
        )}

        {error && <div className="card p-6 text-sm text-red-500">Could not load recent predictions: {error}</div>}

        {predictions && predictions.length === 0 && (
          <EmptyState
            title="No predictions yet"
            description="Upload your first eye image to get an AI-assisted classification."
            icon={<ScanEye size={32} />}
            action={<Link to="/predict" className="btn-primary mt-2">Make a prediction</Link>}
          />
        )}

        {predictions && predictions.length > 0 && (
          <div className="card divide-y divide-slate-100 dark:divide-slate-800">
            {predictions.map((p) => (
              <Link key={p.id} to={`/history/${p.id}`} className="flex items-center justify-between p-4 text-sm hover:bg-slate-50 dark:hover:bg-slate-800/60">
                <div>
                  <p className="font-medium text-slate-700 dark:text-slate-200">{p.predicted_class}</p>
                  <p className="text-xs text-slate-400">{p.model_name} • {new Date(p.created_at).toLocaleString()}</p>
                </div>
                <span className="badge bg-brand-50 text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">{Math.round(p.confidence * 100)}%</span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
