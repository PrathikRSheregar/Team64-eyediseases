import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { History as HistoryIcon, Filter } from "lucide-react";
import { api } from "../services/api";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import type { PredictionHistoryItem } from "../types";

const CLASS_OPTIONS = ["", "Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"];

export default function History() {
  const [items, setItems] = useState<PredictionHistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [disease, setDisease] = useState("");
  const [model, setModel] = useState("");

  useEffect(() => {
    setItems(null);
    const params = new URLSearchParams();
    if (disease) params.set("disease", disease);
    if (model) params.set("model_name", model);
    api.get<PredictionHistoryItem[]>(`/predictions?${params.toString()}`)
      .then((res) => setItems(res.data))
      .catch((err) => setError(err.message));
  }, [disease, model]);

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">Prediction History</h1>
          <p className="text-sm text-slate-400">Every image you've analyzed, with filters.</p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <Filter size={16} className="text-slate-400" />
          <select value={disease} onChange={(e) => setDisease(e.target.value)} className="input-field w-auto">
            {CLASS_OPTIONS.map((c) => <option key={c} value={c}>{c || "All classes"}</option>)}
          </select>
          <input
            placeholder="Filter by model…"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="input-field w-auto"
          />
        </div>
      </div>

      {items === null && !error && (
        <div className="card divide-y divide-slate-100 dark:divide-slate-800">
          {[...Array(4)].map((_, i) => <div key={i} className="p-4"><Skeleton className="h-6 w-full" /></div>)}
        </div>
      )}

      {error && <div className="card p-6 text-sm text-red-500">Could not load history: {error}</div>}

      {items && items.length === 0 && (
        <EmptyState title="No matching predictions" description="Try clearing your filters or make a new prediction." icon={<HistoryIcon size={32} />} />
      )}

      {items && items.length > 0 && (
        <div className="card overflow-hidden">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400 dark:bg-slate-800/60">
              <tr>
                <th className="px-4 py-3">Prediction</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Date</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {items.map((p) => (
                <tr key={p.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/60">
                  <td className="px-4 py-3 font-medium text-slate-700 dark:text-slate-200">{p.predicted_class}</td>
                  <td className="px-4 py-3">{Math.round(p.confidence * 100)}%</td>
                  <td className="px-4 py-3 text-slate-500">{p.model_name}</td>
                  <td className="px-4 py-3 text-slate-400">{new Date(p.created_at).toLocaleString()}</td>
                  <td className="px-4 py-3 text-right">
                    <Link to={`/history/${p.id}`} className="font-medium text-brand-600">View details →</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
