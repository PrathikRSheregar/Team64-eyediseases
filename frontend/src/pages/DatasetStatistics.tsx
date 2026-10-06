import { useEffect, useState } from "react";
import { Database } from "lucide-react";
import { api } from "../services/api";
import { StatCard } from "../components/ui/StatCard";
import { ClassDistributionChart } from "../components/charts/ClassDistributionChart";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import type { DatasetStatistics as DatasetStatisticsType } from "../types";

export default function DatasetStatistics() {
  const [stats, setStats] = useState<DatasetStatisticsType | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get<DatasetStatisticsType>("/dataset/statistics")
      .then((res) => setStats(res.data))
      .catch((err) => {
        if (err.message?.includes("No dataset statistics")) setNotFound(true);
        else setError(err.message);
      });
  }, []);

  if (error) return <div className="card p-6 text-sm text-red-500">{error}</div>;

  if (notFound) {
    return (
      <EmptyState
        title="No dataset statistics yet"
        description="Place your images under dataset/<class_name>/ and run `python -m ml.data.analyze_dataset`."
        icon={<Database size={32} />}
      />
    );
  }

  return (
    <div>
      <h1 className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">Dataset Statistics</h1>
      <p className="mt-1 text-sm text-slate-400">Size, class balance, and train/val/test split of the training dataset.</p>

      {!stats ? (
        <div className="mt-6 card p-6"><Skeleton className="h-64 w-full" /></div>
      ) : (
        <>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Total images" value={stats.total_images.toLocaleString()} />
            <StatCard label="Training images" value={stats.train_images.toLocaleString()} />
            <StatCard label="Validation images" value={stats.val_images.toLocaleString()} />
            <StatCard label="Test images" value={stats.test_images.toLocaleString()} />
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="card p-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Class distribution</p>
              <ClassDistributionChart data={stats.images_per_class} />
            </div>
            <div className="card p-5">
              <p className="mb-3 text-sm font-semibold text-slate-600 dark:text-slate-300">Class percentages</p>
              <div className="space-y-3">
                {Object.entries(stats.class_percentages).map(([name, pct]) => (
                  <div key={name}>
                    <div className="mb-1 flex justify-between text-sm"><span>{name}</span><span className="text-slate-400">{pct}%</span></div>
                    <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-800">
                      <div className="h-2 rounded-full bg-brand-500" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card mt-6 overflow-x-auto p-4">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-400">
                <tr><th className="px-3 py-2">Class</th><th className="px-3 py-2">Train</th><th className="px-3 py-2">Validation</th><th className="px-3 py-2">Test</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {Object.keys(stats.images_per_class).map((cls) => (
                  <tr key={cls}>
                    <td className="px-3 py-2 font-medium text-slate-700 dark:text-slate-200">{cls}</td>
                    <td className="px-3 py-2">{stats.train_per_class[cls]}</td>
                    <td className="px-3 py-2">{stats.val_per_class[cls]}</td>
                    <td className="px-3 py-2">{stats.test_per_class[cls]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
