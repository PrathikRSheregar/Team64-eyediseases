import { useEffect, useState } from "react";
import { Users, Images, Cpu, Activity } from "lucide-react";
import { api } from "../services/api";
import { StatCard } from "../components/ui/StatCard";
import { ClassDistributionChart } from "../components/charts/ClassDistributionChart";
import { Skeleton } from "../components/ui/Skeleton";
import type { AdminOverview } from "../types";

export default function AdminDashboard() {
  const [data, setData] = useState<AdminOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get<AdminOverview>("/admin/overview")
      .then((res) => setData(res.data))
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="card p-6 text-sm text-red-500">{error}</div>;

  return (
    <div>
      <h1 className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">Admin Dashboard</h1>
      <p className="mt-1 text-sm text-slate-400">Platform-wide usage and model activity.</p>

      {!data ? (
        <div className="mt-6 card p-6"><Skeleton className="h-64 w-full" /></div>
      ) : (
        <>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Total users" value={data.total_users} icon={<Users size={20} />} />
            <StatCard label="Total predictions" value={data.total_predictions} icon={<Images size={20} />} />
            <StatCard label="Models in use" value={Object.keys(data.model_usage).length} icon={<Cpu size={20} />} />
            <StatCard label="Recent activity" value={data.recent_activity.length} icon={<Activity size={20} />} />
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="card p-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">Disease distribution</p>
              {Object.keys(data.disease_distribution).length > 0
                ? <ClassDistributionChart data={data.disease_distribution} />
                : <p className="py-10 text-center text-sm text-slate-400">No predictions yet.</p>}
            </div>
            <div className="card p-5">
              <p className="mb-3 text-sm font-semibold text-slate-600 dark:text-slate-300">Model usage</p>
              <div className="space-y-2">
                {Object.entries(data.model_usage).map(([name, count]) => (
                  <div key={name} className="flex items-center justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-300">{name}</span>
                    <span className="font-semibold text-slate-800 dark:text-slate-100">{count}</span>
                  </div>
                ))}
                {Object.keys(data.model_usage).length === 0 && <p className="text-sm text-slate-400">No predictions yet.</p>}
              </div>
            </div>
          </div>

          <div className="card mt-6 overflow-hidden">
            <p className="border-b border-slate-100 px-5 py-3 text-sm font-semibold text-slate-600 dark:border-slate-800 dark:text-slate-300">Recent activity</p>
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-400">
                <tr><th className="px-5 py-2">Prediction</th><th className="px-5 py-2">Confidence</th><th className="px-5 py-2">Model</th><th className="px-5 py-2">Date</th></tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {data.recent_activity.map((row) => (
                  <tr key={row.prediction_id}>
                    <td className="px-5 py-2 font-medium text-slate-700 dark:text-slate-200">{row.predicted_class}</td>
                    <td className="px-5 py-2">{Math.round(row.confidence * 100)}%</td>
                    <td className="px-5 py-2 text-slate-500">{row.model_name}</td>
                    <td className="px-5 py-2 text-slate-400">{new Date(row.created_at).toLocaleString()}</td>
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
