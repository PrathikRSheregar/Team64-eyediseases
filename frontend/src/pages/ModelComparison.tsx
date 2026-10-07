import { useEffect, useState } from "react";
import { Trophy, Layers } from "lucide-react";
import { api } from "../services/api";
import { MetricBarChart } from "../components/charts/MetricBarChart";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import type { ModelComparisonReport } from "../types";

export default function ModelComparison() {
  const [report, setReport] = useState<ModelComparisonReport | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<ModelComparisonReport>("/model-performance")
      .then((res) => {
        console.log("Model comparison response:", res.data);
        setReport(res.data);
      })
      .catch((err) => {
        console.error("Model comparison error:", err);

        if (err.message?.includes("No model comparison report")) {
          setNotFound(true);
        } else {
          setError(err.message);
        }
      });
  }, []);

  if (error) {
    return (
      <div className="card p-6 text-sm text-red-500">
        {error}
      </div>
    );
  }

  if (notFound) {
    return (
      <EmptyState
        title="Models haven't been trained yet"
        description="Run `python -m ml.training.train_all` on your dataset to train every model and generate this comparison."
        icon={<Trophy size={32} />}
      />
    );
  }

  return (
    <div>
      <h1 className="font-display text-2xl font-bold text-slate-800 dark:text-slate-100">
        Model Comparison
      </h1>

      <p className="mt-1 text-sm text-slate-400">
        Every trained model evaluated on the same held-out test set.
      </p>

      {!report ? (
        <div className="mt-6 card p-6">
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <>
          {/* Best Model */}
          <div className="card mt-6 flex items-center gap-3 p-5">
            <Trophy className="text-amber-500" size={22} />

            <div>
              <p className="font-semibold text-slate-800 dark:text-slate-100">
                Best model: {report.best_model}{" "}
                <span className="font-normal text-slate-400">
                  (score {report.best_model_score.toFixed(3)})
                </span>
              </p>

              <p className="text-xs text-slate-400">
                {report.selection_criterion}
              </p>
            </div>
          </div>

          {/* ResNet + EfficientNet Ensemble */}
          {report.ensemble && (
            <div className="card mt-6 p-5">
              <div className="flex items-center gap-3">
                <Layers className="text-blue-500" size={22} />

                <div>
                  <h2 className="font-semibold text-slate-800 dark:text-slate-100">
                    ResNet + EfficientNet Ensemble
                  </h2>

                  <p className="text-xs text-slate-400">
                    Weighted ensemble evaluated on the same held-out test set.
                  </p>
                </div>
              </div>

              <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-lg bg-slate-50 p-4 dark:bg-slate-800">
                  <p className="text-xs text-slate-400">
                    ResNet Weight
                  </p>

                  <p className="mt-1 text-xl font-semibold text-slate-800 dark:text-slate-100">
                    {(report.ensemble.ResNet_weight * 100).toFixed(0)}%
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 p-4 dark:bg-slate-800">
                  <p className="text-xs text-slate-400">
                    EfficientNet Weight
                  </p>

                  <p className="mt-1 text-xl font-semibold text-slate-800 dark:text-slate-100">
                    {(report.ensemble.EfficientNet_weight * 100).toFixed(0)}%
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 p-4 dark:bg-slate-800">
                  <p className="text-xs text-slate-400">
                    Accuracy
                  </p>

                  <p className="mt-1 text-xl font-semibold text-slate-800 dark:text-slate-100">
                    {(report.ensemble.accuracy * 100).toFixed(2)}%
                  </p>
                </div>

                <div className="rounded-lg bg-amber-50 p-4 dark:bg-amber-950/20">
                  <p className="text-xs text-slate-400">
                    Selection Score
                  </p>

                  <p className="mt-1 text-xl font-semibold text-amber-600">
                    {report.ensemble.selection_score.toFixed(3)}
                  </p>
                </div>
              </div>

              <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <div>
                  <p className="text-xs text-slate-400">
                    Macro F1
                  </p>

                  <p className="font-semibold">
                    {(report.ensemble.macro_f1 * 100).toFixed(2)}%
                  </p>
                </div>

                <div>
                  <p className="text-xs text-slate-400">
                    Balanced Accuracy
                  </p>

                  <p className="font-semibold">
                    {(report.ensemble.balanced_accuracy * 100).toFixed(2)}%
                  </p>
                </div>

                <div>
                  <p className="text-xs text-slate-400">
                    Macro Recall
                  </p>

                  <p className="font-semibold">
                    {(report.ensemble.macro_recall * 100).toFixed(2)}%
                  </p>
                </div>

                <div>
                  <p className="text-xs text-slate-400">
                    ROC-AUC
                  </p>

                  <p className="font-semibold">
                    {(report.ensemble.macro_roc_auc * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Model Comparison Table */}
          <div className="card mt-6 overflow-x-auto p-4">
            <table className="w-full min-w-[900px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-400">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2">Accuracy</th>
                  <th className="px-3 py-2">Precision</th>
                  <th className="px-3 py-2">Recall</th>
                  <th className="px-3 py-2">F1</th>
                  <th className="px-3 py-2">ROC-AUC</th>
                  <th className="px-3 py-2">Inference (ms)</th>
                  <th className="px-3 py-2">Params</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {report.comparison_table.map((row) => {
                  const isHybrid =
                    row.model.includes("Hybrid ResNet");

                  return (
                    <tr
                      key={row.model}
                      className={
                        row.model === report.best_model || isHybrid
                          ? "bg-amber-50 dark:bg-amber-950/20"
                          : ""
                      }
                    >
                      <td className="px-3 py-2 font-medium text-slate-700 dark:text-slate-200">
                        {row.model}

                        {isHybrid && (
                          <span className="ml-2 rounded-full bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                            Hybrid
                          </span>
                        )}
                      </td>

                      <td className="px-3 py-2 font-semibold">
                        {row.accuracy != null
                          ? `${(row.accuracy * 100).toFixed(2)}%`
                          : "—"}
                      </td>

                      <td className="px-3 py-2">
                        {row.precision_macro != null
                          ? row.precision_macro.toFixed(3)
                          : "—"}
                      </td>

                      <td className="px-3 py-2">
                        {row.recall_macro != null
                          ? row.recall_macro.toFixed(3)
                          : "—"}
                      </td>

                      <td className="px-3 py-2">
                        {row.f1_macro != null
                          ? row.f1_macro.toFixed(3)
                          : "—"}
                      </td>

                      <td className="px-3 py-2">
                        {row.roc_auc_macro != null
                          ? row.roc_auc_macro.toFixed(3)
                          : "—"}
                      </td>

                      <td className="px-3 py-2">
                        {row.inference_time_ms != null
                          ? row.inference_time_ms.toFixed(2)
                          : "—"}
                      </td>

                      <td className="px-3 py-2">
                        {row.n_parameters != null
                          ? row.n_parameters.toLocaleString()
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Hybrid Model Highlight */}
          {(() => {
            const hybrid = report.comparison_table.find((row) =>
              row.model.includes("Hybrid ResNet")
            );

            if (!hybrid) {
              return null;
            }

            return (
              <div className="card mt-6 border border-amber-200 bg-amber-50 p-5 dark:border-amber-900/40 dark:bg-amber-950/20">
                <div className="flex items-center gap-3">
                  <Trophy
                    className="text-amber-500"
                    size={24}
                  />

                  <div>
                    <h2 className="font-semibold text-slate-800 dark:text-slate-100">
                      Best Hybrid Model
                    </h2>

                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      ResNet deep features + traditional image features +
                      RBF SVM
                    </p>
                  </div>
                </div>

                <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="rounded-lg bg-white p-4 dark:bg-slate-800">
                    <p className="text-xs text-slate-400">
                      Accuracy
                    </p>

                    <p className="mt-1 text-2xl font-bold text-slate-800 dark:text-slate-100">
                      {hybrid.accuracy != null
                        ? `${(hybrid.accuracy * 100).toFixed(2)}%`
                        : "—"}
                    </p>
                  </div>

                  <div className="rounded-lg bg-white p-4 dark:bg-slate-800">
                    <p className="text-xs text-slate-400">
                      Balanced Accuracy
                    </p>

                    <p className="mt-1 text-2xl font-bold text-slate-800 dark:text-slate-100">
                      {hybrid.balanced_accuracy != null
                        ? `${(
                            hybrid.balanced_accuracy * 100
                          ).toFixed(2)}%`
                        : "—"}
                    </p>
                  </div>

                  <div className="rounded-lg bg-white p-4 dark:bg-slate-800">
                    <p className="text-xs text-slate-400">
                      Macro F1
                    </p>

                    <p className="mt-1 text-2xl font-bold text-slate-800 dark:text-slate-100">
                      {hybrid.f1_macro != null
                        ? `${(
                            hybrid.f1_macro * 100
                          ).toFixed(2)}%`
                        : "—"}
                    </p>
                  </div>

                  <div className="rounded-lg bg-white p-4 dark:bg-slate-800">
                    <p className="text-xs text-slate-400">
                      Training Time
                    </p>

                    <p className="mt-1 text-2xl font-bold text-slate-800 dark:text-slate-100">
                      {hybrid.training_time_seconds != null
                        ? `${hybrid.training_time_seconds.toFixed(
                            1
                          )}s`
                        : "—"}
                    </p>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Charts */}
          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="card p-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">
                Accuracy
              </p>

              <MetricBarChart
                rows={report.comparison_table}
                metricKey="accuracy"
                label="Accuracy"
                color="#2f63f7"
              />
            </div>

            <div className="card p-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">
                Precision (macro)
              </p>

              <MetricBarChart
                rows={report.comparison_table}
                metricKey="precision_macro"
                label="Precision"
                color="#0f9b8e"
              />
            </div>

            <div className="card p-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">
                Recall (macro)
              </p>

              <MetricBarChart
                rows={report.comparison_table}
                metricKey="recall_macro"
                label="Recall"
                color="#e8a33d"
              />
            </div>

            <div className="card p-5">
              <p className="mb-2 text-sm font-semibold text-slate-600 dark:text-slate-300">
                F1-score (macro)
              </p>

              <MetricBarChart
                rows={report.comparison_table}
                metricKey="f1_macro"
                label="F1"
                color="#e8615a"
              />
            </div>
          </div>

          <p className="mt-6 text-xs text-slate-400">
            Confusion matrices and training curves for each model are
            generated as PNG files under{" "}
            <code className="mx-1 rounded bg-slate-100 px-1 dark:bg-slate-800">
              ml/artifacts/plots/
            </code>{" "}
            during training.
          </p>
        </>
      )}
    </div>
  );
}