import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import type { ModelComparisonRow } from "../../types";

export function MetricBarChart({ rows, metricKey, label, color }: {
  rows: ModelComparisonRow[]; metricKey: keyof ModelComparisonRow; label: string; color: string;
}) {
  const data = rows.map((r) => ({ name: r.model, value: (r[metricKey] as number) ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ bottom: 40 }}>
        <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
        <Tooltip />
        <Legend />
        <Bar dataKey="value" name={label} fill={color} radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
