import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const COLORS: Record<string, string> = {
  Cataract: "#2f63f7",
  "Diabetic Retinopathy": "#e8615a",
  Glaucoma: "#e8a33d",
  Normal: "#0f9b8e",
};

export function ProbabilityBarChart({ probabilities }: { probabilities: Record<string, number> }) {
  const data = Object.entries(probabilities).map(([name, value]) => ({
    name, value: Math.round(value * 1000) / 10,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24 }}>
        <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 12 }} />
        <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v: number) => [`${v}%`, "Probability"]} />
        <Bar dataKey="value" radius={[0, 6, 6, 0]}>
          {data.map((d) => <Cell key={d.name} fill={COLORS[d.name] || "#94a3b8"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
