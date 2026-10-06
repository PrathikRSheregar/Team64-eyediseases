import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import { Eye } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  function validate() {
    const e: Record<string, string> = {};
    if (form.name.trim().length < 1) e.name = "Name is required.";
    if (!/^\S+@\S+\.\S+$/.test(form.email)) e.email = "Enter a valid email address.";
    if (form.password.length < 8) e.password = "Password must be at least 8 characters.";
    if (form.confirm !== form.password) e.confirm = "Passwords do not match.";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  async function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    if (!validate()) return;
    setSubmitting(true);
    try {
      await register(form.name, form.email, form.password, form.confirm);
      toast.success("Account created!");
      navigate("/dashboard");
    } catch (err: any) {
      toast.error(err.message || "Registration failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 dark:bg-slate-950">
      <div className="w-full max-w-md">
        <div className="mb-6 flex items-center justify-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white"><Eye size={18} /></div>
          <span className="font-display text-lg font-bold">RetinaAI</span>
        </div>
        <div className="card p-8">
          <h1 className="font-display text-xl font-bold text-slate-800 dark:text-slate-100">Create an account</h1>
          <p className="mt-1 text-sm text-slate-400">Start predicting eye conditions with AI.</p>

          <form onSubmit={handleSubmit} className="mt-6 space-y-4" noValidate>
            <Field label="Full name" id="name" value={form.name} onChange={(v) => setForm({ ...form, name: v })} error={errors.name} autoComplete="name" />
            <Field label="Email" id="email" type="email" value={form.email} onChange={(v) => setForm({ ...form, email: v })} error={errors.email} autoComplete="email" />
            <Field label="Password" id="password" type="password" value={form.password} onChange={(v) => setForm({ ...form, password: v })} error={errors.password} autoComplete="new-password" />
            <Field label="Confirm password" id="confirm" type="password" value={form.confirm} onChange={(v) => setForm({ ...form, confirm: v })} error={errors.confirm} autoComplete="new-password" />
            <button type="submit" disabled={submitting} className="btn-primary w-full">
              {submitting ? "Creating account…" : "Create account"}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-slate-400">
            Already have an account? <Link to="/login" className="font-semibold text-brand-600">Log in</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

function Field({ label, id, value, onChange, error, type = "text", autoComplete }: {
  label: string; id: string; value: string; onChange: (v: string) => void; error?: string; type?: string; autoComplete?: string;
}) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-sm font-medium text-slate-600 dark:text-slate-300">{label}</label>
      <input id={id} type={type} className="input-field" value={value} onChange={(e) => onChange(e.target.value)} autoComplete={autoComplete} />
      {error && <p className="mt-1 text-xs text-red-500">{error}</p>}
    </div>
  );
}
