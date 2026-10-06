import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-3 text-center">
      <p className="font-display text-3xl font-bold">404</p>
      <p className="text-slate-400">Page not found.</p>
      <Link to="/" className="btn-primary mt-2">Go home</Link>
    </div>
  );
}
