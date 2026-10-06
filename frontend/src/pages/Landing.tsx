import { Link } from "react-router-dom";
import { Eye, ScanEye, BarChart3, ShieldCheck, Layers, Microscope } from "lucide-react";
import { Disclaimer } from "../components/ui/Disclaimer";

const diseases = [
  { name: "Cataract", desc: "Clouding of the eye's natural lens." },
  { name: "Diabetic Retinopathy", desc: "Retinal damage from diabetes." },
  { name: "Glaucoma", desc: "Optic nerve damage, often pressure-related." },
  { name: "Normal", desc: "No signs of the above conditions detected." },
];

const features = [
  { icon: ScanEye, title: "Instant AI Prediction", desc: "Upload a retinal image and get a classification with confidence scores in seconds." },
  { icon: Layers, title: "9 Models Compared", desc: "Traditional ML (Decision Tree, Random Forest, SVM, KNN, Logistic Regression) and deep transfer-learning models (MobileNet, ResNet, EfficientNet, DenseNet)." },
  { icon: BarChart3, title: "Research Dashboard", desc: "Full metric comparison — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and training curves." },
  { icon: Microscope, title: "Explainable AI", desc: "Grad-CAM heatmaps show which regions of the image most influenced each prediction." },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-brand-50 via-white to-white dark:from-slate-950 dark:via-slate-950 dark:to-slate-950">
      <header className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white"><Eye size={18} /></div>
          <span className="font-display text-lg font-bold">RetinaAI</span>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/login" className="btn-secondary">Log in</Link>
          <Link to="/register" className="btn-primary">Get started</Link>
        </div>
      </header>

      <section className="mx-auto max-w-5xl px-6 py-16 text-center">
        <span className="badge bg-brand-50 text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">AI/ML Research Platform</span>
        <h1 className="mt-5 font-display text-4xl font-extrabold leading-tight text-slate-900 dark:text-white sm:text-5xl">
          Eye Disease Prediction<br className="hidden sm:block" /> Using AI/ML Algorithms
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-lg text-slate-500 dark:text-slate-400">
          Upload a retinal image and compare predictions from nine machine-learning and deep-learning
          models trained to recognize Cataract, Diabetic Retinopathy, Glaucoma, and Normal eyes.
        </p>
        <div className="mt-8 flex justify-center gap-4">
          <Link to="/register" className="btn-primary px-6 py-3 text-base">Create an account</Link>
          <Link to="/login" className="btn-secondary px-6 py-3 text-base">I already have one</Link>
        </div>
        <div className="mx-auto mt-10 max-w-2xl"><Disclaimer /></div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-12">
        <h2 className="mb-6 text-center font-display text-2xl font-bold">How it works</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Step n={1} title="Upload" desc="Drag & drop an eye or retinal image." />
          <Step n={2} title="Preprocess" desc="The backend resizes, denoises, and normalizes the image." />
          <Step n={3} title="Predict" desc="The best-performing trained model classifies the image." />
          <Step n={4} title="Explain" desc="View confidence scores and a Grad-CAM explanation." />
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-12">
        <h2 className="mb-6 text-center font-display text-2xl font-bold">Supported classifications</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {diseases.map((d) => (
            <div key={d.name} className="card p-5">
              <p className="font-semibold text-slate-800 dark:text-slate-100">{d.name}</p>
              <p className="mt-1 text-sm text-slate-400">{d.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-12">
        <h2 className="mb-6 text-center font-display text-2xl font-bold">Platform features</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {features.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="card p-5">
              <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-900/40 dark:text-brand-300">
                <Icon size={20} />
              </div>
              <p className="font-semibold text-slate-800 dark:text-slate-100">{title}</p>
              <p className="mt-1 text-sm text-slate-400">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-4xl px-6 pb-20">
        <div className="card flex items-start gap-3 p-6">
          <ShieldCheck className="mt-0.5 shrink-0 text-brand-600" size={22} />
          <p className="text-sm text-slate-500 dark:text-slate-400">
            This is an academic AI/ML methodology demonstration. Model performance depends entirely
            on the training dataset provided and may not generalize to real-world clinical populations.
            It is not a certified medical device.
          </p>
        </div>
      </section>
    </div>
  );
}

function Step({ n, title, desc }: { n: number; title: string; desc: string }) {
  return (
    <div className="card p-5">
      <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-sm font-bold text-white">{n}</div>
      <p className="font-semibold text-slate-800 dark:text-slate-100">{title}</p>
      <p className="mt-1 text-sm text-slate-400">{desc}</p>
    </div>
  );
}
