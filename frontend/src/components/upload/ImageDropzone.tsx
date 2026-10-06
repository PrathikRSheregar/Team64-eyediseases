import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, X } from "lucide-react";

export function ImageDropzone({ onFileSelected }: { onFileSelected: (file: File | null) => void }) {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[], rejected: any[]) => {
    setError(null);
    if (rejected.length > 0) {
      setError("Please upload a JPG, PNG, BMP, or TIFF image under 10MB.");
      return;
    }
    const file = accepted[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    onFileSelected(file);
  }, [onFileSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [], "image/png": [], "image/bmp": [], "image/tiff": [] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

  function clear() {
    setPreview(null);
    onFileSelected(null);
  }

  if (preview) {
    return (
      <div className="relative">
        <img src={preview} alt="Selected eye image preview" className="mx-auto max-h-72 rounded-2xl border border-slate-200 object-contain dark:border-slate-700" />
        <button
          onClick={clear}
          className="absolute right-2 top-2 rounded-full bg-white/90 p-1.5 text-slate-600 shadow hover:bg-white dark:bg-slate-900/90 dark:text-slate-300"
          aria-label="Remove image"
        >
          <X size={16} />
        </button>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-12 text-center transition ${
        isDragActive ? "border-brand-500 bg-brand-50 dark:bg-brand-950/30" : "border-slate-300 hover:border-brand-400 dark:border-slate-700"
      }`}
    >
      <input {...getInputProps()} aria-label="Upload eye image" />
      <UploadCloud className="text-brand-500" size={36} />
      <div>
        <p className="font-medium text-slate-700 dark:text-slate-200">Drag & drop an eye image, or click to browse</p>
        <p className="mt-1 text-xs text-slate-400">JPG, PNG, BMP, or TIFF — up to 10MB</p>
      </div>
      {error && <p className="text-xs font-medium text-red-500">{error}</p>}
    </div>
  );
}
