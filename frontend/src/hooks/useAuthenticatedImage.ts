import { useEffect, useState } from "react";
import { api } from "../services/api";

/**
 * Protected image endpoints (original upload, Grad-CAM overlay) require a
 * Bearer JWT, which a plain <img src="..."> cannot send. This hook fetches
 * the image bytes through the authenticated axios client and exposes a
 * local blob: URL that a normal <img> can render.
 */
export function useAuthenticatedImage(url: string | null | undefined): { src: string | null; loading: boolean; error: string | null } {
  const [src, setSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!url);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!url) {
      setSrc(null);
      return;
    }
    let objectUrl: string | null = null;
    setLoading(true);
    setError(null);

    api.get(url, { responseType: "blob" })
      .then((res) => {
        objectUrl = URL.createObjectURL(res.data);
        setSrc(objectUrl);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);

  return { src, loading, error };
}
