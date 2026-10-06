import axios, { AxiosError } from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api",
  timeout: 30000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("eye_app_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (error: AxiosError<{ detail?: string }>) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("eye_app_token");
      localStorage.removeItem("eye_app_user");
      if (!window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    }
    const message = error.response?.data?.detail || error.message || "Something went wrong.";
    return Promise.reject(new Error(message));
  }
);

export function apiOrigin(): string {
  const base = import.meta.env.VITE_API_BASE_URL || "/api";
  return base.replace(/\/api\/?$/, "");
}
