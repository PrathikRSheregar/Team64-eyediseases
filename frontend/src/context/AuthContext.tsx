import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { api } from "../services/api";
import type { AuthResponse } from "../types";

interface StoredUser {
  id: string;
  name: string;
  role: "user" | "admin";
}

interface AuthContextValue {
  user: StoredUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isAdmin: boolean;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string, confirmPassword: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<StoredUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedToken = localStorage.getItem("eye_app_token");
    const storedUser = localStorage.getItem("eye_app_user");
    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  function persist(auth: AuthResponse) {
    const storedUser: StoredUser = { id: auth.user_id, name: auth.name, role: auth.role };
    localStorage.setItem("eye_app_token", auth.access_token);
    localStorage.setItem("eye_app_user", JSON.stringify(storedUser));
    setToken(auth.access_token);
    setUser(storedUser);
  }

  async function login(email: string, password: string) {
    const res = await api.post<AuthResponse>("/auth/login", { email, password });
    persist(res.data);
  }

  async function register(name: string, email: string, password: string, confirm_password: string) {
    const res = await api.post<AuthResponse>("/auth/register", { name, email, password, confirm_password });
    persist(res.data);
  }

  function logout() {
    localStorage.removeItem("eye_app_token");
    localStorage.removeItem("eye_app_user");
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user, token, loading,
        isAuthenticated: !!token,
        isAdmin: user?.role === "admin",
        login, register, logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
