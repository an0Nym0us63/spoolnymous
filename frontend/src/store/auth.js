import { create } from "zustand";
import { persist } from "zustand/middleware";
import client from "../api/client";

export const useAuth = create(
  persist(
    (set) => ({
      token: null,
      isAuthenticated: false,

      login: async (username, password) => {
        const form = new FormData();
        form.append("username", username);
        form.append("password", password);
        const { data } = await client.post("/auth/login", form);
        localStorage.setItem("token", data.access_token);
        set({ token: data.access_token, isAuthenticated: true });
        return true;
      },

      logout: () => {
        localStorage.removeItem("token");
        set({ token: null, isAuthenticated: false });
      },
    }),
    { name: "auth-store", partialize: (s) => ({ token: s.token, isAuthenticated: s.isAuthenticated }) }
  )
);
