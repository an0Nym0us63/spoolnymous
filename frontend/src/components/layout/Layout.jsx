import React from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { Home, Settings, LogOut, Printer, Layers } from "lucide-react";
import { useAuth } from "../../store/auth";
import clsx from "clsx";

const nav = [
  { to: "/",         icon: Home,     label: "Accueil"    },
  { to: "/history",  icon: Layers,   label: "Historique" },
  { to: "/settings", icon: Settings, label: "Paramètres" },
];

export default function Layout() {
  const { logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => { logout(); navigate("/login"); };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      {/* Sidebar desktop */}
      <aside className="hidden md:flex flex-col w-16 lg:w-56 bg-gray-900 border-r border-gray-800 py-4 shrink-0">
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 mb-8">
          <Printer size={22} className="text-brand-500 shrink-0" />
          <span className="hidden lg:block font-bold text-sm tracking-wide">Spoolnymous</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 space-y-1 px-2">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) =>
              clsx("flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                isActive
                  ? "bg-brand-600 text-white"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
              )
            }>
              <Icon size={18} className="shrink-0" />
              <span className="hidden lg:block">{label}</span>
            </NavLink>
          ))}
        </nav>

        <button onClick={handleLogout}
          className="flex items-center gap-3 px-5 py-2 text-gray-500 hover:text-red-400 text-sm transition-colors">
          <LogOut size={18} className="shrink-0" />
          <span className="hidden lg:block">Déconnexion</span>
        </button>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet />
        </main>

        {/* Bottom nav mobile */}
        <nav className="md:hidden flex border-t border-gray-800 bg-gray-900">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) =>
              clsx("flex-1 flex flex-col items-center py-3 text-xs transition-colors",
                isActive ? "text-brand-500" : "text-gray-500"
              )
            }>
              <Icon size={20} />
              <span className="mt-1">{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  );
}
