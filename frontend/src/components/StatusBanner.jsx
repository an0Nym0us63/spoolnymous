import React from "react";
import { Wifi, WifiOff, Loader2 } from "lucide-react";
import clsx from "clsx";

const STATUS_LABELS = {
  IDLE: "En veille", RUNNING: "En cours", PAUSE: "En pause",
  FINISH: "Terminé", FAILED: "Erreur", PREPARE: "Préparation",
};

const STATUS_COLORS = {
  RUNNING: "from-green-500 to-blue-500",
  PAUSE:   "from-yellow-400 to-yellow-600",
  FINISH:  "from-emerald-500 to-green-400",
  FAILED:  "from-red-600 to-pink-500",
  IDLE:    "from-gray-700 to-gray-600",
  PREPARE: "from-blue-500 to-indigo-500",
};

export default function StatusBanner({ status }) {
  if (!status) return null;

  const pct = status.progress ?? 0;
  const gradient = STATUS_COLORS[status.status] ?? STATUS_COLORS.IDLE;
  const label = STATUS_LABELS[status.status] ?? status.status;

  return (
    <div className="relative rounded-2xl overflow-hidden glass">
      {/* Barre de progression */}
      <div className="absolute inset-0 flex pointer-events-none">
        <div
          className={clsx("h-full bg-gradient-to-r transition-all duration-700 opacity-30", gradient)}
          style={{ width: `${pct}%`, minWidth: "2rem" }}
        />
      </div>

      <div className="relative z-10 flex items-center justify-between px-4 py-3 gap-4">
        {/* Gauche : thumbnail + nom */}
        <div className="flex items-center gap-3 min-w-0">
          {status.thumbnail ? (
            <img src={`/uploads/prints/${status.thumbnail}`} alt=""
              className="w-10 h-10 rounded-lg object-cover shrink-0 ring-1 ring-white/10" />
          ) : (
            <div className="w-10 h-10 rounded-lg bg-gray-800 shrink-0" />
          )}
          <div className="min-w-0">
            <p className="text-sm font-medium truncate">{status.print_name || "—"}</p>
            <p className="text-xs text-gray-400">{label}</p>
          </div>
        </div>

        {/* Centre : progression */}
        {status.status === "RUNNING" && (
          <div className="text-center shrink-0">
            <p className="text-2xl font-bold mono">{pct}%</p>
            {status.remaining_minutes > 0 && (
              <p className="text-xs text-gray-400">
                {Math.floor(status.remaining_minutes / 60)}h {status.remaining_minutes % 60}min restantes
              </p>
            )}
          </div>
        )}

        {/* Droite : connexion */}
        <div className="shrink-0">
          {status.connected
            ? <Wifi size={18} className="text-green-400" />
            : <WifiOff size={18} className="text-red-400" />}
        </div>
      </div>
    </div>
  );
}
