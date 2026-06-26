import React from "react";
import clsx from "clsx";

function colorStyle(hex) {
  if (!hex || hex.length < 6) return {};
  const h = hex.startsWith("#") ? hex : `#${hex}`;
  return { backgroundColor: h };
}

function TraySlot({ tray, active }) {
  const isEmpty = tray.empty || !tray.filament_type;
  return (
    <div className={clsx(
      "relative rounded-xl p-3 border transition-all",
      active ? "border-brand-500 shadow-lg shadow-brand-500/20" : "border-gray-700/50",
      isEmpty ? "bg-gray-800/40" : "bg-gray-800/70"
    )}>
      {/* Couleur */}
      <div className="flex items-center gap-2 mb-2">
        <div className="w-5 h-5 rounded-md ring-1 ring-white/20 shrink-0"
          style={isEmpty ? { backgroundColor: "#374151" } : colorStyle(tray.color)} />
        <span className="text-xs font-mono text-gray-300">#{tray.id + 1}</span>
        {active && <span className="ml-auto text-[10px] text-brand-400 font-semibold">ACTIF</span>}
      </div>

      {isEmpty ? (
        <p className="text-xs text-gray-600">Vide</p>
      ) : (
        <>
          <p className="text-xs font-medium text-gray-200 truncate">{tray.filament_type}</p>
          {/* Barre de reste */}
          <div className="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all"
              style={{ width: `${tray.remain}%`, ...colorStyle(tray.color) }} />
          </div>
          <p className="text-[10px] text-gray-500 mt-1 mono">{tray.remain}%</p>
        </>
      )}
    </div>
  );
}

function AMSUnit({ ams }) {
  return (
    <div className="glass rounded-2xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          AMS {ams.id + 1}
        </h3>
        <div className="flex gap-3 text-xs text-gray-500">
          <span>💧 {ams.humidity}%</span>
          <span>🌡 {ams.temp}°C</span>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {ams.trays.map((tray) => (
          <TraySlot key={tray.id} tray={tray} active={false} />
        ))}
      </div>
    </div>
  );
}

export default function AMSGrid({ amsList }) {
  if (!amsList?.length) return (
    <div className="glass rounded-2xl p-6 text-center text-gray-600 text-sm">
      Aucun AMS détecté
    </div>
  );

  return (
    <div className="space-y-3">
      {amsList.map((ams) => <AMSUnit key={ams.id} ams={ams} />)}
    </div>
  );
}
