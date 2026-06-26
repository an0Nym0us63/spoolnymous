import React from "react";
import { Thermometer, Wind, Layers } from "lucide-react";

function TempPill({ label, current, target, icon: Icon }) {
  return (
    <div className="glass rounded-xl px-3 py-2 flex items-center gap-2 text-sm">
      <Icon size={15} className="text-gray-400 shrink-0" />
      <span className="text-gray-400 text-xs">{label}</span>
      <span className="mono font-semibold ml-auto">{current ?? "—"}°</span>
      {target > 0 && <span className="text-xs text-gray-500 mono">/{target}°</span>}
    </div>
  );
}

export default function PrinterCard({ status }) {
  if (!status) return (
    <div className="glass rounded-2xl p-5 animate-pulse h-28" />
  );

  return (
    <div className="glass rounded-2xl p-5">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Températures</h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <TempPill label="Buse"    current={status.nozzle_temp}  target={status.target_nozzle_temp}  icon={Thermometer} />
        <TempPill label="Plateau" current={status.bed_temp}     target={status.target_bed_temp}     icon={Thermometer} />
        <TempPill label="Chambre" current={status.chamber_temp} target={0}                          icon={Wind} />
        {status.total_layers > 0 && (
          <div className="glass rounded-xl px-3 py-2 flex items-center gap-2 text-sm">
            <Layers size={15} className="text-gray-400 shrink-0" />
            <span className="text-gray-400 text-xs">Couches</span>
            <span className="mono font-semibold ml-auto">{status.layer}</span>
            <span className="text-xs text-gray-500 mono">/{status.total_layers}</span>
          </div>
        )}
      </div>
    </div>
  );
}
