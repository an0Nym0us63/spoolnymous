import React, { useState, useEffect } from "react";
import { Save, Wifi, WifiOff, RefreshCw } from "lucide-react";
import client from "../api/client";

export default function Settings() {
  const [form, setForm] = useState({
    PRINTER_IP: "", PRINTER_ID: "", PRINTER_ACCESS_CODE: "",
    PRINTER_NAME: "", ADMIN_USERNAME: "admin",
    ADMIN_PASSWORD: "", COST_BY_HOUR: "0", AUTO_SPEND: "true",
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    client.get("/settings").then(({ data }) => {
      setForm(f => ({ ...f, ...data, ADMIN_PASSWORD: "" }));
      setLoading(false);
    });
  }, []);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    const payload = { ...form };
    if (!payload.ADMIN_PASSWORD) delete payload.ADMIN_PASSWORD;
    try {
      await client.patch("/settings", payload);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } finally {
      setSaving(false);
    }
  };

  const Field = ({ label, name, type = "text", placeholder = "" }) => (
    <div>
      <label className="block text-xs text-gray-400 mb-1.5">{label}</label>
      <input
        type={type} value={form[name] || ""}
        placeholder={placeholder}
        onChange={(e) => setForm(f => ({ ...f, [name]: e.target.value }))}
        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-brand-500 transition-colors"
      />
    </div>
  );

  if (loading) return <div className="flex items-center justify-center h-64 text-gray-500">Chargement...</div>;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-xl font-bold">Paramètres</h1>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Imprimante */}
        <section className="glass rounded-2xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
            <Wifi size={16} className="text-brand-500" /> Imprimante
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Adresse IP" name="PRINTER_IP" placeholder="192.168.1.xxx" />
            <Field label="Numéro de série" name="PRINTER_ID" placeholder="01P..." />
            <Field label="Code d'accès" name="PRINTER_ACCESS_CODE" placeholder="••••••••" />
            <Field label="Nom affiché" name="PRINTER_NAME" placeholder="Mon imprimante" />
          </div>
        </section>

        {/* Compte */}
        <section className="glass rounded-2xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Compte</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Nom d'utilisateur" name="ADMIN_USERNAME" />
            <Field label="Nouveau mot de passe" name="ADMIN_PASSWORD" type="password" placeholder="Laisser vide pour ne pas changer" />
          </div>
        </section>

        {/* Coûts */}
        <section className="glass rounded-2xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Coûts électricité</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Field label="Tarif (€/h)" name="COST_BY_HOUR" placeholder="0.20" />
          </div>
        </section>

        <button type="submit" disabled={saving}
          className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors">
          {saving ? <RefreshCw size={16} className="animate-spin" /> : <Save size={16} />}
          {saved ? "Sauvegardé ✓" : "Sauvegarder"}
        </button>
      </form>
    </div>
  );
}
