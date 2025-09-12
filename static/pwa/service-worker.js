// static/pwa/service-worker.js
// SW sans cache : ne stocke rien, ne répond jamais depuis caches.
// Sert uniquement de filet de sécurité pour la navigation offline.
const SW_VERSION = "nocache-1";

self.addEventListener("install", (event) => {
  // pas de precache
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // purge TOUT ce que d’anciennes versions auraient laissé
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

// Politique :
// - Pour les navigations (HTML) : réseau d'abord, et si offline -> /offline (si dispo côté serveur)
// - Pour tout le reste (static, API) : réseau d'abord, pas de cache.
self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const isHTML = req.mode === "navigate" || (req.headers.get("accept") || "").includes("text/html");

  event.respondWith((async () => {
    try {
      return await fetch(req, { cache: "no-store" }); // force réseau
    } catch (e) {
      if (isHTML) {
        // si tu as une page /offline côté Flask
        return fetch("/offline", { cache: "no-store" }).catch(() => new Response("Offline", { status: 503 }));
      }
      return new Response("", { status: 503, statusText: "Offline" });
    }
  })());
});
