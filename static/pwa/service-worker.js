// Versionnez à chaque release pour invalider proprement le cache
const SW_VERSION = "v1.0.0";
const PRECACHE = `precache-${SW_VERSION}`;
const RUNTIME = `runtime-${SW_VERSION}`;

// Routes/ressources critiques disponibles hors-ligne
const PRECACHE_URLS = [
  "/",                // page d’accueil
  "/offline",         // fallback hors-ligne
  "/static/css/main.css",
  "/static/js/main.js",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png"
].filter(Boolean);

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(PRECACHE).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((key) => {
        if (!key.includes(SW_VERSION)) return caches.delete(key);
      }))
    )
  );
  self.clients.claim();
});

// Stratégies de cache simples :
// - HTML/navigation: Network-first avec fallback offline
// - Static assets (CSS/JS/images): Stale-while-revalidate
self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Ignorer les appels non-GET
  if (req.method !== "GET") return;

  // API: Network-first (si tu as /api/..., ajuste ici)
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(networkFirst(req));
    return;
  }

  // Pages HTML -> navigation
  if (req.mode === "navigate" || (req.headers.get("accept") || "").includes("text/html")) {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(RUNTIME).then((cache) => cache.put(req, copy));
          return res;
        })
        .catch(async () => (await caches.match(req)) || caches.match("/offline"))
    );
    return;
  }

  // Assets statiques
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(staleWhileRevalidate(req));
    return;
  }

  // Par défaut: SWR
  event.respondWith(staleWhileRevalidate(req));
});

async function networkFirst(req) {
  try {
    const fresh = await fetch(req);
    const cache = await caches.open(RUNTIME);
    cache.put(req, fresh.clone());
    return fresh;
  } catch {
    const cached = await caches.match(req);
    if (cached) return cached;
    // Pour requêtes HTML, proposer la page offline
    if ((req.headers.get("accept") || "").includes("text/html")) {
      return caches.match("/offline");
    }
    return new Response("", { status: 503, statusText: "Service Unavailable" });
  }
}

async function staleWhileRevalidate(req) {
  const cache = await caches.open(RUNTIME);
  const cached = await cache.match(req);
  const networkPromise = fetch(req).then((res) => {
    cache.put(req, res.clone());
    return res;
  }).catch(() => null);
  return cached || networkPromise || fetch(req);
}
