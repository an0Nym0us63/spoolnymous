// Versionnez à chaque release pour invalider proprement le cache
const SW_VERSION = "v1.0.0";
const PRECACHE = `precache-${SW_VERSION}`;
const RUNTIME = `runtime-${SW_VERSION}`;

// Routes/ressources critiques disponibles hors-ligne
const PRECACHE_URLS = [
  "/",                // page d’accueil
  "/offline",         // fallback hors-ligne
  "/static/css/style.css",
  "/static/js/print_history.js",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png"
].filter(Boolean);
self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(PRECACHE);

    // Utilise cache: 'reload' pour éviter de tomber sur des réponses opaques/anciennes
    const requests = PRECACHE_URLS.map(
      (u) => new Request(u, { cache: "reload" })
    );

    const results = await Promise.allSettled(
      requests.map((req) => fetch(req))
    );

    const okRequests = [];
    results.forEach((res, i) => {
      if (res.status === "fulfilled" && res.value && res.value.ok && res.value.type === "basic") {
        okRequests.push(requests[i]);
      } else {
        console.warn("[SW] Pré-cache ignoré (échec) →", requests[i].url, res);
      }
    });

    // On met en cache uniquement les réponses valides
    await cache.addAll(okRequests);
  })());
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((key) => {
      if (!key.includes(SW_VERSION)) return caches.delete(key);
    }));
    await self.clients.claim();
  })());
});

// Network-first pour HTML/navigation, SWR pour static
self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  if (req.method !== "GET") return;

  // API (si tu en as) : réseau d’abord
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(networkFirst(req));
    return;
  }

  // Navigation HTML
  if (req.mode === "navigate" || (req.headers.get("accept") || "").includes("text/html")) {
    event.respondWith((async () => {
      try {
        const fresh = await fetch(req);
        const cache = await caches.open(RUNTIME);
        cache.put(req, fresh.clone());
        return fresh;
      } catch {
        const cached = await caches.match(req);
        return cached || caches.match("/offline");
      }
    })());
    return;
  }

  // Assets statiques
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(staleWhileRevalidate(req));
    return;
  }

  // Par défaut
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