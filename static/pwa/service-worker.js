// Un service worker qui ne fait rien sauf permettre l'installation PWA

self.addEventListener("install", (event) => {
  // Pas de cache, on passe directement à l'activation
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // Supprime tous les caches existants (juste au cas où)
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

// Aucun fetch intercepté → comportement identique au navigateur
