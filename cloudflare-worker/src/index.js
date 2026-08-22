const WORDPRESS_ORIGIN = "https://www.deadikace.com";
const TOKEN_HEADER = "X-Deadikace-Proxy-Token";

async function tokenMatches(provided, expected) {
  if (!provided || !expected) return false;
  const encoder = new TextEncoder();
  const left = encoder.encode(provided);
  const right = encoder.encode(expected);
  if (left.byteLength !== right.byteLength) return false;
  return crypto.subtle.timingSafeEqual(left, right);
}

export default {
  async fetch(request, env) {
    if (!(await tokenMatches(request.headers.get(TOKEN_HEADER), env.PROXY_TOKEN))) {
      return new Response("Unauthorized", { status: 401 });
    }

    const incoming = new URL(request.url);
    const isRootProbe = incoming.pathname === "/wp-json/";
    const allowed = (isRootProbe && (request.method === "GET" || request.method === "HEAD")) ||
      incoming.pathname.startsWith("/wp-json/wp/v2/");
    if (!allowed) return new Response("Not found", { status: 404 });

    const target = new URL(incoming.pathname + incoming.search, WORDPRESS_ORIGIN);
    const headers = new Headers(request.headers);
    headers.delete(TOKEN_HEADER);
    headers.delete("host");
    headers.delete("cf-connecting-ip");
    headers.delete("cf-ipcountry");
    headers.delete("cf-ray");
    headers.delete("cf-visitor");
    return fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    });
  },
};
