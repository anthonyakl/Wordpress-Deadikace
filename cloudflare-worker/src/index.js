const WORDPRESS_ORIGIN = "https://www.deadikace.com";
const TOKEN_HEADER = "X-Deadikace-Proxy-Token";

function safeEqual(left, right) {
  if (!left || !right || left.length !== right.length) return false;
  let different = 0;
  for (let i = 0; i < left.length; i++) different |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return different === 0;
}

export default {
  async fetch(request, env) {
    if (!safeEqual(request.headers.get(TOKEN_HEADER), env.PROXY_TOKEN)) {
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
    headers.set("Host", target.host);

    return fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    });
  },
};
