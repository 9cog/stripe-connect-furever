const http = require("node:http");
const { formatPrice } = require("@acme/shared");

const api = process.env.NEXT_PUBLIC_API_URL;
if (!api) {
  console.error("NEXT_PUBLIC_API_URL is not set (copy .env.local.example to .env.local)");
  process.exit(1);
}

http
  .createServer((_req, res) => res.end(`Acme web — API: ${api} — sample price ${formatPrice(1999)}\n`))
  .listen(3000, () => console.log("web on http://localhost:3000"));
