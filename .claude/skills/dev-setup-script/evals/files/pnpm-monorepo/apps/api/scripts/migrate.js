// Stand-in for a real migration runner: checks the database is reachable.
const net = require("node:net");

const url = process.env.DATABASE_URL;
if (!url) {
  console.error("DATABASE_URL is not set");
  process.exit(1);
}
const { hostname, port } = new URL(url);
const sock = net.connect({ host: hostname, port: Number(port || 5432) });
sock.setTimeout(3000);
sock.on("connect", () => { console.log("migrations: up to date"); sock.end(); });
sock.on("timeout", () => { console.error(`cannot reach ${hostname}:${port}`); process.exit(1); });
sock.on("error", (e) => { console.error(`cannot reach ${hostname}:${port}: ${e.message}`); process.exit(1); });
