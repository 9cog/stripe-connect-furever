// Stand-in for the real migration runner: needs MONGO_URL and a reachable Mongo.
const net = require("node:net");

const url = process.env.MONGO_URL;
if (!url) {
  console.error("MONGO_URL is not set");
  process.exit(1);
}
const { hostname, port } = new URL(url);
const sock = net.connect({ host: hostname, port: Number(port || 27017) });
sock.setTimeout(3000);
sock.on("connect", () => { console.log("migrations: up to date"); sock.end(); });
sock.on("timeout", () => { console.error(`cannot reach mongo at ${hostname}:${port}`); process.exit(1); });
sock.on("error", (e) => { console.error(`cannot reach mongo at ${hostname}:${port}: ${e.message}`); process.exit(1); });
