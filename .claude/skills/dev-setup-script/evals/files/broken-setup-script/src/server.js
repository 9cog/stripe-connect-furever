const http = require("node:http");
const port = Number(process.env.PORT || 3300);
http.createServer((_req, res) => res.end("notes\n")).listen(port, () => console.log(`notes on ${port}`));
