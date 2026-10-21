const http = require("node:http");
const port = Number(process.env.PORT || 4100);
http.createServer((_req, res) => res.end("ok\n")).listen(port, () => console.log(`api on ${port}`));
