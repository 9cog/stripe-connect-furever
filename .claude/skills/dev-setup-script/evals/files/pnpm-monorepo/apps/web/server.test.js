const test = require("node:test");
const assert = require("node:assert");
const { formatPrice } = require("@acme/shared");

test("shared package resolves from the web app", () => {
  assert.strictEqual(typeof formatPrice, "function");
});
