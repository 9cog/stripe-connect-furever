const test = require("node:test");
const assert = require("node:assert");
const { formatPrice } = require("./index");

test("formats cents as dollars", () => {
  assert.strictEqual(formatPrice(1999), "$19.99");
});
