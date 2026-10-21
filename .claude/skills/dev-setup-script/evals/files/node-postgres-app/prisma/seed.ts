import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// Inserts demo rows; running it twice creates duplicates.
async function main() {
  await prisma.order.createMany({
    data: [
      { sku: "WIDGET-1", quantity: 3 },
      { sku: "GADGET-7", quantity: 1 },
    ],
  });
}

main().finally(() => prisma.$disconnect());
