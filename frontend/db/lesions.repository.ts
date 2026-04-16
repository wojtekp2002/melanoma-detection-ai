import { getDb } from "./database";
import { Lesion } from "@/types/lesion";

export async function createLesion(lesion: Lesion): Promise<number> {
  const db = getDb();

  const result = await db.runAsync(
    `
      INSERT INTO lesions (
        name,
        body_side,
        body_label,
        x,
        y,
        created_at
      )
      VALUES (?, ?, ?, ?, ?, ?)
    `,
    [
      lesion.name,
      lesion.bodySide,
      lesion.bodyLabel ?? null,
      lesion.x ?? null,
      lesion.y ?? null,
      lesion.createdAt,
    ]
  );

  return result.lastInsertRowId;
}

export async function getAllLesions(): Promise<Lesion[]> {
  const db = getDb();

  const rows = await db.getAllAsync<{
    id: number;
    name: string;
    body_side: "front" | "back";
    body_label: string | null;
    x: number | null;
    y: number | null;
    created_at: string;
  }>(`
    SELECT *
    FROM lesions
    ORDER BY datetime(created_at) DESC
  `);

  return rows.map((row) => ({
    id: row.id,
    name: row.name,
    bodySide: row.body_side,
    bodyLabel: row.body_label,
    x: row.x,
    y: row.y,
    createdAt: row.created_at,
  }));
}

export async function getLesionsCount(): Promise<number> {
  const db = getDb();

  const row = await db.getFirstAsync<{ count: number }>(
    `SELECT COUNT(*) as count FROM lesions`
  );

  return row?.count ?? 0;
}