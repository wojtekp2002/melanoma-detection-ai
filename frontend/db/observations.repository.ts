import { getDb } from "./database";
import { Observation } from "@/types/observation";

export async function createObservation(observation: Observation): Promise<number> {
  const db = getDb();

  const result = await db.runAsync(
    `
      INSERT INTO observations (
        lesion_id,
        image_uri,
        probability,
        label,
        created_at,
        note
      )
      VALUES (?, ?, ?, ?, ?, ?)
    `,
    [
      observation.lesionId ?? null,
      observation.imageUri,
      observation.probability,
      observation.label,
      observation.createdAt,
      observation.note ?? null,
    ]
  );

  return result.lastInsertRowId;
}

export async function getAllObservations(): Promise<Observation[]> {
  const db = getDb();

  const rows = await db.getAllAsync<{
    id: number;
    lesion_id: number | null;
    image_uri: string;
    probability: number;
    label: "low_risk" | "high_risk";
    created_at: string;
    note: string | null;
  }>(`
    SELECT *
    FROM observations
    ORDER BY datetime(created_at) DESC
  `);

  return rows.map((row) => ({
    id: row.id,
    lesionId: row.lesion_id,
    imageUri: row.image_uri,
    probability: row.probability,
    label: row.label,
    createdAt: row.created_at,
    note: row.note,
  }));
}

export async function getObservationById(id: number): Promise<Observation | null> {
  const db = getDb();

  const row = await db.getFirstAsync<{
    id: number;
    lesion_id: number | null;
    image_uri: string;
    probability: number;
    label: "low_risk" | "high_risk";
    created_at: string;
    note: string | null;
  }>(
    `
      SELECT *
      FROM observations
      WHERE id = ?
    `,
    [id]
  );

  if (!row) {
    return null;
  }

  return {
    id: row.id,
    lesionId: row.lesion_id,
    imageUri: row.image_uri,
    probability: row.probability,
    label: row.label,
    createdAt: row.created_at,
    note: row.note,
  };
}

export async function getObservationsByLesionId(
  lesionId: number
): Promise<Observation[]> {
  const db = getDb();

  const rows = await db.getAllAsync<{
    id: number;
    lesion_id: number | null;
    image_uri: string;
    probability: number;
    label: "low_risk" | "high_risk";
    created_at: string;
    note: string | null;
  }>(
    `
      SELECT *
      FROM observations
      WHERE lesion_id = ?
      ORDER BY datetime(created_at) DESC
    `,
    [lesionId]
  );

  return rows.map((row) => ({
    id: row.id,
    lesionId: row.lesion_id,
    imageUri: row.image_uri,
    probability: row.probability,
    label: row.label,
    createdAt: row.created_at,
    note: row.note,
  }));
}

export async function getObservationsCount(): Promise<number> {
  const db = getDb();

  const row = await db.getFirstAsync<{ count: number }>(
    `SELECT COUNT(*) as count FROM observations`
  );

  return row?.count ?? 0;
}