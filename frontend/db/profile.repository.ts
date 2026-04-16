import { getDb } from "./database";
import { UserRiskProfile } from "@/types/profile";

function boolToInt(value: boolean) {
  return value ? 1 : 0;
}

function intToBool(value: number | null | undefined) {
  return value === 1;
}

export async function getUserRiskProfile(): Promise<UserRiskProfile | null> {
  const db = getDb();

  const row = await db.getFirstAsync<{
    id: number;
    age: number | null;
    skin_phototype: UserRiskProfile["skinPhototype"];
    family_history_skin_cancer: number;
    family_history_relation: string | null;
    had_severe_sunburns: number;
    many_moles: number;
  }>(`SELECT * FROM user_profile WHERE id = 1`);

  if (!row) {
    return null;
  }

  return {
    id: row.id,
    age: row.age,
    skinPhototype: row.skin_phototype,
    familyHistorySkinCancer: intToBool(row.family_history_skin_cancer),
    familyHistoryRelation: row.family_history_relation,
    hadSevereSunburns: intToBool(row.had_severe_sunburns),
    manyMoles: intToBool(row.many_moles),
  };
}

export async function saveUserRiskProfile(profile: UserRiskProfile): Promise<void> {
  const db = getDb();

  await db.runAsync(
    `
      INSERT OR REPLACE INTO user_profile (
        id,
        age,
        skin_phototype,
        family_history_skin_cancer,
        family_history_relation,
        had_severe_sunburns,
        many_moles
      )
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `,
    [
      1,
      profile.age ?? null,
      profile.skinPhototype ?? null,
      profile.familyHistorySkinCancer ? 1 : 0,
      profile.familyHistoryRelation ?? null,
      profile.hadSevereSunburns ? 1 : 0,
      profile.manyMoles ? 1 : 0,
    ]
  );
}