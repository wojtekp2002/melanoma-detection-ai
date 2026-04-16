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

    family_history_other_cancer: number;
    family_history_other_cancer_relation: string | null;

    had_severe_sunburns: number;
    frequent_sun_exposure: number;
    uses_tanning_beds: number;

    many_moles: number;
    atypical_moles: number;
    very_fair_skin: number;
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

    familyHistoryOtherCancer: intToBool(row.family_history_other_cancer),
    familyHistoryOtherCancerRelation: row.family_history_other_cancer_relation,

    hadSevereSunburns: intToBool(row.had_severe_sunburns),
    frequentSunExposure: intToBool(row.frequent_sun_exposure),
    usesTanningBeds: intToBool(row.uses_tanning_beds),

    manyMoles: intToBool(row.many_moles),
    atypicalMoles: intToBool(row.atypical_moles),
    veryFairSkin: intToBool(row.very_fair_skin),
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

        family_history_other_cancer,
        family_history_other_cancer_relation,

        had_severe_sunburns,
        frequent_sun_exposure,
        uses_tanning_beds,

        many_moles,
        atypical_moles,
        very_fair_skin
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
    [
      1,
      profile.age ?? null,
      profile.skinPhototype ?? null,

      boolToInt(profile.familyHistorySkinCancer),
      profile.familyHistoryRelation ?? null,

      boolToInt(profile.familyHistoryOtherCancer),
      profile.familyHistoryOtherCancerRelation ?? null,

      boolToInt(profile.hadSevereSunburns),
      boolToInt(profile.frequentSunExposure),
      boolToInt(profile.usesTanningBeds),

      boolToInt(profile.manyMoles),
      boolToInt(profile.atypicalMoles),
      boolToInt(profile.veryFairSkin),
    ]
  );
}