export type SkinPhototype = "I" | "II" | "III" | "IV" | "V" | "VI";

export type UserRiskProfile = {
  id?: number;
  age?: number | null;
  skinPhototype?: SkinPhototype | null;
  familyHistorySkinCancer: boolean;
  familyHistoryRelation?: string | null;
  hadSevereSunburns: boolean;
  manyMoles: boolean;
};