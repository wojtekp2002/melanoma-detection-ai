export type SkinPhototype = "I" | "II" | "III" | "IV" | "V" | "VI";

export type UserRiskProfile = {
  id?: number;
  age?: number | null;
  skinPhototype?: SkinPhototype | null;

  familyHistorySkinCancer: boolean;
  familyHistoryRelation?: string | null;

  familyHistoryOtherCancer: boolean;
  familyHistoryOtherCancerRelation?: string | null;

  hadSevereSunburns: boolean;
  frequentSunExposure: boolean;
  usesTanningBeds: boolean;

  manyMoles: boolean;
  atypicalMoles: boolean;
  veryFairSkin: boolean;
};