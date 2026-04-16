export type ObservationLabel = "low_risk" | "high_risk";

export type Observation = {
  id?: number;
  lesionId?: number | null;
  imageUri: string;
  probability: number;
  label: ObservationLabel;
  createdAt: string;
  note?: string | null;
};