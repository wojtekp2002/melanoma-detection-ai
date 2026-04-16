export type BodySide = "front" | "back";

export type Lesion = {
  id?: number;
  name: string;
  bodySide: BodySide;
  bodyLabel?: string | null;
  x?: number | null;
  y?: number | null;
  createdAt: string;
};