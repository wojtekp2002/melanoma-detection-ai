import { API_BASE_URL } from "@/constants/api";

export type PredictResponse = {
  probability: number;
  threshold: number;
  label: "low_risk" | "high_risk";
  disclaimer: string;
};

export async function predictImage(uri: string): Promise<PredictResponse> {
  const form = new FormData();
    
  form.append("file", {
    uri,
    name: "photo.jpg",
    type: "image/jpeg",
  } as any);

  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      Accept: "application/json",

    },
    body: form,
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${txt}`);
  }

  return res.json();
}
