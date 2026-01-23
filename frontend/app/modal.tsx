import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, router } from "expo-router";
import { Colors } from "../constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function ResultModal() {
  const params = useLocalSearchParams();
  const probability = Number(params.probability ?? 0);
  const prediction = (params.prediction ?? "low_risk") as
    | "low_risk"
    | "high_risk";

  const pct = Math.round(probability * 100);
  const isHigh = prediction === "high_risk";

  const badgeColor = isHigh ? Colors.danger : Colors.ok;
  const title = isHigh ? "Podwyższone ryzyko" : "Niskie ryzyko";
  const msg = isHigh
    ? "Wynik sugeruje podwyższone ryzyko. Skonsultuj zmianę z dermatologiem."
    : "Wynik sugeruje niskie ryzyko. Obserwuj zmianę i w razie wątpliwości skonsultuj się ze specjalistą.";

  return (
    <LinearGradient colors={[Colors.bg, "#141C33"]} style={styles.container}>
      <View style={styles.card}>
        <View style={[styles.badge, { borderColor: badgeColor }]}>
          <Text style={[styles.badgeText, { color: badgeColor }]}>{title}</Text>
        </View>

        <Text style={styles.pct}>{pct}%</Text>
        <Text style={styles.sub}>Prawdopodobieństwo czerniaka (model AI)</Text>

        <View style={styles.hr} />
        <Text style={styles.msg}>{msg}</Text>

        <PrimaryButton
          title="Zamknij"
          onPress={() => router.back()}
          style={{ marginTop: 16 }}
        />
      </View>

      <Text style={styles.footer}>
        To narzędzie nie jest poradą lekarską. W razie niepokoju – dermatolog.
      </Text>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, justifyContent: "center" },
  card: {
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 24,
    padding: 20,
  },
  badge: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "rgba(255,255,255,0.04)",
  },
  badgeText: { fontWeight: "900" },
  pct: { color: Colors.text, fontSize: 52, fontWeight: "900", marginTop: 14 },
  sub: { color: Colors.muted, marginTop: 4 },
  hr: { height: 1, backgroundColor: Colors.border, marginVertical: 14 },
  msg: { color: Colors.text, lineHeight: 22 },
  footer: {
    color: Colors.muted,
    textAlign: "center",
    marginTop: 14,
    fontSize: 12,
  },
});
