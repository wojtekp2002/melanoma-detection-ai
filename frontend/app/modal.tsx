import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function ResultModal() {
  const params = useLocalSearchParams();
  const probability = Number(params.probability ?? 0);
  const prediction = (params.prediction ?? "low_risk") as
    | "low_risk"
    | "high_risk";

  const pct = Math.round(probability * 100);

  const isHigh = prediction === "high_risk";

  const riskTitle =
    pct < 30 ? "Niskie ryzyko" : pct < 60 ? "Umiarkowane ryzyko" : "Podwyższone ryzyko";

  const badgeColor =
    pct < 30 ? Colors.ok : pct < 60 ? Colors.primary2 : Colors.danger;

  const recommendation =
    pct < 30
      ? "Wynik sugeruje niskie ryzyko. Obserwuj zmianę i w razie wątpliwości skonsultuj się ze specjalistą."
      : pct < 60
      ? "Wynik nie jest jednoznaczny. Warto obserwować zmianę i rozważyć konsultację dermatologiczną."
      : "Wynik sugeruje podwyższone ryzyko. Zalecana jest konsultacja z dermatologiem.";

  return (
    <LinearGradient colors={[Colors.bg, "#141C33", "#1A2240"]} style={styles.container}>
      <View style={styles.card}>
        <View style={[styles.badge, { borderColor: badgeColor }]}>
          <Text style={[styles.badgeText, { color: badgeColor }]}>{riskTitle}</Text>
        </View>

        <Text style={styles.percent}>{pct}%</Text>
        <Text style={styles.subtext}>Szacowane prawdopodobieństwo czerniaka</Text>

        <View style={styles.divider} />

        <Text style={styles.sectionTitle}>Interpretacja</Text>
        <Text style={styles.description}>{recommendation}</Text>

        <View style={styles.infoBox}>
          <Text style={styles.infoTitle}>Pamiętaj</Text>
          <Text style={styles.infoText}>
            Model ma charakter przesiewowy i nie zastępuje badania lekarskiego.
          </Text>
        </View>

        <PrimaryButton
          title="Zamknij"
          onPress={() => router.back()}
          style={{ marginTop: 16 }}
        />
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: "center",
  },
  card: {
    backgroundColor: Colors.card,
    borderRadius: 26,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 22,
  },
  badge: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "rgba(255,255,255,0.04)",
  },
  badgeText: {
    fontWeight: "900",
    fontSize: 13,
  },
  percent: {
    color: Colors.text,
    fontSize: 56,
    fontWeight: "900",
    marginTop: 16,
  },
  subtext: {
    color: Colors.muted,
    marginTop: 4,
    fontSize: 14,
  },
  divider: {
    height: 1,
    backgroundColor: Colors.border,
    marginVertical: 16,
  },
  sectionTitle: {
    color: Colors.text,
    fontWeight: "900",
    fontSize: 16,
    marginBottom: 8,
  },
  description: {
    color: Colors.text,
    lineHeight: 22,
    fontSize: 15,
  },
  infoBox: {
    marginTop: 16,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 18,
    padding: 14,
  },
  infoTitle: {
    color: Colors.text,
    fontWeight: "800",
    marginBottom: 6,
  },
  infoText: {
    color: Colors.muted,
    lineHeight: 20,
    fontSize: 13,
  },
});