import React from "react";
import { View, Text, StyleSheet, ScrollView, Pressable } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, router } from "expo-router";
import { FontAwesome } from "@expo/vector-icons";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function ResultModal() {
  const params = useLocalSearchParams();

  const probability = Number(params.probability ?? 0);
  const prediction = (params.prediction ?? "low_risk") as
    | "low_risk"
    | "high_risk";

  const pct = Math.round(probability * 100);

  const riskTitle =
    pct < 30
      ? "Niskie ryzyko"
      : pct < 60
      ? "Umiarkowane ryzyko"
      : "Podwyższone ryzyko";

  const badgeColor =
    pct < 30 ? Colors.ok : pct < 60 ? Colors.primary2 : Colors.danger;

  const headline =
    pct < 30
      ? "Model nie wykrył silnych cech wysokiego ryzyka."
      : pct < 60
      ? "Wynik wymaga ostrożnej interpretacji."
      : "W obrazie wykryto cechy, które mogą wymagać dalszej konsultacji.";

  const interpretation =
    pct < 30
      ? "To oznacza, że na podstawie tego zdjęcia model ocenił zmianę jako mniej podejrzaną. Mimo to nadal warto obserwować ją w czasie i reagować, jeśli zacznie się zmieniać."
      : pct < 60
      ? "Wynik nie jest jednoznaczny. Taka analiza nie przesądza o rozpoznaniu, ale może być sygnałem, żeby dokładniej monitorować zmianę i rozważyć konsultację dermatologiczną."
      : "Ten wynik nie oznacza diagnozy czerniaka, ale sugeruje, że warto skonsultować zmianę z dermatologiem, szczególnie jeśli zmiana rośnie, swędzi, krwawi lub zmienia kolor czy kształt.";

  const nextSteps =
    pct < 30
      ? [
          "Zapisz wynik w historii i obserwuj zmianę w czasie.",
          "Powtórz analizę przy kolejnym zdjęciu w podobnych warunkach.",
          "Skonsultuj się z dermatologiem, jeśli zmiana Cię niepokoi.",
        ]
      : pct < 60
      ? [
          "Zapisz wynik i monitoruj zmianę w kolejnych dniach lub tygodniach.",
          "Zrób kolejne zdjęcie przy dobrym świetle i bliskim kadrze.",
          "Rozważ konsultację dermatologiczną, jeśli zmiana wygląda nietypowo.",
        ]
      : [
          "Rozważ możliwie szybką konsultację dermatologiczną.",
          "Zapisz wynik i zachowaj historię obserwacji tej zmiany.",
          "Jeśli zmiana się zmienia lub daje objawy, nie odkładaj wizyty.",
        ];

  return (
    <LinearGradient
      colors={[Colors.background, "#0A1627", "#102038"]}
      style={styles.container}
    >
      <ScrollView
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.card}>
          <View style={styles.topRow}>
            <View style={[styles.badge, { borderColor: badgeColor }]}>
              <Text style={[styles.badgeText, { color: badgeColor }]}>
                {riskTitle}
              </Text>
            </View>

            <View style={styles.aiBadge}>
              <FontAwesome name="shield" size={12} color={Colors.primary2} />
              <Text style={styles.aiBadgeText}>AI screening</Text>
            </View>
          </View>

          <Text style={styles.percent}>{pct}%</Text>
          <Text style={styles.subtext}>Szacowane prawdopodobieństwo ryzyka na podstawie obrazu</Text>

          <View style={styles.heroBox}>
            <Text style={styles.heroTitle}>{headline}</Text>
            <Text style={styles.heroText}>{interpretation}</Text>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Co dalej</Text>

            {nextSteps.map((step, index) => (
              <View key={index} style={styles.stepRow}>
                <View style={styles.stepDot} />
                <Text style={styles.stepText}>{step}</Text>
              </View>
            ))}
          </View>

          <View style={styles.infoBox}>
            <View style={styles.infoHeader}>
              <FontAwesome name="info-circle" size={15} color={Colors.primary2} />
              <Text style={styles.infoTitle}>Ważne</Text>
            </View>

            <Text style={styles.infoText}>
              Ten wynik ma charakter informacyjny i nie stanowi diagnozy medycznej.
              Model analizuje wyłącznie obraz i może się mylić, szczególnie dla zdjęć
              o słabej jakości lub wykonanych w trudnych warunkach.
            </Text>
          </View>

          <PrimaryButton
            title="Przejdź do historii"
            onPress={() => router.replace("/(tabs)/history")}
            style={{ marginTop: 18 }}
          />

          <Pressable onPress={() => router.back()} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Wróć do aplikacji</Text>
          </Pressable>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    justifyContent: "center",
    padding: 20,
  },
  card: {
    backgroundColor: Colors.surface,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 22,
  },

  topRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 6,
    gap: 12,
  },
  badge: {
    alignSelf: "flex-start",
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
    backgroundColor: "rgba(255,255,255,0.04)",
  },
  badgeText: {
    fontWeight: "900",
    fontSize: 13,
  },
  aiBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: "rgba(38,215,255,0.08)",
    borderWidth: 1,
    borderColor: "rgba(38,215,255,0.16)",
  },
  aiBadgeText: {
    color: Colors.text,
    fontSize: 12,
    fontWeight: "700",
  },

  percent: {
    color: Colors.text,
    fontSize: 56,
    fontWeight: "900",
    marginTop: 14,
    lineHeight: 64,
  },
  subtext: {
    color: Colors.textSecondary,
    marginTop: 6,
    fontSize: 14,
    lineHeight: 20,
  },

  heroBox: {
    marginTop: 18,
    borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 16,
  },
  heroTitle: {
    color: Colors.text,
    fontWeight: "900",
    fontSize: 17,
    marginBottom: 8,
  },
  heroText: {
    color: Colors.textSecondary,
    lineHeight: 22,
    fontSize: 14,
  },

  section: {
    marginTop: 18,
  },
  sectionTitle: {
    color: Colors.text,
    fontWeight: "900",
    fontSize: 16,
    marginBottom: 12,
  },
  stepRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 10,
  },
  stepDot: {
    width: 7,
    height: 7,
    borderRadius: 999,
    backgroundColor: Colors.primary2,
    marginTop: 7,
    marginRight: 10,
  },
  stepText: {
    flex: 1,
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 21,
  },

  infoBox: {
    marginTop: 18,
    backgroundColor: "rgba(38,215,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(38,215,255,0.16)",
    borderRadius: 18,
    padding: 14,
  },
  infoHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  infoTitle: {
    color: Colors.text,
    fontWeight: "800",
    fontSize: 14,
  },
  infoText: {
    color: Colors.textSecondary,
    lineHeight: 20,
    fontSize: 13,
  },

  secondaryButton: {
    marginTop: 12,
    minHeight: 54,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
    alignItems: "center",
    justifyContent: "center",
  },
  secondaryButtonText: {
    color: Colors.textSecondary,
    fontSize: 14,
    fontWeight: "800",
  },
});