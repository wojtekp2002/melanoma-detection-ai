import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
  Image,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { router, useLocalSearchParams } from "expo-router";
import { Colors } from "@/constants/Colors";
import { getLesionById } from "@/db/lesions.repository";
import { getObservationsByLesionId } from "@/db/observations.repository";
import { Lesion } from "@/types/lesion";
import { Observation } from "@/types/observation";

export default function LesionDetailsScreen() {
  const params = useLocalSearchParams();
  const lesionId = Number(params.id);

  const [loading, setLoading] = useState(true);
  const [lesion, setLesion] = useState<Lesion | null>(null);
  const [observations, setObservations] = useState<Observation[]>([]);

  useEffect(() => {
    loadData();
  }, [lesionId]);

  async function loadData() {
    try {
      setLoading(true);

      if (!lesionId || Number.isNaN(lesionId)) {
        setLesion(null);
        setObservations([]);
        return;
      }

      const lesionData = await getLesionById(lesionId);
      const observationsData = await getObservationsByLesionId(lesionId);

      setLesion(lesionData);
      setObservations(observationsData);
    } catch (error) {
      console.error("Błąd ładowania szczegółów zmiany:", error);
    } finally {
      setLoading(false);
    }
  }

  function getRiskText(label: Observation["label"], probability: number) {
    if (label === "high_risk") {
      return probability >= 0.7 ? "Podwyższone ryzyko" : "Umiarkowane ryzyko";
    }
    return "Niskie ryzyko";
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <LinearGradient
          colors={[Colors.background, "#0A1627", "#102038"]}
          style={[styles.container, styles.centered]}
        >
          <ActivityIndicator size="large" color={Colors.primary2} />
          <Text style={styles.loadingText}>Wczytywanie szczegółów...</Text>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  if (!lesion) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <LinearGradient
          colors={[Colors.background, "#0A1627", "#102038"]}
          style={[styles.container, styles.centered]}
        >
          <Text style={styles.emptyTitle}>Nie znaleziono zmiany</Text>
          <Pressable onPress={() => router.back()} style={styles.backSimpleButton}>
            <Text style={styles.backSimpleButtonText}>Wróć</Text>
          </Pressable>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <LinearGradient
        colors={[Colors.background, "#0A1627", "#102038"]}
        style={styles.container}
      >
        <ScrollView
          contentContainerStyle={styles.content}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.headerRow}>
            <Pressable onPress={() => router.back()} style={styles.backButton}>
              <FontAwesome name="angle-left" size={20} color={Colors.text} />
            </Pressable>

            <View style={styles.headerTextWrap}>
              <Text style={styles.headerKicker}>Szczegóły zmiany</Text>
              <Text style={styles.headerTitle}>{lesion.name}</Text>
            </View>
          </View>

          <View style={styles.heroCard}>
            <View style={styles.heroIcon}>
              <FontAwesome name="map-marker" size={20} color={Colors.primary2} />
            </View>

            <View style={styles.heroContent}>
              <Text style={styles.heroTitle}>{lesion.name}</Text>
              <Text style={styles.heroSubtitle}>
                {lesion.bodySide === "front" ? "Przód ciała" : "Tył ciała"}
                {lesion.bodyLabel ? ` • ${lesion.bodyLabel}` : ""}
              </Text>
              <Text style={styles.heroDate}>
                Dodano: {new Date(lesion.createdAt).toLocaleDateString("pl-PL")}
              </Text>
            </View>
          </View>

          <View style={styles.summaryRow}>
            <View style={styles.summaryCard}>
              <Text style={styles.summaryValue}>{observations.length}</Text>
              <Text style={styles.summaryLabel}>Obserwacje</Text>
            </View>

            <View style={styles.summaryCard}>
              <Text style={styles.summaryValue}>
                {observations.length > 0
                  ? `${(observations[0].probability * 100).toFixed(0)}%`
                  : "—"}
              </Text>
              <Text style={styles.summaryLabel}>Ostatni wynik</Text>
            </View>
          </View>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Historia tej zmiany</Text>
          </View>

          {observations.length === 0 ? (
            <View style={styles.emptyCard}>
              <FontAwesome name="camera" size={28} color={Colors.textMuted} />
              <Text style={styles.emptyTitle}>Brak przypisanych analiz</Text>
              <Text style={styles.emptyText}>
                W kolejnym kroku podepniemy możliwość przypisywania nowych analiz
                bezpośrednio do tej zmiany.
              </Text>
            </View>
          ) : (
            observations.map((item) => (
              <View key={item.id} style={styles.observationCard}>
                <Image source={{ uri: item.imageUri }} style={styles.observationImage} />

                <View style={styles.observationContent}>
                  <Text style={styles.observationDate}>
                    {new Date(item.createdAt).toLocaleDateString("pl-PL")}
                  </Text>
                  <Text style={styles.observationTitle}>
                    {getRiskText(item.label, item.probability)}
                  </Text>
                  <Text style={styles.observationSubtitle}>
                    Szacowane prawdopodobieństwo: {(item.probability * 100).toFixed(1)}%
                  </Text>

                  <View style={styles.badgeSmall}>
                    <Text style={styles.badgeSmallText}>
                      {item.label === "high_risk" ? "high_risk" : "low_risk"}
                    </Text>
                  </View>
                </View>
              </View>
            ))
          )}

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="lightbulb-o" size={16} color={Colors.warning} />
              <Text style={styles.infoTitle}>Co dalej</Text>
            </View>

            <Text style={styles.infoText}>
              Następny krok to przypisywanie nowej analizy bezpośrednio do tej zmiany,
              aby tracking był pełny i naprawdę użyteczny.
            </Text>
          </View>
        </ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  container: {
    flex: 1,
  },
  centered: {
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  content: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 36,
  },
  loadingText: {
    marginTop: 12,
    color: Colors.textSecondary,
    fontSize: 14,
  },

  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 18,
  },
  backButton: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: Colors.border,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  headerTextWrap: {
    flex: 1,
  },
  headerKicker: {
    color: Colors.primary2,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.6,
    marginBottom: 4,
  },
  headerTitle: {
    color: Colors.text,
    fontSize: 24,
    fontWeight: "900",
  },

  heroCard: {
    flexDirection: "row",
    borderRadius: 24,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  heroIcon: {
    width: 54,
    height: 54,
    borderRadius: 18,
    backgroundColor: "rgba(38,215,255,0.10)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 14,
  },
  heroContent: {
    flex: 1,
    justifyContent: "center",
  },
  heroTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
    marginBottom: 6,
  },
  heroSubtitle: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 6,
  },
  heroDate: {
    color: Colors.textMuted,
    fontSize: 12,
  },

  summaryRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 20,
  },
  summaryCard: {
    flex: 1,
    borderRadius: 22,
    padding: 16,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  summaryValue: {
    color: Colors.text,
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 4,
  },
  summaryLabel: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },

  sectionHeader: {
    marginBottom: 14,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
  },

  emptyCard: {
    alignItems: "center",
    borderRadius: 24,
    padding: 24,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 14,
  },
  emptyTitle: {
    color: Colors.text,
    fontSize: 17,
    fontWeight: "800",
    marginTop: 12,
    marginBottom: 8,
    textAlign: "center",
  },
  emptyText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
    textAlign: "center",
  },

  observationCard: {
    flexDirection: "row",
    backgroundColor: Colors.surface,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
    marginBottom: 14,
  },
  observationImage: {
    width: 96,
    height: 120,
  },
  observationContent: {
    flex: 1,
    padding: 14,
    justifyContent: "center",
  },
  observationDate: {
    color: Colors.primary2,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.6,
    marginBottom: 6,
  },
  observationTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 6,
  },
  observationSubtitle: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 12,
  },

  badgeSmall: {
    alignSelf: "flex-start",
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  badgeSmallText: {
    color: Colors.textSecondary,
    fontSize: 11,
    fontWeight: "700",
  },

  infoCard: {
    marginTop: 8,
    borderRadius: 22,
    padding: 16,
    backgroundColor: "rgba(38,215,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(38,215,255,0.16)",
  },
  infoHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  infoTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
  },
  infoText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },

  backSimpleButton: {
    marginTop: 16,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.08)",
  },
  backSimpleButtonText: {
    color: Colors.text,
    fontWeight: "800",
  },
});