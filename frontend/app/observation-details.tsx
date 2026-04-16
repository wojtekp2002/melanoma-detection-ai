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
import { getObservationById } from "@/db/observations.repository";
import { getLesionById } from "@/db/lesions.repository";
import { Observation } from "@/types/observation";
import { Lesion } from "@/types/lesion";

export default function ObservationDetailsScreen() {
  const params = useLocalSearchParams();
  const observationId = Number(params.id);

  const [loading, setLoading] = useState(true);
  const [observation, setObservation] = useState<Observation | null>(null);
  const [lesion, setLesion] = useState<Lesion | null>(null);

  useEffect(() => {
    loadData();
  }, [observationId]);

  async function loadData() {
    try {
      setLoading(true);

      if (!observationId || Number.isNaN(observationId)) {
        setObservation(null);
        setLesion(null);
        return;
      }

      const observationData = await getObservationById(observationId);
      setObservation(observationData);

      if (observationData?.lesionId) {
        const lesionData = await getLesionById(observationData.lesionId);
        setLesion(lesionData);
      } else {
        setLesion(null);
      }
    } catch (error) {
      console.error("Błąd ładowania szczegółów obserwacji:", error);
    } finally {
      setLoading(false);
    }
  }

  function getRiskTitle(label: Observation["label"], probability: number) {
    if (label === "high_risk") {
      return probability >= 0.7 ? "Podwyższone ryzyko" : "Umiarkowane ryzyko";
    }
    return "Niskie ryzyko";
  }

  function getRiskDescription(label: Observation["label"]) {
    if (label === "high_risk") {
      return "Model wykrył cechy obrazu, które mogą wymagać dalszej konsultacji dermatologicznej.";
    }
    return "Model nie wykrył silnych cech wysokiego ryzyka w tym obrazie, ale wynik nie wyklucza problemu medycznego.";
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <LinearGradient
          colors={[Colors.background, "#0A1627", "#102038"]}
          style={[styles.container, styles.centered]}
        >
          <ActivityIndicator size="large" color={Colors.primary2} />
          <Text style={styles.loadingText}>Wczytywanie obserwacji...</Text>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  if (!observation) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <LinearGradient
          colors={[Colors.background, "#0A1627", "#102038"]}
          style={[styles.container, styles.centered]}
        >
          <Text style={styles.emptyTitle}>Nie znaleziono obserwacji</Text>
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
              <Text style={styles.headerKicker}>Szczegóły analizy</Text>
              <Text style={styles.headerTitle}>
                {getRiskTitle(observation.label, observation.probability)}
              </Text>
            </View>
          </View>

          <View style={styles.imageCard}>
            <Image source={{ uri: observation.imageUri }} style={styles.image} />
          </View>

          <View style={styles.summaryCard}>
            <View style={styles.summaryTop}>
              <View style={styles.riskBadge}>
                <Text style={styles.riskBadgeText}>
                  {observation.label === "high_risk" ? "high_risk" : "low_risk"}
                </Text>
              </View>

              <Text style={styles.summaryDate}>
                {new Date(observation.createdAt).toLocaleDateString("pl-PL")}
              </Text>
            </View>

            <Text style={styles.summaryTitle}>
              {getRiskTitle(observation.label, observation.probability)}
            </Text>

            <Text style={styles.summaryProbability}>
              Szacowane prawdopodobieństwo: {(observation.probability * 100).toFixed(1)}%
            </Text>

            <Text style={styles.summaryDescription}>
              {getRiskDescription(observation.label)}
            </Text>

            {!!observation.note && (
              <View style={styles.noteBox}>
                <Text style={styles.noteBoxTitle}>Notatka użytkownika</Text>
                <Text style={styles.noteBoxText}>{observation.note}</Text>
              </View>
            )}

          </View>

          <View style={styles.infoBlock}>
            <Text style={styles.infoBlockTitle}>Przypisanie do zmiany</Text>

            {lesion ? (
              <View style={styles.linkedCard}>
                <View style={styles.linkedIcon}>
                  <FontAwesome name="map-marker" size={16} color={Colors.primary2} />
                </View>

                <View style={styles.linkedContent}>
                  <Text style={styles.linkedTitle}>{lesion.name}</Text>
                  <Text style={styles.linkedSubtitle}>
                    {lesion.bodySide === "front" ? "Przód ciała" : "Tył ciała"}
                    {lesion.bodyLabel ? ` • ${lesion.bodyLabel}` : ""}
                  </Text>
                </View>
              </View>
            ) : (
              <View style={styles.unlinkedCard}>
                <Text style={styles.unlinkedText}>
                  Ta obserwacja nie została przypisana do konkretnej zmiany skórnej.
                </Text>
              </View>
            )}
          </View>

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="info-circle" size={16} color={Colors.primary2} />
              <Text style={styles.infoTitle}>Ważne</Text>
            </View>

            <Text style={styles.infoText}>
              Wynik ma charakter informacyjny i nie stanowi diagnozy medycznej.
              Jeśli zmiana budzi niepokój, warto skonsultować się z dermatologiem.
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

  imageCard: {
    height: 320,
    borderRadius: 24,
    overflow: "hidden",
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  image: {
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },

  summaryCard: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  summaryTop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  riskBadge: {
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  riskBadgeText: {
    color: Colors.textSecondary,
    fontSize: 11,
    fontWeight: "700",
  },
  summaryDate: {
    color: Colors.textMuted,
    fontSize: 12,
  },
  summaryTitle: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: "900",
    marginBottom: 8,
  },
  summaryProbability: {
    color: Colors.primary2,
    fontSize: 14,
    fontWeight: "800",
    marginBottom: 10,
  },
  summaryDescription: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 22,
  },

  infoBlock: {
    marginBottom: 16,
  },
  infoBlockTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 12,
  },

  linkedCard: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: 20,
    padding: 16,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  linkedIcon: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: "rgba(38,215,255,0.10)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  linkedContent: {
    flex: 1,
  },
  linkedTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
    marginBottom: 4,
  },
  linkedSubtitle: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 18,
  },

  unlinkedCard: {
    borderRadius: 20,
    padding: 16,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  unlinkedText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },

  infoCard: {
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

  emptyTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
    textAlign: "center",
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
  noteBox: {
    marginTop: 14,
    borderRadius: 16,
    padding: 14,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  noteBoxTitle: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: "800",
    marginBottom: 6,
  },
  noteBoxText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },
});