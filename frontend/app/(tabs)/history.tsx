import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Pressable,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { router, useFocusEffect } from "expo-router";
import { Colors } from "@/constants/Colors";
import { getAllObservations } from "@/db/observations.repository";
import { Observation } from "@/types/observation";

export default function HistoryScreen() {
  const [observations, setObservations] = useState<Observation[]>([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      loadObservations();
    }, [])
  );

  async function loadObservations() {
    try {
      setLoading(true);
      const data = await getAllObservations();
      setObservations(data);
    } catch (error) {
      console.error("Błąd ładowania obserwacji:", error);
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
          <View style={styles.header}>
            <View style={styles.badge}>
              <FontAwesome name="clock-o" size={14} color={Colors.primary2} />
              <Text style={styles.badgeText}>Historia analiz</Text>
            </View>

            <Text style={styles.title}>Twoje wcześniejsze wyniki</Text>
            <Text style={styles.description}>
              Przeglądaj zapisane analizy, porównuj wcześniejsze obserwacje i
              buduj historię monitorowania zmian skórnych.
            </Text>
          </View>

          <View style={styles.summaryCard}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>{observations.length}</Text>
              <Text style={styles.summaryLabel}>Wszystkich analiz</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {
                  observations.filter((item) => {
                    const d = new Date(item.createdAt);
                    const now = new Date();
                    return (
                      d.getMonth() === now.getMonth() &&
                      d.getFullYear() === now.getFullYear()
                    );
                  }).length
                }
              </Text>
              <Text style={styles.summaryLabel}>W tym miesiącu</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>
                {observations.filter((item) => item.label === "high_risk").length}
              </Text>
              <Text style={styles.summaryLabel}>High risk</Text>
            </View>
          </View>

          <View style={styles.listHeader}>
            <Text style={styles.listTitle}>Ostatnie obserwacje</Text>
          </View>

          {loading ? (
            <View style={styles.centerBox}>
              <ActivityIndicator size="large" color={Colors.primary2} />
              <Text style={styles.centerText}>Wczytywanie historii...</Text>
            </View>
          ) : observations.length === 0 ? (
            <View style={styles.emptyCard}>
              <FontAwesome name="history" size={28} color={Colors.textMuted} />
              <Text style={styles.emptyTitle}>Brak zapisanych analiz</Text>
              <Text style={styles.emptyText}>
                Gdy wykonasz analizę zdjęcia, wynik pojawi się tutaj automatycznie.
              </Text>
            </View>
          ) : (
            observations.map((item) => (
              <Pressable 
              key={item.id} 
              style={styles.card}
                onPress={() =>
                router.push({
                  pathname: "/observation-details",
                  params: { id: String(item.id) },
                } as any)
              }
              >
                <Image source={{ uri: item.imageUri }} style={styles.image} />

                <View style={styles.cardContent}>
                  <Text style={styles.cardDate}>
                    {new Date(item.createdAt).toLocaleDateString("pl-PL")}
                  </Text>
                  <Text style={styles.cardTitle}>
                    {getRiskText(item.label, item.probability)}
                  </Text>
                  <Text style={styles.cardSubtitle}>
                    Szacowane prawdopodobieństwo: {(item.probability * 100).toFixed(1)}%
                  </Text>

                  <View style={styles.cardFooter}>
                    <View style={styles.badgeSmall}>
                      <Text style={styles.badgeSmallText}>
                        {item.label === "high_risk" ? "high_risk" : "low_risk"}
                      </Text>
                    </View>

                    <FontAwesome
                      name="angle-right"
                      size={18}
                      color={Colors.textSecondary}
                    />
                  </View>
                </View>
              </Pressable>
            ))
          )}

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="info-circle" size={16} color={Colors.primary2} />
              <Text style={styles.infoTitle}>Co dalej</Text>
            </View>

            <Text style={styles.infoText}>
              W kolejnym kroku podepniemy przypisywanie analizy do konkretnej zmiany
              skórnej, żeby historia była jeszcze bardziej użyteczna.
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
  content: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 120,
  },

  header: {
    marginBottom: 20,
  },
  badge: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
    marginBottom: 16,
  },
  badgeText: {
    color: Colors.text,
    fontSize: 12,
    fontWeight: "700",
  },
  title: {
    color: Colors.text,
    fontSize: 30,
    lineHeight: 37,
    fontWeight: "900",
    marginBottom: 10,
  },
  description: {
    color: Colors.textSecondary,
    fontSize: 15,
    lineHeight: 24,
  },

  summaryCard: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 22,
  },
  summaryItem: {
    flex: 1,
    borderRadius: 20,
    paddingVertical: 16,
    paddingHorizontal: 12,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  summaryValue: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: "900",
    marginBottom: 4,
  },
  summaryLabel: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },

  listHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 14,
  },
  listTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
  },

  centerBox: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 40,
  },
  centerText: {
    marginTop: 12,
    color: Colors.textSecondary,
    fontSize: 14,
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
  },
  emptyText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
    textAlign: "center",
  },

  card: {
    flexDirection: "row",
    backgroundColor: Colors.surface,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
    marginBottom: 14,
  },
  image: {
    width: 96,
    height: 120,
  },
  cardContent: {
    flex: 1,
    padding: 14,
    justifyContent: "center",
  },
  cardDate: {
    color: Colors.primary2,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.6,
    marginBottom: 6,
  },
  cardTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 6,
  },
  cardSubtitle: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 12,
  },
  cardFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  badgeSmall: {
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
});