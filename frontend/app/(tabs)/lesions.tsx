import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { router, useFocusEffect } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";
import { getAllLesions } from "@/db/lesions.repository";
import { Lesion } from "@/types/lesion";


export default function LesionsScreen() {
  const [lesions, setLesions] = useState<Lesion[]>([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      loadLesions();
    }, [])
  );

  async function loadLesions() {
    try {
      setLoading(true);
      const data = await getAllLesions();
      setLesions(data);
    } catch (error) {
      console.error("Błąd ładowania zmian:", error);
    } finally {
      setLoading(false);
    }
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
              <FontAwesome name="map-marker" size={14} color={Colors.primary2} />
              <Text style={styles.badgeText}>Moje zmiany</Text>
            </View>

            <Text style={styles.title}>Monitoruj konkretne zmiany skórne</Text>
            <Text style={styles.description}>
              Przypisuj zdjęcia do konkretnych miejsc na ciele i buduj historię
              obserwacji dla każdej zmiany osobno.
            </Text>
          </View>

          <View style={styles.topCard}>
            <View style={styles.topCardIcon}>
              <FontAwesome name="male" size={34} color={Colors.primary2} />
            </View>

            <View style={styles.topCardText}>
              <Text style={styles.topCardTitle}>Dodaj nową zmianę</Text>
              <Text style={styles.topCardDescription}>
                Wskaż miejsce na sylwetce, nazwij zmianę i przygotuj ją do dalszego śledzenia.
              </Text>
            </View>
          </View>

          <PrimaryButton
            title="Dodaj zmianę"
            onPress={() => router.push({ pathname: "/add-lesion" } as any)}
            style={{ marginBottom: 22 }}
          />

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Zapisane zmiany</Text>
            <Text style={styles.sectionCount}>{lesions.length}</Text>
          </View>

          {loading ? (
            <View style={styles.centerBox}>
              <ActivityIndicator size="large" color={Colors.primary2} />
              <Text style={styles.centerText}>Wczytywanie zmian...</Text>
            </View>
          ) : lesions.length === 0 ? (
            <View style={styles.emptyCard}>
              <FontAwesome name="map-o" size={28} color={Colors.textMuted} />
              <Text style={styles.emptyTitle}>Brak zapisanych zmian</Text>
              <Text style={styles.emptyText}>
                Dodaj pierwszą zmianę, aby zacząć budować historię obserwacji.
              </Text>
            </View>
          ) : (
            lesions.map((item) => (
              <Pressable 
                key={item.id}
                style={styles.card}
                onPress={() =>
                  router.push({
                    pathname: "/lesion-details",
                    params: { id: String(item.id) },
                  } as any)
                }
              >
                <View style={styles.cardIcon}>
                  <FontAwesome name="circle" size={14} color={Colors.warning} />
                </View>

                <View style={styles.cardContent}>
                  <Text style={styles.cardTitle}>{item.name}</Text>
                  <Text style={styles.cardLocation}>
                    {item.bodySide === "front" ? "Przód ciała" : "Tył ciała"}
                    {item.bodyLabel ? ` • ${item.bodyLabel}` : ""}
                  </Text>

                  <View style={styles.metaRow}>
                    <View style={styles.metaBadge}>
                      <Text style={styles.metaBadgeText}>Gotowa do obserwacji</Text>
                    </View>
                  </View>

                  <Text style={styles.cardDate}>
                    Dodano: {new Date(item.createdAt).toLocaleDateString("pl-PL")}
                  </Text>
                </View>

                <FontAwesome
                  name="angle-right"
                  size={20}
                  color={Colors.textSecondary}
                />
              </Pressable>
            ))
          )}

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="lightbulb-o" size={16} color={Colors.warning} />
              <Text style={styles.infoTitle}>Po co to robimy</Text>
            </View>

            <Text style={styles.infoText}>
              Dzięki przypisaniu analiz do konkretnych zmian użytkownik nie gubi się
              w historii zdjęć i może porównywać obserwacje tej samej zmiany w czasie.
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

  topCard: {
    flexDirection: "row",
    alignItems: "center",
    borderRadius: 24,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  topCardIcon: {
    width: 58,
    height: 58,
    borderRadius: 18,
    backgroundColor: "rgba(38,215,255,0.10)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 14,
  },
  topCardText: {
    flex: 1,
  },
  topCardTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 6,
  },
  topCardDescription: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },

  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 14,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
  },
  sectionCount: {
    color: Colors.primary2,
    fontSize: 14,
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
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 16,
    marginBottom: 14,
  },
  cardIcon: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: "rgba(255,184,77,0.10)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  cardContent: {
    flex: 1,
    marginRight: 10,
  },
  cardTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
    marginBottom: 4,
  },
  cardLocation: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 10,
  },
  metaRow: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 8,
    flexWrap: "wrap",
  },
  metaBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  metaBadgeText: {
    color: Colors.textSecondary,
    fontSize: 11,
    fontWeight: "700",
  },
  cardDate: {
    color: Colors.textMuted,
    fontSize: 12,
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