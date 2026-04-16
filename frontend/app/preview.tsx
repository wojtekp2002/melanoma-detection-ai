import React, { useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Image,
  Alert,
  ActivityIndicator,
  ScrollView,
  Pressable,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";
import { predictImage } from "@/services/api";
import { createObservation } from "@/db/observations.repository";
import { getAllLesions } from "@/db/lesions.repository";
import { Lesion } from "@/types/lesion";
import { FontAwesome } from "@expo/vector-icons";

export default function Preview() {
  const params = useLocalSearchParams();
  const uri = useMemo(() => String(params.uri ?? ""), [params]);

  const [loading, setLoading] = useState(false);
  const [lesionsLoading, setLesionsLoading] = useState(true);
  const [lesions, setLesions] = useState<Lesion[]>([]);
  const [selectedLesionId, setSelectedLesionId] = useState<number | null>(null);

  useEffect(() => {
    loadLesions();
  }, []);

  async function loadLesions() {
    try {
      setLesionsLoading(true);
      const data = await getAllLesions();
      setLesions(data);
    } catch (error) {
      console.error("Błąd ładowania zmian:", error);
    } finally {
      setLesionsLoading(false);
    }
  }

  async function onAnalyze() {
    if (!uri) {
      Alert.alert("Brak zdjęcia", "Nie znaleziono URI zdjęcia.");
      return;
    }

    try {
      setLoading(true);

      const res = await predictImage(uri);

      await createObservation({
        lesionId: selectedLesionId,
        imageUri: uri,
        probability: res.probability,
        label: res.label,
        createdAt: new Date().toISOString(),
        note: null,
      });

      router.push({
        pathname: "/modal",
        params: {
          probability: String(res.probability),
          prediction: res.label,
        },
      } as any);
    } catch (e: any) {
      console.error("Błąd analizy lub zapisu obserwacji:", e);

      Alert.alert(
        "Błąd analizy",
        e?.message ??
          "Nie udało się połączyć z API albo zapisać wyniku. Sprawdź czy serwer działa i czy IP jest poprawne."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <LinearGradient
      colors={[Colors.bg, "#141C33", "#1A2240"]}
      style={styles.container}
    >
      <ScrollView
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <Text style={styles.title}>Podgląd zdjęcia</Text>
          <Text style={styles.subtitle}>
            Sprawdź, czy zmiana jest dobrze widoczna, ostra i zajmuje sporą część kadru.
          </Text>
        </View>

        <View style={styles.imageCard}>
          {!!uri && <Image source={{ uri }} style={styles.image} />}
        </View>

        <View style={styles.assignmentCard}>
          <View style={styles.assignmentHeader}>
            <FontAwesome name="map-marker" size={16} color={Colors.primary2} />
            <Text style={styles.assignmentTitle}>Przypisz analizę do zmiany</Text>
          </View>

          <Text style={styles.assignmentSubtitle}>
            Dzięki temu wynik trafi do historii konkretnej zmiany skórnej.
          </Text>

          <Pressable
            onPress={() => setSelectedLesionId(null)}
            style={[
              styles.optionCard,
              selectedLesionId === null && styles.optionCardActive,
            ]}
          >
            <View style={styles.optionLeft}>
              <View
                style={[
                  styles.optionRadio,
                  selectedLesionId === null && styles.optionRadioActive,
                ]}
              >
                {selectedLesionId === null && (
                  <View style={styles.optionRadioInner} />
                )}
              </View>

              <View>
                <Text style={styles.optionTitle}>Bez przypisania</Text>
                <Text style={styles.optionSubtitle}>
                  Zapisz analizę tylko w ogólnej historii
                </Text>
              </View>
            </View>
          </Pressable>

          {lesionsLoading ? (
            <View style={styles.loadingLesionsBox}>
              <ActivityIndicator size="small" color={Colors.primary2} />
              <Text style={styles.loadingLesionsText}>Wczytywanie zmian...</Text>
            </View>
          ) : lesions.length === 0 ? (
            <View style={styles.emptyLesionsBox}>
              <Text style={styles.emptyLesionsText}>
                Nie masz jeszcze zapisanych zmian. Możesz przejść dalej bez przypisania.
              </Text>
            </View>
          ) : (
            lesions.map((lesion) => {
              const active = selectedLesionId === lesion.id;

              return (
                <Pressable
                  key={lesion.id}
                  onPress={() => setSelectedLesionId(lesion.id ?? null)}
                  style={[styles.optionCard, active && styles.optionCardActive]}
                >
                  <View style={styles.optionLeft}>
                    <View
                      style={[
                        styles.optionRadio,
                        active && styles.optionRadioActive,
                      ]}
                    >
                      {active && <View style={styles.optionRadioInner} />}
                    </View>

                    <View style={styles.optionTextWrap}>
                      <Text style={styles.optionTitle}>{lesion.name}</Text>
                      <Text style={styles.optionSubtitle}>
                        {lesion.bodySide === "front" ? "Przód ciała" : "Tył ciała"}
                        {lesion.bodyLabel ? ` • ${lesion.bodyLabel}` : ""}
                      </Text>
                    </View>
                  </View>
                </Pressable>
              );
            })
          )}
        </View>

        <View style={styles.checklistCard}>
          <Text style={styles.checkTitle}>Checklista przed analizą</Text>
          <Text style={styles.checkItem}>• Zdjęcie jest ostre</Text>
          <Text style={styles.checkItem}>• Zmiana jest w centrum kadru</Text>
          <Text style={styles.checkItem}>• Nie ma mocnych cieni i odbić</Text>
          <Text style={styles.checkItem}>• Zdjęcie jest wykonane z bliska</Text>
        </View>

        {loading ? (
          <View style={styles.loadingBox}>
            <ActivityIndicator />
            <Text style={styles.loadingText}>Analizuję zdjęcie…</Text>
          </View>
        ) : (
          <View style={styles.actions}>
            <PrimaryButton title="🔎 Analizuj" onPress={onAnalyze} />
            <PrimaryButton
              title="↩️ Wróć"
              onPress={() => router.back()}
              style={{ marginTop: 12 }}
            />
          </View>
        )}

        <Text style={styles.footer}>
          Wynik ma charakter informacyjny i nie stanowi diagnozy medycznej.
        </Text>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 20,
    paddingBottom: 36,
  },
  header: {
    marginTop: 12,
    marginBottom: 14,
  },
  title: {
    color: Colors.text,
    fontSize: 28,
    fontWeight: "900",
  },
  subtitle: {
    color: Colors.muted,
    marginTop: 8,
    lineHeight: 21,
  },

  imageCard: {
    height: 320,
    borderRadius: 24,
    overflow: "hidden",
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  image: {
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },

  assignmentCard: {
    marginTop: 16,
    borderRadius: 20,
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 16,
  },
  assignmentHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  assignmentTitle: {
    color: Colors.text,
    fontWeight: "900",
    fontSize: 15,
  },
  assignmentSubtitle: {
    color: Colors.muted,
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 14,
  },

  optionCard: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
    padding: 14,
    marginBottom: 10,
  },
  optionCardActive: {
    borderColor: "rgba(38,215,255,0.35)",
    backgroundColor: "rgba(38,215,255,0.08)",
  },
  optionLeft: {
    flexDirection: "row",
    alignItems: "center",
  },
  optionRadio: {
    width: 20,
    height: 20,
    borderRadius: 999,
    borderWidth: 1.5,
    borderColor: Colors.borderStrong,
    marginRight: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  optionRadioActive: {
    borderColor: Colors.primary2,
  },
  optionRadioInner: {
    width: 10,
    height: 10,
    borderRadius: 999,
    backgroundColor: Colors.primary2,
  },
  optionTextWrap: {
    flex: 1,
  },
  optionTitle: {
    color: Colors.text,
    fontSize: 14,
    fontWeight: "800",
    marginBottom: 4,
  },
  optionSubtitle: {
    color: Colors.muted,
    fontSize: 12,
    lineHeight: 18,
  },

  loadingLesionsBox: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 10,
  },
  loadingLesionsText: {
    marginLeft: 10,
    color: Colors.muted,
    fontSize: 13,
  },
  emptyLesionsBox: {
    paddingVertical: 6,
  },
  emptyLesionsText: {
    color: Colors.muted,
    fontSize: 12,
    lineHeight: 18,
  },

  checklistCard: {
    marginTop: 16,
    borderRadius: 20,
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 16,
  },
  checkTitle: {
    color: Colors.text,
    fontWeight: "900",
    fontSize: 15,
    marginBottom: 10,
  },
  checkItem: {
    color: Colors.muted,
    fontSize: 13,
    lineHeight: 20,
  },

  loadingBox: {
    marginTop: 18,
    alignItems: "center",
  },
  loadingText: {
    color: Colors.muted,
    marginTop: 10,
  },
  actions: {
    marginTop: 18,
  },
  footer: {
    color: Colors.muted,
    textAlign: "center",
    marginTop: 16,
    fontSize: 12,
    lineHeight: 18,
  },
});