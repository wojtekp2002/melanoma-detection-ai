import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  TextInput,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";
import { createLesion } from "@/db/lesions.repository";

export default function AddLesionScreen() {
  const [name, setName] = useState("");
  const [bodySide, setBodySide] = useState<"front" | "back">("front");
  const [bodyLabel, setBodyLabel] = useState("");
  const [saving, setSaving] = useState(false);

  async function handleSave() {
    if (!name.trim()) {
      Alert.alert("Brak nazwy", "Nadaj nazwę zmianie, aby łatwo ją rozpoznać.");
      return;
    }

    try {
      setSaving(true);

      await createLesion({
        name: name.trim(),
        bodySide,
        bodyLabel: bodyLabel.trim() || null,
        x: null,
        y: null,
        createdAt: new Date().toISOString(),
      });

      Alert.alert("Gotowe", "Zmiana została zapisana.");
      router.back();
    } catch (error) {
      console.error("Błąd zapisu zmiany:", error);
      Alert.alert("Błąd", "Nie udało się zapisać zmiany.");
    } finally {
      setSaving(false);
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
          <View style={styles.headerRow}>
            <Pressable onPress={() => router.back()} style={styles.backButton}>
              <FontAwesome name="angle-left" size={20} color={Colors.text} />
            </Pressable>

            <View style={styles.headerTextWrap}>
              <Text style={styles.headerKicker}>Dodawanie zmiany</Text>
              <Text style={styles.headerTitle}>Oznacz nowe miejsce obserwacji</Text>
            </View>
          </View>

          <Text style={styles.description}>
            Nadaj zmianie nazwę i wskaż jej przybliżone położenie. To przygotuje
            bazę pod późniejsze śledzenie zdjęć i analiz dla konkretnej zmiany.
          </Text>

          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Podstawowe informacje</Text>

            <Text style={styles.label}>Nazwa zmiany</Text>
            <TextInput
              value={name}
              onChangeText={setName}
              placeholder="Np. pieprzyk na lewym barku"
              placeholderTextColor={Colors.textMuted}
              style={styles.input}
            />

            <Text style={styles.label}>Opis lokalizacji</Text>
            <TextInput
              value={bodyLabel}
              onChangeText={setBodyLabel}
              placeholder="Np. lewy bark, prawa łydka, kark"
              placeholderTextColor={Colors.textMuted}
              style={styles.input}
            />
          </View>

          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Strona ciała</Text>

            <View style={styles.sideRow}>
              <Pressable
                onPress={() => setBodySide("front")}
                style={[styles.sideButton, bodySide === "front" && styles.sideButtonActive]}
              >
                <FontAwesome
                  name="male"
                  size={28}
                  color={bodySide === "front" ? Colors.primary2 : Colors.textSecondary}
                />
                <Text
                  style={[
                    styles.sideButtonText,
                    bodySide === "front" && styles.sideButtonTextActive,
                  ]}
                >
                  Przód
                </Text>
              </Pressable>

              <Pressable
                onPress={() => setBodySide("back")}
                style={[styles.sideButton, bodySide === "back" && styles.sideButtonActive]}
              >
                <FontAwesome
                  name="male"
                  size={28}
                  color={bodySide === "back" ? Colors.primary2 : Colors.textSecondary}
                />
                <Text
                  style={[
                    styles.sideButtonText,
                    bodySide === "back" && styles.sideButtonTextActive,
                  ]}
                >
                  Tył
                </Text>
              </Pressable>
            </View>
          </View>

          <View style={styles.bodyMapCard}>
            <View style={styles.bodyMapHeader}>
              <FontAwesome name="crosshairs" size={16} color={Colors.primary2} />
              <Text style={styles.bodyMapTitle}>Mapa ciała — kolejny krok</Text>
            </View>

            <View style={styles.bodyPlaceholder}>
              <FontAwesome name="male" size={90} color="rgba(255,255,255,0.14)" />
              <Text style={styles.bodyPlaceholderText}>
                Tutaj dodamy interaktywną sylwetkę przód / tył, aby zaznaczać
                dokładne położenie zmian na ciele.
              </Text>
            </View>
          </View>

          <PrimaryButton
            title={saving ? "Zapisywanie..." : "Zapisz zmianę"}
            onPress={handleSave}
            disabled={saving}
            style={{ marginTop: 18 }}
          />
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
    paddingBottom: 36,
  },

  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
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
  description: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 18,
  },

  card: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 16,
  },
  label: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 8,
    marginTop: 4,
  },
  input: {
    minHeight: 54,
    borderRadius: 18,
    paddingHorizontal: 16,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
    color: Colors.text,
    fontSize: 15,
    marginBottom: 14,
  },

  sideRow: {
    flexDirection: "row",
    gap: 12,
  },
  sideButton: {
    flex: 1,
    minHeight: 110,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  sideButtonActive: {
    backgroundColor: "rgba(38,215,255,0.08)",
    borderColor: "rgba(38,215,255,0.30)",
  },
  sideButtonText: {
    color: Colors.textSecondary,
    fontSize: 14,
    fontWeight: "800",
  },
  sideButtonTextActive: {
    color: Colors.primary2,
  },

  bodyMapCard: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  bodyMapHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 14,
  },
  bodyMapTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
  },
  bodyPlaceholder: {
    minHeight: 260,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  bodyPlaceholderText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
    textAlign: "center",
    marginTop: 14,
    maxWidth: 260,
  },
});