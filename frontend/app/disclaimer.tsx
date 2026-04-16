import React, { useState } from "react";
import { View, Text, StyleSheet, Pressable, ScrollView } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { FontAwesome } from "@expo/vector-icons";
import { Colors } from "../constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function Disclaimer() {
  const [accepted, setAccepted] = useState(false);

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
            <View style={styles.headerBadge}>
              <FontAwesome name="info-circle" size={14} color={Colors.warning} />
              <Text style={styles.headerBadgeText}>Ważna informacja</Text>
            </View>

            <Text style={styles.title}>Przeczytaj przed rozpoczęciem analizy</Text>

            <Text style={styles.description}>
              To narzędzie służy do wstępnej oceny ryzyka na podstawie zdjęcia.
              Nie jest poradą lekarską, diagnozą ani zamiennikiem wizyty u specjalisty.
            </Text>
          </View>

          <View style={styles.card}>
            <View style={styles.sectionHeader}>
              <FontAwesome name="shield" size={16} color={Colors.primary2} />
              <Text style={styles.sectionTitle}>O czym trzeba pamiętać</Text>
            </View>

            <View style={styles.item}>
              <View style={styles.dot} />
              <Text style={styles.itemText}>
                Wynik analizy może być błędny i nie daje pewności medycznej.
              </Text>
            </View>

            <View style={styles.item}>
              <View style={styles.dot} />
              <Text style={styles.itemText}>
                Jakość zdjęcia ma bardzo duży wpływ na końcowy rezultat.
              </Text>
            </View>

            <View style={styles.item}>
              <View style={styles.dot} />
              <Text style={styles.itemText}>
                Jeśli zmiana rośnie, krwawi, swędzi, zmienia kolor lub kształt,
                skonsultuj się z dermatologiem.
              </Text>
            </View>

            <View style={styles.item}>
              <View style={styles.dot} />
              <Text style={styles.itemText}>
                Najbardziej wiarygodna ocena wymaga badania dermatologicznego,
                najlepiej z dermatoskopią.
              </Text>
            </View>
          </View>

          <Pressable
            onPress={() => setAccepted((prev) => !prev)}
            style={[styles.checkboxCard, accepted && styles.checkboxCardActive]}
          >
            <View style={[styles.checkbox, accepted && styles.checkboxActive]}>
              {accepted && (
                <FontAwesome name="check" size={11} color={Colors.background} />
              )}
            </View>

            <View style={styles.checkboxTextWrap}>
              <Text style={styles.checkboxTitle}>Rozumiem ograniczenia aplikacji</Text>
              <Text style={styles.checkboxText}>
                Potwierdzam, że wynik ma charakter informacyjny i nie stanowi diagnozy.
              </Text>
            </View>
          </Pressable>

          <PrimaryButton
            title="Przejdź do aplikacji"
            disabled={!accepted}
            onPress={() => router.replace("/(tabs)")}
            style={{ marginTop: 18 }}
          />

          <Text style={styles.footer}>
            Korzystaj z aplikacji odpowiedzialnie i traktuj ją jako wsparcie,
            a nie ostateczną ocenę medyczną.
          </Text>
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
    paddingTop: 18,
    paddingBottom: 36,
  },

  header: {
    marginBottom: 20,
  },
  headerBadge: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: "rgba(255,184,77,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,184,77,0.18)",
    marginBottom: 16,
  },
  headerBadgeText: {
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

  card: {
    borderRadius: 26,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 18,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 14,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
  },

  item: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 14,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 999,
    backgroundColor: Colors.primary2,
    marginTop: 7,
    marginRight: 10,
  },
  itemText: {
    flex: 1,
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 22,
  },

  checkboxCard: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
  },
  checkboxCardActive: {
    borderColor: "rgba(38,215,255,0.30)",
    backgroundColor: "rgba(38,215,255,0.08)",
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors.borderStrong,
    backgroundColor: "rgba(255,255,255,0.03)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 14,
  },
  checkboxActive: {
    backgroundColor: Colors.primary2,
    borderColor: Colors.primary2,
  },
  checkboxTextWrap: {
    flex: 1,
  },
  checkboxTitle: {
    color: Colors.text,
    fontSize: 14,
    fontWeight: "800",
    marginBottom: 4,
  },
  checkboxText: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 18,
  },

  footer: {
    marginTop: 16,
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 18,
    textAlign: "center",
  },
});