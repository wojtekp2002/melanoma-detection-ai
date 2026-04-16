import React from "react";
import { View, Text, StyleSheet, Pressable } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";
import { FontAwesome } from "@expo/vector-icons";
import { Colors } from "../constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function Welcome() {
  return (
    <SafeAreaView style={styles.safeArea}>
      <LinearGradient
        colors={[Colors.background, "#0A1627", "#102038"]}
        style={styles.container}
      >
        <View style={styles.content}>
          <View style={styles.topGlow} />

          <View style={styles.heroCard}>
            <View style={styles.badge}>
              <FontAwesome name="shield" size={13} color={Colors.primary2} />
              <Text style={styles.badgeText}>AI wspierające monitorowanie zmian</Text>
            </View>

            <Text style={styles.title}>
              Nowoczesna ocena ryzyka zmian skórnych
            </Text>

            <Text style={styles.description}>
              Zrób zdjęcie lub wybierz je z galerii. Aplikacja przeanalizuje obraz,
              oszacuje poziom ryzyka i pomoże Ci śledzić obserwacje w czasie.
            </Text>

            <View style={styles.statsRow}>
              <View style={styles.statBox}>
                <Text style={styles.statValue}>AI</Text>
                <Text style={styles.statLabel}>Analiza obrazu</Text>
              </View>

              <View style={styles.statBox}>
                <Text style={styles.statValue}>Fast</Text>
                <Text style={styles.statLabel}>Szybki wynik</Text>
              </View>

              <View style={styles.statBox}>
                <Text style={styles.statValue}>Safe</Text>
                <Text style={styles.statLabel}>Nie diagnoza</Text>
              </View>
            </View>
          </View>

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="stethoscope" size={16} color={Colors.accent} />
              <Text style={styles.infoTitle}>Do czego służy aplikacja</Text>
            </View>

            <Text style={styles.infoText}>
              To narzędzie pomaga we wstępnej ocenie ryzyka i monitorowaniu zmian
              skórnych, ale nie zastępuje konsultacji z dermatologiem.
            </Text>
          </View>

          <View style={styles.actions}>
            <PrimaryButton
              title="Zaloguj się"
              onPress={() => router.push("/login")}
            />

            <Pressable
              style={styles.secondaryButton}
              onPress={() => router.push("/register")}
            >
              <Text style={styles.secondaryButtonText}>Załóż konto</Text>
            </Pressable>

            <Pressable onPress={() => router.replace("/disclaimer")}>
              <Text style={styles.skipText}>Kontynuuj bez logowania</Text>
            </Pressable>
          </View>

          <Text style={styles.footerNote}>
            Wyniki mają charakter informacyjny i nie stanowią diagnozy medycznej.
          </Text>
        </View>
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
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 28,
    justifyContent: "space-between",
  },
  topGlow: {
    position: "absolute",
    top: 40,
    right: -40,
    width: 180,
    height: 180,
    borderRadius: 999,
    backgroundColor: "rgba(38,215,255,0.08)",
  },

  heroCard: {
    marginTop: 24,
    borderRadius: 30,
    padding: 24,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.borderStrong,
    overflow: "hidden",
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
    marginBottom: 18,
  },
  badgeText: {
    color: Colors.text,
    fontSize: 12,
    fontWeight: "700",
  },
  title: {
    color: Colors.text,
    fontSize: 34,
    lineHeight: 41,
    fontWeight: "900",
    marginBottom: 12,
  },
  description: {
    color: Colors.textSecondary,
    fontSize: 15,
    lineHeight: 24,
    marginBottom: 22,
  },

  statsRow: {
    flexDirection: "row",
    gap: 10,
  },
  statBox: {
    flex: 1,
    borderRadius: 18,
    paddingVertical: 14,
    paddingHorizontal: 12,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
  },
  statValue: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "900",
    marginBottom: 4,
  },
  statLabel: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },

  infoCard: {
    marginTop: 18,
    borderRadius: 24,
    padding: 18,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
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
    fontSize: 14,
    lineHeight: 22,
  },

  actions: {
    marginTop: 24,
  },
  secondaryButton: {
    marginTop: 12,
    minHeight: 56,
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
    fontWeight: "700",
  },

  footerNote: {
    marginTop: 18,
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 18,
    textAlign: "center",
  },

  skipText: {
    marginTop: 16,
    textAlign: "center",
    color: Colors.textMuted,
    fontSize: 13,
  },
});