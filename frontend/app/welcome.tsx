import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { Colors } from "../constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function Welcome() {
  return (
    <LinearGradient colors={[Colors.bg, "#141C33"]} style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.kicker}>Melanoma AI</Text>
        <Text style={styles.title}>Szybka ocena ryzyka zmiany skórnej</Text>
        <Text style={styles.desc}>
          Zrób zdjęcie lub wybierz je z galerii. Otrzymasz wynik w procentach i
          krótką interpretację.
        </Text>

        <PrimaryButton
          title="Zaczynamy"
          onPress={() => router.replace("/disclaimer")}
          style={{ marginTop: 16 }}
        />

        <Text style={styles.note}>
          Projekt edukacyjny • Wynik ma charakter informacyjny
        </Text>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, justifyContent: "center" },
  card: {
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 24,
    padding: 20,
  },
  kicker: { color: Colors.primary2, fontWeight: "900", letterSpacing: 0.7 },
  title: { color: Colors.text, fontSize: 28, fontWeight: "900", marginTop: 8 },
  desc: { color: Colors.muted, marginTop: 10, lineHeight: 22, fontSize: 15 },
  note: { color: Colors.muted, marginTop: 12, fontSize: 12 },
});
