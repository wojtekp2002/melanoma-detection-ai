import React, { useState } from "react";
import { View, Text, StyleSheet, Pressable } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { Colors } from "../constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

export default function Disclaimer() {
  const [accepted, setAccepted] = useState(false);

  return (
    <LinearGradient colors={[Colors.bg, "#141C33"]} style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.title}>Ważna informacja</Text>
        <Text style={styles.desc}>
          To narzędzie nie jest poradą lekarską ani diagnozą. Wynik może być
          błędny. Jeśli zmiana skórna budzi niepokój — skontaktuj się z
          dermatologiem.
        </Text>

        <View style={styles.bullets}>
          <Text style={styles.bullet}>• To screening edukacyjny.</Text>
          <Text style={styles.bullet}>• Jakość zdjęcia wpływa na wynik.</Text>
          <Text style={styles.bullet}>
            • Najpewniejsza ocena: dermatolog + dermatoskopia.
          </Text>
        </View>

        <Pressable
          onPress={() => setAccepted((v) => !v)}
          style={[styles.row, accepted && styles.rowOn]}
        >
          <View style={[styles.box, accepted && styles.boxOn]} />
          <Text style={styles.rowText}>Rozumiem i chcę przejść dalej</Text>
        </Pressable>

        <PrimaryButton
          title="Przejdź do analizy"
          disabled={!accepted}
          onPress={() => router.replace("/(tabs)")}
          style={{ marginTop: 14 }}
        />
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
  title: { color: Colors.text, fontSize: 22, fontWeight: "900" },
  desc: { color: Colors.muted, marginTop: 10, lineHeight: 22, fontSize: 15 },
  bullets: { marginTop: 14, gap: 8 },
  bullet: { color: Colors.muted, lineHeight: 20 },
  row: {
    marginTop: 16,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    padding: 12,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
  },
  rowOn: {
    borderColor: "rgba(38,215,255,0.35)",
    backgroundColor: "rgba(38,215,255,0.08)",
  },
  box: {
    width: 18,
    height: 18,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  boxOn: { backgroundColor: Colors.primary2, borderColor: Colors.primary2 },
  rowText: { color: Colors.text, fontWeight: "800" },
});
