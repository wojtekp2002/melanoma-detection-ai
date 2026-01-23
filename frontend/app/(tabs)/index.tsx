import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import PrimaryButton from "@/components/PrimaryButton";
import { Colors } from "@/constants/Colors";

export default function HomeTab() {
  return (
    <LinearGradient colors={[Colors.bg, "#141C33"]} style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Analiza zdjęcia</Text>
        <Text style={styles.subtitle}>
          W następnym kroku dodamy: galerię, aparat, kadrowanie i wysyłkę do API.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Test UI</Text>
        <Text style={styles.desc}>
          Kliknij, aby zobaczyć ekran wyniku (na razie fake).
        </Text>

        <PrimaryButton
          title="Pokaż wynik (test)"
          onPress={() =>
            router.push({
              pathname: "/modal",
              params: { probability: "0.82", prediction: "high_risk" },
            })
          }
          style={{ marginTop: 14 }}
        />
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  header: { marginTop: 12, marginBottom: 14 },
  title: { color: Colors.text, fontSize: 26, fontWeight: "900" },
  subtitle: { color: Colors.muted, marginTop: 6, lineHeight: 20 },
  card: {
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 24,
    padding: 18,
  },
  cardTitle: { color: Colors.text, fontSize: 16, fontWeight: "900" },
  desc: { color: Colors.muted, marginTop: 10, lineHeight: 22 },
});
