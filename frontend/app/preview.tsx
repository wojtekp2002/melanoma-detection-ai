import React, { useMemo, useState } from "react";
import { View, Text, StyleSheet, Image, Alert, ActivityIndicator } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";
import { predictImage } from "@/services/api";

export default function Preview() {
  const params = useLocalSearchParams();
  const uri = useMemo(() => String(params.uri ?? ""), [params]);

  const [loading, setLoading] = useState(false);

  async function onAnalyze() {
    if (!uri) {
      Alert.alert("Brak zdjęcia", "Nie znaleziono URI zdjęcia.");
      return;
    }

    try {
      setLoading(true);
      const res = await predictImage(uri);

      router.push({
        pathname: "/modal",
        params: {
          probability: String(res.probability),
          prediction: res.label,
        },
      });
    } catch (e: any) {
      Alert.alert(
        "Błąd analizy",
        e?.message ??
          "Nie udało się połączyć z API. Sprawdź czy serwer działa i czy IP jest poprawne."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <LinearGradient colors={[Colors.bg, "#141C33"]} style={styles.container}>
      <Text style={styles.title}>Podgląd</Text>
      <Text style={styles.subtitle}>Jeśli zdjęcie jest nieostre — zrób je ponownie.</Text>

      <View style={styles.imgWrap}>
        {!!uri && <Image source={{ uri }} style={styles.img} />}
      </View>

      {loading ? (
        <View style={{ marginTop: 16 }}>
          <ActivityIndicator />
          <Text style={styles.loadingText}>Analizuję…</Text>
        </View>
      ) : (
        <View style={{ marginTop: 16, gap: 12 }}>
          <PrimaryButton title="🔎 Analizuj" onPress={onAnalyze} />
          <PrimaryButton title="↩️ Wróć" onPress={() => router.back()} />
        </View>
      )}

      <Text style={styles.note}>
        Wynik ma charakter informacyjny i nie jest diagnozą.
      </Text>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  title: { color: Colors.text, fontSize: 24, fontWeight: "900", marginTop: 12 },
  subtitle: { color: Colors.muted, marginTop: 6, lineHeight: 20 },

  imgWrap: {
    marginTop: 16,
    borderRadius: 24,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: Colors.card,
    height: 340,
  },
  img: { width: "100%", height: "100%", resizeMode: "cover" },

  loadingText: { color: Colors.muted, marginTop: 10, textAlign: "center" },
  note: { color: Colors.muted, marginTop: 16, fontSize: 12, lineHeight: 18 },
});
