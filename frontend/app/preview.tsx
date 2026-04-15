import React, { useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Image,
  Alert,
  ActivityIndicator,
} from "react-native";
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
      } as any);
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
    <LinearGradient colors={[Colors.bg, "#141C33", "#1A2240"]} style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Podgląd zdjęcia</Text>
        <Text style={styles.subtitle}>
          Sprawdź, czy zmiana jest dobrze widoczna, ostra i zajmuje sporą część kadru.
        </Text>
      </View>

      <View style={styles.imageCard}>
        {!!uri && <Image source={{ uri }} style={styles.image} />}
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
          <PrimaryButton title="↩️ Wróć" onPress={() => router.back()} style={{ marginTop: 12 }} />
        </View>
      )}

      <Text style={styles.footer}>
        Wynik ma charakter informacyjny i nie stanowi diagnozy medycznej.
      </Text>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
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