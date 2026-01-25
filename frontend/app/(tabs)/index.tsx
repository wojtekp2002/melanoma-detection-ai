import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, Alert, Image } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";

export default function HomeTab() {
  const [lastUri, setLastUri] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      await ImagePicker.requestMediaLibraryPermissionsAsync();
      await Camera.requestCameraPermissionsAsync();
    })();
  }, []);

  async function pickFromGallery() {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert("Brak uprawnień", "Nadaj dostęp do galerii w ustawieniach.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
      allowsEditing: true, 
      aspect: [1, 1], 
    });

    if (result.canceled) return;

    const uri = result.assets[0].uri;
    setLastUri(uri);
    router.push({ pathname: "/preview", params: { uri } } as any);
  }

  async function takePhoto() {
    const perm = await Camera.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert("Brak uprawnień", "Nadaj dostęp do aparatu w ustawieniach.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.9,
      allowsEditing: true,
      aspect: [1, 1],
    });

    if (result.canceled) return;

    const uri = result.assets[0].uri;
    setLastUri(uri);
    router.push({ pathname: "/preview", params: { uri } } as any );
  }

  return (
    <LinearGradient colors={[Colors.bg, "#141C33"]} style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Analiza zdjęcia</Text>
        <Text style={styles.subtitle}>
          Wybierz zdjęcie zmiany skórnej. Najlepiej w dobrym świetle, ostro, bez
          mocnych cieni.
        </Text>
      </View>

      {lastUri && (
        <View style={styles.previewBox}>
          <Image source={{ uri: lastUri }} style={styles.previewImg} />
          <Text style={styles.previewText}>Ostatnio wybrane</Text>
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Wybierz źródło</Text>

        <PrimaryButton
          title="📷 Zrób zdjęcie"
          onPress={takePhoto}
          style={{ marginTop: 12 }}
        />
        <PrimaryButton
          title="🖼️ Wybierz z galerii"
          onPress={pickFromGallery}
          style={{ marginTop: 12 }}
        />

        <Text style={styles.note}>
          To narzędzie nie jest diagnozą. Jeśli zmiana budzi niepokój — dermatolog.
        </Text>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  header: { marginTop: 12, marginBottom: 14 },
  title: { color: Colors.text, fontSize: 26, fontWeight: "900" },
  subtitle: { color: Colors.muted, marginTop: 6, lineHeight: 20 },

  previewBox: {
    alignSelf: "flex-start",
    marginBottom: 14,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 18,
    overflow: "hidden",
    backgroundColor: Colors.card,
  },
  previewImg: { width: 120, height: 120 },
  previewText: { color: Colors.muted, padding: 8, fontSize: 12 },

  card: {
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 24,
    padding: 18,
  },
  cardTitle: { color: Colors.text, fontSize: 16, fontWeight: "900" },
  note: { color: Colors.muted, marginTop: 14, fontSize: 12, lineHeight: 18 },
});
