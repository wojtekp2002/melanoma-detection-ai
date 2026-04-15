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
    router.push({ pathname: "/preview", params: { uri } } as any);
  }

  return (
    <LinearGradient colors={[Colors.bg, "#141C33", "#1A2240"]} style={styles.container}>
      <View style={styles.hero}>
        <Text style={styles.kicker}>Melanoma AI</Text>
        <Text style={styles.title}>Nowa analiza</Text>
        <Text style={styles.subtitle}>
          Zrób ostre zdjęcie zmiany skórnej lub wybierz je z galerii. Najlepsze
          wyniki uzyskasz przy dobrym świetle i bliskim kadrze.
        </Text>
      </View>

      {lastUri && (
        <View style={styles.lastCard}>
          <Image source={{ uri: lastUri }} style={styles.lastImage} />
          <View style={styles.lastContent}>
            <Text style={styles.lastTitle}>Ostatnio wybrane zdjęcie</Text>
            <Text style={styles.lastDesc}>
              Możesz od razu wrócić do analizy albo wybrać nowe zdjęcie.
            </Text>
          </View>
        </View>
      )}

      <View style={styles.actionCard}>
        <Text style={styles.sectionTitle}>Wybierz źródło zdjęcia</Text>

        <PrimaryButton
          title="📷 Zrób zdjęcie"
          onPress={takePhoto}
          style={{ marginTop: 14 }}
        />

        <PrimaryButton
          title="🖼️ Wybierz z galerii"
          onPress={pickFromGallery}
          style={{ marginTop: 12 }}
        />

        <View style={styles.tipBox}>
          <Text style={styles.tipTitle}>Wskazówki</Text>
          <Text style={styles.tipText}>• Zadbaj o dobre, równe światło</Text>
          <Text style={styles.tipText}>• Zmiana powinna być blisko i wyraźna</Text>
          <Text style={styles.tipText}>• Unikaj rozmazania i mocnych cieni</Text>
        </View>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  hero: {
    marginTop: 14,
    marginBottom: 18,
  },
  kicker: {
    color: Colors.primary2,
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 0.8,
    marginBottom: 6,
  },
  title: {
    color: Colors.text,
    fontSize: 30,
    fontWeight: "900",
  },
  subtitle: {
    color: Colors.muted,
    marginTop: 10,
    lineHeight: 22,
    fontSize: 15,
  },
  lastCard: {
    flexDirection: "row",
    backgroundColor: Colors.card,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
    marginBottom: 16,
  },
  lastImage: {
    width: 96,
    height: 96,
  },
  lastContent: {
    flex: 1,
    padding: 14,
    justifyContent: "center",
  },
  lastTitle: {
    color: Colors.text,
    fontSize: 14,
    fontWeight: "800",
  },
  lastDesc: {
    color: Colors.muted,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 6,
  },
  actionCard: {
    backgroundColor: Colors.card,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 18,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 17,
    fontWeight: "900",
  },
  tipBox: {
    marginTop: 18,
    padding: 14,
    borderRadius: 18,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  tipTitle: {
    color: Colors.text,
    fontWeight: "800",
    marginBottom: 8,
  },
  tipText: {
    color: Colors.muted,
    lineHeight: 20,
    fontSize: 13,
  },
});