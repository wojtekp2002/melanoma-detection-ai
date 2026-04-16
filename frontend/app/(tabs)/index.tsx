import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Alert,
  Image,
  ScrollView,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";

import * as ImagePicker from "expo-image-picker";
import { Camera } from "expo-camera";
import { FontAwesome } from "@expo/vector-icons";

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
      Alert.alert("Brak uprawnień", "Nadaj aplikacji dostęp do galerii w ustawieniach.");
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
      Alert.alert("Brak uprawnień", "Nadaj aplikacji dostęp do aparatu w ustawieniach.");
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

  function reopenLastImage() {
    if (!lastUri) return;
    router.push({ pathname: "/preview", params: { uri: lastUri } } as any);
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
          <LinearGradient
            colors={["rgba(124,92,255,0.20)", "rgba(38,215,255,0.10)"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.heroCard}
          >
            <View style={styles.heroBadge}>
              <FontAwesome name="shield" size={13} color={Colors.primary2} />
              <Text style={styles.heroBadgeText}>Monitoring zmian skórnych</Text>
            </View>

            <Text style={styles.heroTitle}>Przeanalizuj nowe zdjęcie</Text>

            <Text style={styles.heroSubtitle}>
              Zrób wyraźne zdjęcie zmiany skórnej lub wybierz je z galerii.
              Aplikacja oceni poziom ryzyka i pomoże Ci monitorować obserwacje w czasie.
            </Text>

            <View style={styles.heroStatsRow}>
              <View style={styles.heroStatBox}>
                <Text style={styles.heroStatValue}>AI</Text>
                <Text style={styles.heroStatLabel}>Analiza obrazu</Text>
              </View>

              <View style={styles.heroStatBox}>
                <Text style={styles.heroStatValue}>24/7</Text>
                <Text style={styles.heroStatLabel}>Szybki dostęp</Text>
              </View>

              <View style={styles.heroStatBox}>
                <Text style={styles.heroStatValue}>Safe</Text>
                <Text style={styles.heroStatLabel}>To nie diagnoza</Text>
              </View>
            </View>
          </LinearGradient>

          <View style={styles.actionsCard}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Dodaj zdjęcie</Text>
              <Text style={styles.sectionSubtitle}>
                Wybierz najlepszy sposób rozpoczęcia analizy.
              </Text>
            </View>

            <PrimaryButton title="Zrób zdjęcie" onPress={takePhoto} />

            <Pressable onPress={pickFromGallery} style={({ pressed }) => [
              styles.secondaryAction,
              pressed && styles.secondaryActionPressed,
            ]}>
              <View style={styles.secondaryActionIcon}>
                <FontAwesome name="image" size={18} color={Colors.primary2} />
              </View>

              <View style={styles.secondaryActionTextWrap}>
                <Text style={styles.secondaryActionTitle}>Wybierz z galerii</Text>
                <Text style={styles.secondaryActionSubtitle}>
                  Użyj istniejącego zdjęcia zapisane­go w telefonie
                </Text>
              </View>

              <FontAwesome
                name="angle-right"
                size={20}
                color={Colors.textSecondary}
              />
            </Pressable>

            <View style={styles.tipsCard}>
              <View style={styles.tipsHeader}>
                <FontAwesome name="lightbulb-o" size={16} color={Colors.warning} />
                <Text style={styles.tipsTitle}>Jak uzyskać lepszy wynik</Text>
              </View>

              <View style={styles.tipItem}>
                <View style={styles.tipDot} />
                <Text style={styles.tipText}>Użyj dobrego, naturalnego lub równego światła</Text>
              </View>

              <View style={styles.tipItem}>
                <View style={styles.tipDot} />
                <Text style={styles.tipText}>Ustaw zmianę blisko i centralnie w kadrze</Text>
              </View>

              <View style={styles.tipItem}>
                <View style={styles.tipDot} />
                <Text style={styles.tipText}>Unikaj rozmazania, cieni i kilku zmian naraz</Text>
              </View>
            </View>
          </View>

          {lastUri && (
            <View style={styles.lastCard}>
              <Image source={{ uri: lastUri }} style={styles.lastImage} />

              <View style={styles.lastContent}>
                <Text style={styles.lastLabel}>OSTATNIE ZDJĘCIE</Text>
                <Text style={styles.lastTitle}>Kontynuuj poprzednią analizę</Text>
                <Text style={styles.lastDescription}>
                  Otwórz ostatnio wybrane zdjęcie i przejdź od razu do ekranu podglądu.
                </Text>

                <Pressable
                  onPress={reopenLastImage}
                  style={({ pressed }) => [
                    styles.lastButton,
                    pressed && styles.lastButtonPressed,
                  ]}
                >
                  <Text style={styles.lastButtonText}>Otwórz podgląd</Text>
                </Pressable>
              </View>
            </View>
          )}

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="info-circle" size={16} color={Colors.primary2} />
              <Text style={styles.infoTitle}>Ważna informacja</Text>
            </View>

            <Text style={styles.infoText}>
              Aplikacja służy do wstępnej oceny ryzyka i monitorowania zmian w czasie.
              Nie zastępuje konsultacji z dermatologiem ani diagnozy medycznej.
            </Text>
          </View>
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
    paddingTop: 10,
    paddingBottom: 120,
  },

  heroCard: {
    borderRadius: 28,
    padding: 22,
    borderWidth: 1,
    borderColor: Colors.borderStrong,
    backgroundColor: Colors.surface,
    marginBottom: 18,
    overflow: "hidden",
  },
  heroBadge: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
    marginBottom: 16,
  },
  heroBadgeText: {
    color: Colors.text,
    fontSize: 12,
    fontWeight: "700",
  },
  heroTitle: {
    color: Colors.text,
    fontSize: 31,
    lineHeight: 38,
    fontWeight: "900",
    marginBottom: 10,
  },
  heroSubtitle: {
    color: Colors.textSecondary,
    fontSize: 15,
    lineHeight: 24,
    marginBottom: 20,
  },
  heroStatsRow: {
    flexDirection: "row",
    gap: 10,
  },
  heroStatBox: {
    flex: 1,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.08)",
    paddingVertical: 14,
    paddingHorizontal: 12,
  },
  heroStatValue: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "900",
    marginBottom: 4,
  },
  heroStatLabel: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },

  actionsCard: {
    backgroundColor: Colors.surface,
    borderRadius: 26,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 18,
    marginBottom: 18,
  },
  sectionHeader: {
    marginBottom: 16,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: "900",
    marginBottom: 6,
  },
  sectionSubtitle: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 21,
  },

  secondaryAction: {
    marginTop: 12,
    minHeight: 64,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
  },
  secondaryActionPressed: {
    opacity: 0.92,
  },
  secondaryActionIcon: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: "rgba(38,215,255,0.10)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  secondaryActionTextWrap: {
    flex: 1,
  },
  secondaryActionTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
    marginBottom: 2,
  },
  secondaryActionSubtitle: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 17,
  },

  tipsCard: {
    marginTop: 18,
    borderRadius: 20,
    padding: 16,
    backgroundColor: "rgba(255,255,255,0.035)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  tipsHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  tipsTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
  },
  tipItem: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 10,
  },
  tipDot: {
    width: 7,
    height: 7,
    borderRadius: 999,
    backgroundColor: Colors.primary2,
    marginTop: 7,
    marginRight: 10,
  },
  tipText: {
    flex: 1,
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },

  lastCard: {
    flexDirection: "row",
    backgroundColor: Colors.surface,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
    marginBottom: 18,
  },
  lastImage: {
    width: 110,
    height: 140,
  },
  lastContent: {
    flex: 1,
    padding: 16,
    justifyContent: "center",
  },
  lastLabel: {
    color: Colors.primary2,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.8,
    marginBottom: 8,
  },
  lastTitle: {
    color: Colors.text,
    fontSize: 17,
    fontWeight: "900",
    marginBottom: 8,
  },
  lastDescription: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 14,
  },
  lastButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.07)",
    borderWidth: 1,
    borderColor: Colors.borderStrong,
  },
  lastButtonPressed: {
    opacity: 0.92,
  },
  lastButtonText: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: "800",
  },

  infoCard: {
    borderRadius: 22,
    padding: 16,
    backgroundColor: "rgba(38,215,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(38,215,255,0.16)",
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
    fontSize: 13,
    lineHeight: 20,
  },
});