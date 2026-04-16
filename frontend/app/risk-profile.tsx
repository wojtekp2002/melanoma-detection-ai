import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  TextInput,
  Alert,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { router } from "expo-router";
import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";
import { getUserRiskProfile, saveUserRiskProfile } from "@/db/profile.repository";
import { SkinPhototype } from "@/types/profile";

const phototypes: SkinPhototype[] = ["I", "II", "III", "IV", "V", "VI"];

function ToggleRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <Pressable
      onPress={() => onChange(!value)}
      style={[styles.toggleRow, value && styles.toggleRowActive]}
    >
      <View style={[styles.toggleBox, value && styles.toggleBoxActive]}>
        {value && <FontAwesome name="check" size={10} color={Colors.background} />}
      </View>
      <Text style={styles.toggleLabel}>{label}</Text>
    </Pressable>
  );
}

export default function RiskProfileScreen() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const [age, setAge] = useState("");
  const [familyHistorySkinCancer, setFamilyHistorySkinCancer] = useState(false);
  const [hadSevereSunburns, setHadSevereSunburns] = useState(false);
  const [manyMoles, setManyMoles] = useState(false);
  const [familyRelation, setFamilyRelation] = useState("");
  const [selectedPhototype, setSelectedPhototype] = useState<SkinPhototype | null>(null);

  useEffect(() => {
    loadProfile();
  }, []);

  async function loadProfile() {
    try {
      setLoading(true);

      const profile = await getUserRiskProfile();

      if (profile) {
        setAge(profile.age ? String(profile.age) : "");
        setSelectedPhototype(profile.skinPhototype ?? null);
        setFamilyHistorySkinCancer(profile.familyHistorySkinCancer);
        setFamilyRelation(profile.familyHistoryRelation ?? "");
        setHadSevereSunburns(profile.hadSevereSunburns);
        setManyMoles(profile.manyMoles);
      }
    } catch (error) {
      console.error("Błąd ładowania profilu:", error);
      Alert.alert("Błąd", "Nie udało się wczytać profilu.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSave() {
    try {
      setSaving(true);

      const parsedAge = age.trim() ? Number(age) : null;

      if (age.trim() && Number.isNaN(parsedAge)) {
        Alert.alert("Błąd", "Wiek musi być liczbą.");
        return;
      }

      await saveUserRiskProfile({
        age: parsedAge,
        skinPhototype: selectedPhototype,
        familyHistorySkinCancer,
        familyHistoryRelation: familyHistorySkinCancer ? familyRelation.trim() || null : null,
        hadSevereSunburns,
        manyMoles,
      });

      Alert.alert("Gotowe", "Profil zdrowotny został zapisany.");
      router.back();
    } catch (error) {
      console.error("Błąd zapisu profilu:", error);
      Alert.alert("Błąd", "Nie udało się zapisać profilu.");
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <LinearGradient
          colors={[Colors.background, "#0A1627", "#102038"]}
          style={[styles.container, styles.centered]}
        >
          <ActivityIndicator size="large" color={Colors.primary2} />
          <Text style={styles.loadingText}>Wczytywanie profilu...</Text>
        </LinearGradient>
      </SafeAreaView>
    );
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
          <View style={styles.headerRow}>
            <Pressable onPress={() => router.back()} style={styles.backButton}>
              <FontAwesome name="angle-left" size={20} color={Colors.text} />
            </Pressable>

            <View style={styles.headerTextWrap}>
              <Text style={styles.headerKicker}>Profil zdrowotny</Text>
              <Text style={styles.headerTitle}>Dodatkowe czynniki ryzyka</Text>
            </View>
          </View>

          <Text style={styles.description}>
            Te informacje pomagają budować pełniejszy profil użytkownika i mogą
            wspierać przyszłe rekomendacje w aplikacji. Na ten moment nie są bezpośrednio
            używane przez model obrazu.
          </Text>

          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Dane podstawowe</Text>

            <Text style={styles.label}>Wiek</Text>
            <TextInput
              value={age}
              onChangeText={setAge}
              placeholder="Np. 24"
              placeholderTextColor={Colors.textMuted}
              keyboardType="numeric"
              style={styles.input}
            />

            <Text style={styles.label}>Fototyp skóry</Text>
            <View style={styles.chipsRow}>
              {phototypes.map((type) => {
                const active = selectedPhototype === type;
                return (
                  <Pressable
                    key={type}
                    onPress={() => setSelectedPhototype(type)}
                    style={[styles.chip, active && styles.chipActive]}
                  >
                    <Text style={[styles.chipText, active && styles.chipTextActive]}>
                      {type}
                    </Text>
                  </Pressable>
                );
              })}
            </View>
          </View>

          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Historia i ekspozycja</Text>

            <ToggleRow
              label="W rodzinie występował rak skóry"
              value={familyHistorySkinCancer}
              onChange={setFamilyHistorySkinCancer}
            />

            {familyHistorySkinCancer && (
              <>
                <Text style={styles.label}>Kto chorował?</Text>
                <TextInput
                  value={familyRelation}
                  onChangeText={setFamilyRelation}
                  placeholder="Np. mama, dziadek od strony mamy"
                  placeholderTextColor={Colors.textMuted}
                  style={styles.input}
                />
              </>
            )}

            <ToggleRow
              label="Miałem/miałam w przeszłości ciężkie oparzenia słoneczne"
              value={hadSevereSunburns}
              onChange={setHadSevereSunburns}
            />

            <ToggleRow
              label="Mam dużo znamion / pieprzyków"
              value={manyMoles}
              onChange={setManyMoles}
            />
          </View>

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="info-circle" size={16} color={Colors.primary2} />
              <Text style={styles.infoTitle}>Ważne</Text>
            </View>

            <Text style={styles.infoText}>
              Ten moduł jest dodatkiem produktowym. Uczciwie oddzielamy ocenę obrazu
              od dodatkowych czynników ryzyka, żeby nie sugerować użytkownikowi,
              że obecny model już wykorzystuje wszystkie te dane.
            </Text>
          </View>

          <PrimaryButton
            title={saving ? "Zapisywanie..." : "Zapisz profil"}
            onPress={handleSave}
            disabled={saving}
            style={{ marginTop: 18 }}
          />
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
  centered: {
    alignItems: "center",
    justifyContent: "center",
  },
  content: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 36,
  },
  loadingText: {
    marginTop: 12,
    color: Colors.textSecondary,
    fontSize: 14,
  },

  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  backButton: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: Colors.border,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  headerTextWrap: {
    flex: 1,
  },
  headerKicker: {
    color: Colors.primary2,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.6,
    marginBottom: 4,
  },
  headerTitle: {
    color: Colors.text,
    fontSize: 24,
    fontWeight: "900",
  },
  description: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 18,
  },

  card: {
    borderRadius: 24,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 16,
  },
  sectionTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 16,
  },
  label: {
    color: Colors.text,
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 8,
    marginTop: 4,
  },
  input: {
    minHeight: 54,
    borderRadius: 18,
    paddingHorizontal: 16,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
    color: Colors.text,
    fontSize: 15,
    marginBottom: 14,
  },

  chipsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  chip: {
    minWidth: 52,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderWidth: 1,
    borderColor: Colors.border,
    alignItems: "center",
  },
  chipActive: {
    backgroundColor: "rgba(38,215,255,0.10)",
    borderColor: "rgba(38,215,255,0.30)",
  },
  chipText: {
    color: Colors.textSecondary,
    fontSize: 13,
    fontWeight: "800",
  },
  chipTextActive: {
    color: Colors.primary2,
  },

  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    padding: 14,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: Colors.border,
    backgroundColor: "rgba(255,255,255,0.03)",
    marginBottom: 12,
  },
  toggleRowActive: {
    borderColor: "rgba(38,215,255,0.30)",
    backgroundColor: "rgba(38,215,255,0.08)",
  },
  toggleBox: {
    width: 22,
    height: 22,
    borderRadius: 7,
    borderWidth: 1,
    borderColor: Colors.borderStrong,
    backgroundColor: "rgba(255,255,255,0.03)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  toggleBoxActive: {
    backgroundColor: Colors.primary2,
    borderColor: Colors.primary2,
  },
  toggleLabel: {
    flex: 1,
    color: Colors.text,
    fontSize: 14,
    lineHeight: 20,
    fontWeight: "700",
  },

  infoCard: {
    marginTop: 4,
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