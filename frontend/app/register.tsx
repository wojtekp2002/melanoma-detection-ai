import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Alert,
  ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import { FontAwesome } from "@expo/vector-icons";

import { Colors } from "@/constants/Colors";
import PrimaryButton from "@/components/PrimaryButton";
import AppInput from "@/components/AppInput";

export default function RegisterScreen() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  function handleRegister() {
    if (!name.trim() || !email.trim() || !password.trim() || !confirmPassword.trim()) {
      Alert.alert("Brak danych", "Uzupełnij wszystkie pola.");
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert("Błąd", "Hasła nie są takie same.");
      return;
    }

    if (password.length < 6) {
      Alert.alert("Za krótkie hasło", "Hasło powinno mieć co najmniej 6 znaków.");
      return;
    }

    router.replace("/disclaimer");
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
          <View style={styles.header}>
            <View style={styles.badge}>
              <FontAwesome name="user-plus" size={14} color={Colors.primary2} />
              <Text style={styles.badgeText}>Rejestracja</Text>
            </View>

            <Text style={styles.title}>Załóż konto</Text>
            <Text style={styles.description}>
              Utwórz konto, aby zapisywać analizy, budować historię obserwacji i
              śledzić zmiany skórne w czasie.
            </Text>
          </View>

          <View style={styles.card}>
            <AppInput
              label="Imię"
              value={name}
              onChangeText={setName}
              placeholder="Np. Jan"
            />

            <AppInput
              label="Adres e-mail"
              value={email}
              onChangeText={setEmail}
              placeholder="twoj@email.com"
              keyboardType="email-address"
              autoCapitalize="none"
            />

            <AppInput
              label="Hasło"
              value={password}
              onChangeText={setPassword}
              placeholder="Minimum 6 znaków"
              secureTextEntry
            />

            <AppInput
              label="Powtórz hasło"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              placeholder="Wpisz hasło ponownie"
              secureTextEntry
            />

            <PrimaryButton
              title="Utwórz konto"
              onPress={handleRegister}
              style={{ marginTop: 6 }}
            />
          </View>

          <View style={styles.bottomRow}>
            <Text style={styles.bottomText}>Masz już konto?</Text>
            <Pressable onPress={() => router.push("/login")}>
              <Text style={styles.bottomLink}> Zaloguj się</Text>
            </Pressable>
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
    paddingTop: 20,
    paddingBottom: 36,
    flexGrow: 1,
    justifyContent: "center",
  },
  header: {
    marginBottom: 24,
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
    marginBottom: 16,
  },
  badgeText: {
    color: Colors.text,
    fontSize: 12,
    fontWeight: "700",
  },
  title: {
    color: Colors.text,
    fontSize: 32,
    lineHeight: 39,
    fontWeight: "900",
    marginBottom: 10,
  },
  description: {
    color: Colors.textSecondary,
    fontSize: 15,
    lineHeight: 24,
  },
  card: {
    borderRadius: 26,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  bottomRow: {
    marginTop: 22,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
  },
  bottomText: {
    color: Colors.textMuted,
    fontSize: 14,
  },
  bottomLink: {
    color: Colors.primary2,
    fontSize: 14,
    fontWeight: "800",
  },
});