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

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  function handleLogin() {
    if (!email.trim() || !password.trim()) {
      Alert.alert("Brak danych", "Uzupełnij email i hasło.");
      return;
    }

    router.replace("/(tabs)");
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
              <FontAwesome name="user-circle-o" size={14} color={Colors.primary2} />
              <Text style={styles.badgeText}>Logowanie</Text>
            </View>

            <Text style={styles.title}>Witaj ponownie</Text>
            <Text style={styles.description}>
              Zaloguj się, aby korzystać z historii analiz, profilu użytkownika i
              monitorowania zmian skórnych.
            </Text>
          </View>

          <View style={styles.card}>
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
              placeholder="Wpisz hasło"
              secureTextEntry
            />

            <PrimaryButton
              title="Zaloguj się"
              onPress={handleLogin}
              style={{ marginTop: 6 }}
            />

            <Pressable style={styles.ghostButton}>
              <Text style={styles.ghostButtonText}>Nie pamiętasz hasła?</Text>
            </Pressable>
          </View>

          <View style={styles.bottomRow}>
            <Text style={styles.bottomText}>Nie masz jeszcze konta?</Text>
            <Pressable onPress={() => router.push("/register")}>
              <Text style={styles.bottomLink}> Zarejestruj się</Text>
            </Pressable>
          </View>

          <Pressable onPress={() => router.replace("/disclaimer")}>
            <Text style={styles.skipText}>Kontynuuj bez logowania</Text>
          </Pressable>
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
  ghostButton: {
    marginTop: 12,
    alignSelf: "center",
    paddingVertical: 8,
  },
  ghostButtonText: {
    color: Colors.textSecondary,
    fontSize: 13,
    fontWeight: "700",
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
  skipText: {
    marginTop: 18,
    textAlign: "center",
    color: Colors.textMuted,
    fontSize: 13,
  },
});