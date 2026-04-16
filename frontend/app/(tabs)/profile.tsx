import React from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { router } from "expo-router";
import { Colors } from "@/constants/Colors";

const menuItems = [
  {
    id: "1",
    icon: "heartbeat",
    title: "Profil zdrowotny",
    subtitle: "Wiek, fototyp skóry i dodatkowe czynniki ryzyka",
    route: "/risk-profile",
  },
  {
    id: "2",
    icon: "map-marker",
    title: "Moje zmiany",
    subtitle: "Lista obserwowanych zmian skórnych i ich lokalizacja",
    route: "/(tabs)/lesions",
  },
  {
    id: "3",
    icon: "shield",
    title: "Prywatność i bezpieczeństwo",
    subtitle: "Informacje o przechowywaniu danych i zdjęć",
    route: null,
  },
  {
    id: "4",
    icon: "info-circle",
    title: "O aplikacji",
    subtitle: "Wersja projektu, cel i ograniczenia modelu",
    route: null,
  },
];

export default function ProfileScreen() {
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
            <View style={styles.avatar}>
              <FontAwesome name="user" size={28} color={Colors.background} />
            </View>

            <Text style={styles.name}>Użytkownik aplikacji</Text>
            <Text style={styles.email}>demo@melanoma-ai.app</Text>
          </View>

          <View style={styles.heroCard}>
            <View style={styles.heroBadge}>
              <FontAwesome name="shield" size={14} color={Colors.primary2} />
              <Text style={styles.heroBadgeText}>Profil i ustawienia</Text>
            </View>

            <Text style={styles.heroTitle}>Zarządzaj swoim kontem</Text>
            <Text style={styles.heroDescription}>
              W tej sekcji użytkownik będzie mógł przeglądać dane profilu,
              zarządzać prywatnością, ustawieniami aplikacji i historią obserwacji.
            </Text>
          </View>

          <View style={styles.menuCard}>
            {menuItems.map((item, index) => (
                <Pressable
                key={item.id}
                onPress={() => {
                    if (item.route) {
                    router.push({ pathname: item.route } as any);
                    }
                }}
                style={[
                    styles.menuItem,
                    index !== menuItems.length - 1 && styles.menuItemBorder,
                ]}
                >
                <View style={styles.menuLeft}>
                  <View style={styles.menuIcon}>
                    <FontAwesome
                      name={item.icon as any}
                      size={17}
                      color={Colors.primary2}
                    />
                  </View>

                  <View style={styles.menuTextWrap}>
                    <Text style={styles.menuTitle}>{item.title}</Text>
                    <Text style={styles.menuSubtitle}>{item.subtitle}</Text>
                  </View>
                </View>

                <FontAwesome
                  name="angle-right"
                  size={18}
                  color={Colors.textSecondary}
                />
              </Pressable>
            ))}
          </View>

          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>12</Text>
              <Text style={styles.statLabel}>Analiz zapisanych</Text>
            </View>

            <View style={styles.statCard}>
              <Text style={styles.statValue}>3</Text>
              <Text style={styles.statLabel}>Zmiany śledzone</Text>
            </View>
          </View>

          <Pressable
            onPress={() => router.replace("/welcome")}
            style={styles.logoutButton}
          >
            <FontAwesome name="sign-out" size={16} color={Colors.danger} />
            <Text style={styles.logoutText}>Wyloguj</Text>
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
    paddingTop: 16,
    paddingBottom: 120,
  },

  header: {
    alignItems: "center",
    marginBottom: 22,
  },
  avatar: {
    width: 82,
    height: 82,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: Colors.primary2,
    marginBottom: 14,
  },
  name: {
    color: Colors.text,
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 4,
  },
  email: {
    color: Colors.textSecondary,
    fontSize: 14,
  },

  heroCard: {
    borderRadius: 26,
    padding: 18,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 18,
  },
  heroBadge: {
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
    marginBottom: 14,
  },
  heroBadgeText: {
    color: Colors.text,
    fontSize: 12,
    fontWeight: "700",
  },
  heroTitle: {
    color: Colors.text,
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 8,
  },
  heroDescription: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 22,
  },

  menuCard: {
    borderRadius: 24,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
    marginBottom: 18,
  },
  menuItem: {
    minHeight: 82,
    paddingHorizontal: 16,
    paddingVertical: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  menuItemBorder: {
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.06)",
  },
  menuLeft: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
    marginRight: 10,
  },
  menuIcon: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: "rgba(38,215,255,0.10)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  menuTextWrap: {
    flex: 1,
  },
  menuTitle: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "800",
    marginBottom: 4,
  },
  menuSubtitle: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 17,
  },

  statsRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },
  statCard: {
    flex: 1,
    borderRadius: 22,
    padding: 16,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  statValue: {
    color: Colors.text,
    fontSize: 22,
    fontWeight: "900",
    marginBottom: 4,
  },
  statLabel: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },

  logoutButton: {
    minHeight: 56,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "rgba(255,95,122,0.18)",
    backgroundColor: "rgba(255,95,122,0.08)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  logoutText: {
    color: Colors.danger,
    fontSize: 15,
    fontWeight: "800",
  },
});