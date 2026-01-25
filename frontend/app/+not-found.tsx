import React from "react";
import { StyleSheet, View, Text } from "react-native";
import { Link } from "expo-router";
import { Colors } from "@/constants/Colors";

export default function NotFoundScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>404</Text>
      <Text style={styles.subtitle}>Nie znaleziono tej strony.</Text>

      <Link href="/welcome" style={styles.link}>
        Wróć do startu
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.bg,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  title: {
    color: Colors.text,
    fontSize: 44,
    fontWeight: "900",
  },
  subtitle: {
    color: Colors.muted,
    marginTop: 10,
    marginBottom: 16,
    textAlign: "center",
  },
  link: {
    color: Colors.primary2,
    fontWeight: "800",
    fontSize: 16,
  },
});
