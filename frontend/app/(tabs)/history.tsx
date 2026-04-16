import React from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { FontAwesome } from "@expo/vector-icons";
import { Colors } from "@/constants/Colors";

const mockHistory = [
  {
    id: "1",
    date: "12 kwietnia 2026",
    risk: "Podwyższone ryzyko",
    probability: "72%",
    image:
      "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?q=80&w=600&auto=format&fit=crop",
  },
  {
    id: "2",
    date: "6 kwietnia 2026",
    risk: "Niskie ryzyko",
    probability: "18%",
    image:
      "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=600&auto=format&fit=crop",
  },
  {
    id: "3",
    date: "28 marca 2026",
    risk: "Umiarkowane ryzyko",
    probability: "43%",
    image:
      "https://images.unsplash.com/photo-1584515933487-779824d29309?q=80&w=600&auto=format&fit=crop",
  },
];

export default function HistoryScreen() {
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
              <FontAwesome name="clock-o" size={14} color={Colors.primary2} />
              <Text style={styles.badgeText}>Historia analiz</Text>
            </View>

            <Text style={styles.title}>Twoje wcześniejsze wyniki</Text>
            <Text style={styles.description}>
              Przeglądaj zapisane analizy, porównuj wcześniejsze obserwacje i
              buduj historię monitorowania zmian skórnych.
            </Text>
          </View>

          <View style={styles.summaryCard}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>12</Text>
              <Text style={styles.summaryLabel}>Wszystkich analiz</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>4</Text>
              <Text style={styles.summaryLabel}>W tym miesiącu</Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>3</Text>
              <Text style={styles.summaryLabel}>Obserwowane zmiany</Text>
            </View>
          </View>

          <View style={styles.listHeader}>
            <Text style={styles.listTitle}>Ostatnie obserwacje</Text>
            <Pressable>
              <Text style={styles.listAction}>Zobacz wszystkie</Text>
            </Pressable>
          </View>

          {mockHistory.map((item) => (
            <Pressable key={item.id} style={styles.card}>
              <Image source={{ uri: item.image }} style={styles.image} />

              <View style={styles.cardContent}>
                <Text style={styles.cardDate}>{item.date}</Text>
                <Text style={styles.cardTitle}>{item.risk}</Text>
                <Text style={styles.cardSubtitle}>
                  Szacowane prawdopodobieństwo: {item.probability}
                </Text>

                <View style={styles.cardFooter}>
                  <View style={styles.badgeSmall}>
                    <Text style={styles.badgeSmallText}>Analiza zdjęcia</Text>
                  </View>

                  <FontAwesome
                    name="angle-right"
                    size={18}
                    color={Colors.textSecondary}
                  />
                </View>
              </View>
            </Pressable>
          ))}

          <View style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <FontAwesome name="info-circle" size={16} color={Colors.primary2} />
              <Text style={styles.infoTitle}>Co później tu dodamy</Text>
            </View>

            <Text style={styles.infoText}>
              W następnym kroku podepniemy prawdziwy zapis historii lokalnie,
              a potem porównywanie zmian w czasie i grupowanie obserwacji według
              konkretnej zmiany skórnej.
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
    paddingTop: 16,
    paddingBottom: 120,
  },

  header: {
    marginBottom: 20,
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
    fontSize: 30,
    lineHeight: 37,
    fontWeight: "900",
    marginBottom: 10,
  },
  description: {
    color: Colors.textSecondary,
    fontSize: 15,
    lineHeight: 24,
  },

  summaryCard: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 22,
  },
  summaryItem: {
    flex: 1,
    borderRadius: 20,
    paddingVertical: 16,
    paddingHorizontal: 12,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  summaryValue: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: "900",
    marginBottom: 4,
  },
  summaryLabel: {
    color: Colors.textMuted,
    fontSize: 12,
    lineHeight: 16,
  },

  listHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 14,
  },
  listTitle: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
  },
  listAction: {
    color: Colors.primary2,
    fontSize: 13,
    fontWeight: "700",
  },

  card: {
    flexDirection: "row",
    backgroundColor: Colors.surface,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
    marginBottom: 14,
  },
  image: {
    width: 96,
    height: 120,
  },
  cardContent: {
    flex: 1,
    padding: 14,
    justifyContent: "center",
  },
  cardDate: {
    color: Colors.primary2,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.6,
    marginBottom: 6,
  },
  cardTitle: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: "800",
    marginBottom: 6,
  },
  cardSubtitle: {
    color: Colors.textSecondary,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: 12,
  },
  cardFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  badgeSmall: {
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderWidth: 1,
    borderColor: Colors.border,
  },
  badgeSmallText: {
    color: Colors.textSecondary,
    fontSize: 11,
    fontWeight: "700",
  },

  infoCard: {
    marginTop: 8,
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