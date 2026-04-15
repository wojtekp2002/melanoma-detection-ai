import React from "react";
import { Tabs } from "expo-router";
import { FontAwesome } from "@expo/vector-icons";
import { Colors } from "@/constants/Colors";

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: "#0B1220",
          borderTopColor: "rgba(255,255,255,0.08)",
          height: 74,
          paddingBottom: 10,
          paddingTop: 10,
        },
        tabBarActiveTintColor: Colors.primary2,
        tabBarInactiveTintColor: "rgba(234,240,255,0.45)",
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: "700",
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Analiza",
          tabBarIcon: ({ color }) => (
            <FontAwesome name="camera" size={20} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}