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
          position: "absolute",
          left: 14,
          right: 14,
          bottom: 14,
          height: 74,
          paddingTop: 10,
          paddingBottom: 10,
          backgroundColor: "rgba(12, 22, 38, 0.94)",
          borderTopWidth: 1,
          borderTopColor: "rgba(255,255,255,0.08)",
          borderRadius: 24,
        },
        tabBarActiveTintColor: Colors.primary2,
        tabBarInactiveTintColor: Colors.textMuted,
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: "700",
          marginTop: 2,
        },
        tabBarHideOnKeyboard: true,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Analiza",
          tabBarIcon: ({ color }) => (
            <FontAwesome name="camera" size={18} color={color} />
          ),
        }}
      />

      <Tabs.Screen
        name="history"
        options={{
          title: "Historia",
          tabBarIcon: ({ color }) => (
            <FontAwesome name="clock-o" size={18} color={color} />
          ),
        }}
      />

      <Tabs.Screen
        name="profile"
        options={{
          title: "Profil",
          tabBarIcon: ({ color }) => (
            <FontAwesome name="user-o" size={18} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}