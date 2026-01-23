import React from "react";
import { Pressable, Text, StyleSheet, ViewStyle } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Colors } from "../constants/Colors";

export default function PrimaryButton({
  title,
  onPress,
  disabled,
  style,
}: {
  title: string;
  onPress: () => void;
  disabled?: boolean;
  style?: ViewStyle;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={[styles.wrap, style, disabled && { opacity: 0.45 }]}
    >
      <LinearGradient
        colors={[Colors.primary, Colors.primary2]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.btn}
      >
        <Text style={styles.txt}>{title}</Text>
      </LinearGradient>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: { width: "100%" },
  btn: {
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: "center",
  },
  txt: {
    color: Colors.bg,
    fontSize: 16,
    fontWeight: "800",
    letterSpacing: 0.2,
  },
});
