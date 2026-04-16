import * as SQLite from "expo-sqlite";

const db = SQLite.openDatabaseSync("melanoma-ai-v3.db");

export async function initDatabase() {
  await db.execAsync(`
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS user_profile (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      age INTEGER,
      skin_phototype TEXT,

      family_history_skin_cancer INTEGER NOT NULL DEFAULT 0,
      family_history_relation TEXT,

      family_history_other_cancer INTEGER NOT NULL DEFAULT 0,
      family_history_other_cancer_relation TEXT,

      had_severe_sunburns INTEGER NOT NULL DEFAULT 0,
      frequent_sun_exposure INTEGER NOT NULL DEFAULT 0,
      uses_tanning_beds INTEGER NOT NULL DEFAULT 0,

      many_moles INTEGER NOT NULL DEFAULT 0,
      atypical_moles INTEGER NOT NULL DEFAULT 0,
      very_fair_skin INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS lesions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      body_side TEXT NOT NULL,
      body_label TEXT,
      x REAL,
      y REAL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS observations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      lesion_id INTEGER,
      image_uri TEXT NOT NULL,
      probability REAL NOT NULL,
      label TEXT NOT NULL,
      created_at TEXT NOT NULL,
      note TEXT,
      FOREIGN KEY (lesion_id) REFERENCES lesions(id) ON DELETE CASCADE
    );
  `);
}

export function getDb() {
  return db;
}