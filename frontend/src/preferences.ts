export type Preferences = {
  appearance: "light" | "dark" | "system";
  research: boolean;
  voice: string;
  rate: number;
  dictationLanguage: string;
  performanceOpen: boolean;
};
export const defaultPreferences: Preferences = {
  appearance: "system", research: false, voice: "", rate: 1,
  dictationLanguage: "", performanceOpen: false,
};
export function loadPreferences(): Preferences {
  try {
    const value = JSON.parse(localStorage.getItem("atlas.preferences") ?? "{}");
    const oldTheme = localStorage.getItem("atlas.theme");
    return {
      appearance: ["light", "dark", "system"].includes(value.appearance) ? value.appearance :
        oldTheme === "light" || oldTheme === "dark" ? oldTheme : "system",
      research: value.research === true, voice: typeof value.voice === "string" ? value.voice : "",
      rate: typeof value.rate === "number" && value.rate >= .5 && value.rate <= 2 ? value.rate : 1,
      dictationLanguage: typeof value.dictationLanguage === "string" ? value.dictationLanguage : "",
      performanceOpen: value.performanceOpen === true,
    };
  } catch { return { ...defaultPreferences }; }
}
export function savePreferences(value: Preferences) {
  localStorage.setItem("atlas.preferences", JSON.stringify(value));
}
