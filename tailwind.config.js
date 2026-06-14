/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Core surface palette — a deep "ink" background with slate panels.
        ink: {
          950: "#070912",
          900: "#0b0e1a",
          850: "#0f1322",
          800: "#141a2e",
          700: "#1c2438",
          600: "#27314a",
        },
        // Brand accent — an electric indigo/cyan pairing used for the LLM theme.
        brand: {
          50: "#eef3ff",
          100: "#dbe5ff",
          200: "#bcceff",
          300: "#8eabff",
          400: "#5b7dff",
          500: "#3a55f5",
          600: "#2b3fd9",
          700: "#2433af",
          800: "#222e8a",
          900: "#212c6e",
        },
        accent: {
          cyan: "#22d3ee",
          violet: "#a855f7",
          amber: "#f59e0b",
          emerald: "#34d399",
          rose: "#fb7185",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      typography: {},
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
        "flow-dash": {
          to: { strokeDashoffset: "-16" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.5s ease-out both",
        "pulse-soft": "pulse-soft 2.4s ease-in-out infinite",
        "flow-dash": "flow-dash 0.8s linear infinite",
      },
    },
  },
  plugins: [],
};
