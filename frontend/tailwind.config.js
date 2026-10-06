/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef4ff", 100: "#d9e6ff", 200: "#b3ccff", 300: "#83abff",
          400: "#5285ff", 500: "#2f63f7", 600: "#1f47db", 700: "#1a38ac",
          800: "#1a3189", 900: "#1b2d6d",
        },
        clinical: {
          teal: "#0f9b8e", amber: "#e8a33d", coral: "#e8615a", slate: "#4b5563",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Sora", "Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
