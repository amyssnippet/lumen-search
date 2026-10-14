import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ivory: {
          DEFAULT: "#FAF9F5",
          50: "#FDFDFB",
          100: "#FAF9F5",
          200: "#F4F1EA",
          300: "#EAE5D9",
        },
        gold: {
          subtle: "#FBF8F2",
          light: "#EADDC6",
          DEFAULT: "#BFA16F",
          dark: "#9C7F4E",
          deep: "#7A6237",
        },
        charcoal: {
          50: "#F5F5F4",
          100: "#E7E5E4",
          200: "#D6D3D1",
          400: "#A8A29E",
          600: "#57534E",
          700: "#44403C",
          800: "#292524",
          DEFAULT: "#1C1917",
          900: "#1C1917",
        },
      },
      fontFamily: {
        sans: ["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        serif: ["Georgia", "Cambria", "Times New Roman", "serif"],
      },
      boxShadow: {
        subtle: "0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02)",
        card: "0 2px 8px -2px rgba(28, 25, 23, 0.06), 0 1px 4px -1px rgba(28, 25, 23, 0.04)",
        elevated: "0 10px 25px -5px rgba(28, 25, 23, 0.08), 0 8px 10px -6px rgba(28, 25, 23, 0.04)",
      },
    },
  },
  plugins: [],
};

export default config;
