import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#181A1F",
        paper: "#F7F7F4",
        line: "#D9D7CD",
        muted: "#6D706A",
        accent: "#D44931"
      },
      boxShadow: {
        soft: "0 18px 50px rgba(24, 26, 31, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
