/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        skinora: {
          bg: '#FFFBF1',          // Primary background: Warm ivory / light beige
          surface: '#F7F0E5',     // Secondary background / sections / panels
          card: '#FFFFFF',        // Card / surface
          text: '#644A47',        // Primary text: Deep warm brown
          textDark: '#3F3430',    // Dark text: Rich espresso brown
          muted: '#9B8C7B',       // Secondary / placeholder text: Muted warm gray-brown
          accent: '#A37D6C',      // Primary accent: Terracotta / warm clay
          accentHover: '#8A6454', // Hover state for primary buttons
          border: '#D3C0A8',      // Subtle beige border
          peach: '#E7B697',       // Soft peach accent
          gold: '#F3C88A',        // Warm gold accent
          green: '#B3D1B4',       // Soft sage green
        },
      },
      boxShadow: {
        'soft': '0 2px 10px -1px rgba(100, 74, 71, 0.05), 0 1px 3px rgba(100, 74, 71, 0.03)',
        'soft-md': '0 4px 16px -2px rgba(100, 74, 71, 0.07), 0 2px 6px rgba(100, 74, 71, 0.04)',
      },
    },
  },
  plugins: [],
}
