/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#f0f4ff',
          100: '#dce7ff',
          500: '#4f72ff',
          600: '#3b5bdb',
          700: '#2f4abf',
          900: '#1a2a7a',
        },
        surface: {
          DEFAULT: '#0f1117',
          card:    '#161b27',
          border:  '#252d3d',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
