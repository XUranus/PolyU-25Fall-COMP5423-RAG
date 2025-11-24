// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#111111',
        'sidebar-bg': '#1e1e1e',
        'chat-bg': '#1a1a1a',
        'message-user': '#0d6efd',
        'message-bot': '#2d2d2d',
      },
    },
  },
  plugins: [],
}