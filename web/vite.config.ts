import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: '../weclone/web/dist', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:5175' } },
  preview: { proxy: { '/api': 'http://127.0.0.1:5175' } },
});
