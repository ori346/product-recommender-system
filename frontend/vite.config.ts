import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { tanstackRouter } from '@tanstack/router-plugin/vite';
import { resolve } from 'path';

// Global target URL - change this to switch between local and OpenShift
const TARGET = 'http://localhost:8000';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    // Please make sure that '@tanstack/router-plugin' is passed before '@vitejs/plugin-react'
    tanstackRouter({ target: 'react', autoCodeSplitting: true }),
    react(),
    // ...,
  ],
  resolve: {
    alias: {
      '@': resolve(process.cwd(), './src'), // This line maps '@/' to your 'src' directory
    },
  },
  server: {
    proxy: {
      '/auth': {
        target: TARGET,
        changeOrigin: true,
      },
      '/products': {
        target: TARGET,
        changeOrigin: true,
      },
      '/recommendations': {
        target: TARGET,
        changeOrigin: true,
      },
      '/users': {
        target: TARGET,
        changeOrigin: true,
      },
      '/interactions': {
        target: TARGET,
        changeOrigin: true,
      },
      '/cart': {
        target: TARGET,
        changeOrigin: true,
      },
      '/orders': {
        target: TARGET,
        changeOrigin: true,
      },
      '/checkout': {
        target: TARGET,
        changeOrigin: true,
      },
      '/feedback': {
        target: TARGET,
        changeOrigin: true,
      },
      '/health': {
        target: TARGET,
        changeOrigin: true,
      },
    },
  },
  build: { chunkSizeWarningLimit: 2000 },
});
