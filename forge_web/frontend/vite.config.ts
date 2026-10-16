// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Jungle Fruit
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: { port: 5175, proxy: { '/api': { target: 'http://127.0.0.1:8800', changeOrigin: true } } },
})
