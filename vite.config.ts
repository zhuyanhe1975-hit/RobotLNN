import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  // Project site: https://zhuyanhe1975-hit.github.io/RobotLNN/
  // pages.yml sets GITHUB_PAGES=true so the public bundle uses /RobotLNN/
  // and never inlines GEMINI_API_KEY. Local dev and the Express server stay at /.
  const pagesBuild = process.env.GITHUB_PAGES === 'true';
  const geminiApiKey = pagesBuild ? '' : (env.GEMINI_API_KEY ?? '');
  return {
    base: pagesBuild ? '/RobotLNN/' : '/',
    plugins: [react(), tailwindcss()],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(geminiApiKey),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
