import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";

// Useful-but-not-hardcore: type-aware rules are off (slow, strict); we run the standard
// recommended sets plus React Hooks, and relax the few rules that fight this codebase.
export default tseslint.config(
  { ignores: ["dist", "node_modules", "**/*.config.*"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
      // dynamic JSON (SSE) and RF data casts make `any` occasionally pragmatic
      "@typescript-eslint/no-explicit-any": "off",
      // empty catch is fine for best-effort localStorage (throws in private mode)
      "no-empty": ["error", { allowEmptyCatch: true }],
      // allow intentionally-unused args/vars prefixed with _
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
);
