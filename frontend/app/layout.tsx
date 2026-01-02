import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { ThemeProvider } from "next-themes";
import "./globals.css";
import Sidebar from "@/components/layout/sidebar";
// import { AuthButton } from "@/components/auth/auth-button";
// import { EnvVarWarning } from "@/components/env-var-warning";
import { ThemeSwitcher } from "@/components/layout/theme-switcher";
import { Toaster } from "sonner";
// import { hasEnvVars } from "@/lib/utils";
// import { Suspense } from "react";

const inter = Inter({ subsets: ["latin"] });

const defaultUrl = process.env.VERCEL_URL
  ? `https://${process.env.VERCEL_URL}`
  : "http://localhost:3000";

export const metadata: Metadata = {
  metadataBase: new URL(defaultUrl),
  title: "FLEX-Med Dashboard",
  description: "Federated Learning for Medical Imaging",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="bg-background text-foreground">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex flex-col h-screen bg-background">
            {/* Top Navigation Bar */}
            <nav className="w-full flex justify-center border-b border-b-border bg-card">
              <div className="w-full flex justify-between items-center p-3 px-5 text-sm">
                <div className="flex gap-5 items-center font-semibold">
                  <span>FlexMed Dashboard</span>
                </div>
                <div className="flex items-center gap-4">
                  <ThemeSwitcher />
                  {/* Auth temporarily disabled */}
                  {/* {!hasEnvVars ? (
                    <EnvVarWarning />
                  ) : (
                    <Suspense>
                      <AuthButton />
                    </Suspense>
                  )} */}
                </div>
              </div>
            </nav>

            {/* Main Dashboard Content */}
            <div className="flex flex-1 overflow-hidden">
              <Sidebar />
              <div className="flex-1 overflow-auto">{children}</div>
            </div>
          </div>
          <Toaster position="top-right" />
        </ThemeProvider>
      </body>
    </html>
  );
}
