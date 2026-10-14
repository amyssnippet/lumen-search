import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "LUMEN | Jewellery Visual Retrieval Engine",
  description: "Enterprise fine jewellery visual similarity engine powered by MobileNetV4 FP32 ONNX and Qdrant HNSW vector search.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-ivory text-charcoal-900 selection:bg-gold/30 selection:text-charcoal-900">
        <Navbar />
        <main className="flex-1 pb-16">{children}</main>
        <footer className="border-t border-charcoal-200/60 bg-ivory-100/60 py-6 text-center text-xs text-charcoal-500">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span className="font-serif">
              ✦ LUMEN Jewellery Visual Retrieval Engine &copy; 2026
            </span>
            <span className="font-mono text-[11px] text-charcoal-400">
              MobileNetV4 Conv-Small FP32 ONNX (1280-dim) • Qdrant Vector Collection
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
